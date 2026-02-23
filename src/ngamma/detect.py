from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import warnings
from scipy.ndimage import uniform_filter
from scipy.ndimage import uniform_filter, binary_dilation
from scipy.ndimage import label, generate_binary_structure


def _keep_small_components(mask2d: np.ndarray, max_area: int, connectivity: int = 2) -> np.ndarray:
    """
    Keep only connected components with area <= max_area.
    connectivity=2 -> 8-connected in 2D.
    """
    if max_area <= 0 or not mask2d.any():
        return mask2d

    st = generate_binary_structure(2, connectivity)
    lab, nlab = label(mask2d, structure=st)
    if nlab == 0:
        return mask2d

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep_labels = np.where(sizes <= max_area)[0]
    return np.isin(lab, keep_labels)



def _edge_band_mask(img: np.ndarray, q: float = 0.995) -> np.ndarray:
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
    gy[1:, :] = np.abs(img[1:, :] - img[:-1, :])
    g = gx + gy
    gf = g[np.isfinite(g)]
    if gf.size == 0:
        return np.zeros_like(img, dtype=bool)
    thr = np.quantile(gf, q)
    return np.isfinite(g) & (g >= thr)



def _nan_aware_local_mean_std(img: np.ndarray, size: int, eps: float = 1e-6):
    """
    NaN-aware local mean and std using weighted uniform filters.
    img: (H,W) float32 with NaNs
    Returns (mu, sigma) float32 arrays
    """
    finite = np.isfinite(img)
    w = uniform_filter(finite.astype(np.float32), size=size, mode="reflect")

    x = np.where(finite, img, 0.0).astype(np.float32)
    mu = uniform_filter(x, size=size, mode="reflect") / np.maximum(w, eps)

    x2 = np.where(finite, img * img, 0.0).astype(np.float32)
    mu2 = uniform_filter(x2, size=size, mode="reflect") / np.maximum(w, eps)

    var = np.maximum(mu2 - mu * mu, 0.0)
    sigma = np.sqrt(var).astype(np.float32)
    return mu.astype(np.float32), sigma


def gamma_remove_hampel_tiled(
    T: np.ndarray,
    *,
    k: int = 4,                 # temporal window radius; window size = 2k+1
    tau_t: float = 6.0,         # temporal robust z threshold (Hampel)
    s_floor_t: float = 1e-6,    # temporal scale floor
    tile: int = 256,            # 256 reduces overhead vs 128
    # spatial confirmation:
    spatial_size: int = 9,      # neighborhood (odd), e.g. 7/9/11
    tau_s: float = 6.0,         # spatial z threshold (spike-likeness)
    s_floor_s: float = 1e-6,    # spatial std floor
    # edge gating:
    edge_q: float = 0.99,      # edge band = top (1-edge_q) gradients
    edge_gate: bool = True,     # if True: forbid temporal gamma detection on edges
    edge_dilate: int = 3,   # grow edge band by this many pixels
    cc_max_area: int = 12,     # keep only small connected components
    cc_connectivity: int = 2,  # 2 -> 8-connectivity in 2D

    
) -> tuple[np.ndarray, np.ndarray]:

    """
    Gamma removal = temporal Hampel (angle-consistency) + spatial spike confirmation.

    Scientific intent:
    - Temporal Hampel detects impulses in angle series (per detector pixel).
    - Spatial confirmation rejects false positives near true object edges
      (measured by your 'gradient band' QC).
    - NaN-safe: bad detector pixels remain NaN and do not contaminate statistics.
    - Circular wrap in angle avoids boundary artifacts in tomography.

    Returns:
      Tcorr: corrected transmission
      gmask: final gamma mask (after spatial confirmation)
    """
    if T.ndim != 3:
        raise ValueError("T must have shape (N,H,W)")

    if spatial_size % 2 == 0 or spatial_size < 3:
        raise ValueError("spatial_size must be odd and >= 3")

    N, H, W = T.shape
    win = 2 * k + 1

    # Circular padding for windows along projection axis
    Tpad = np.concatenate([T[-k:], T, T[:k]], axis=0)  # (N+2k,H,W)

    Tcorr = T.copy()
    gmask = np.zeros((N, H, W), dtype=bool)

    y_starts = list(range(0, H, tile))
    x_starts = list(range(0, W, tile))
    total_tiles = len(y_starts) * len(x_starts)

    # expected warnings for fully-NaN pixels in a window
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")

        with tqdm(total=total_tiles, desc="Hampel+spatial tiles", unit="tile") as pbar:
            for y0 in y_starts:
                y1 = min(H, y0 + tile)
                for x0 in x_starts:
                    x1 = min(W, x0 + tile)

                    Tp = Tpad[:, y0:y1, x0:x1]    # (N+2k, th, tw)
                    Tt = T[:,   y0:y1, x0:x1]     # (N,   th, tw)

                    # Sliding windows along axis 0 -> shape (N, th, tw, win)
                    Wv = sliding_window_view(Tp, window_shape=win, axis=0)

                    # Temporal robust center/scale
                    med_t = np.nanmedian(Wv, axis=-1)  # (N, th, tw)
                    mad_t = np.nanmedian(np.abs(Wv - med_t[..., None]), axis=-1)

                    s_t = 1.4826 * mad_t + s_floor_t
                    z_t = (Tt - med_t) / s_t

                    finite_t = np.isfinite(Tt) & np.isfinite(med_t) & np.isfinite(s_t)
                    m = finite_t & (z_t > tau_t)  # temporal candidates

                    # Spatial confirmation (NaN-aware local z) per projection slice
                    # This is the key fix for your "gradient band" failure mode.
                    if np.any(m):
                        m2 = np.zeros_like(m, dtype=bool)
                        for i in range(N):
                            if not m[i].any():
                                continue
                            img = Tt[i]
                            mu_s, sig_s = _nan_aware_local_mean_std(img, size=spatial_size, eps=1e-6)
                            z_s = (img - mu_s) / (sig_s + s_floor_s)
                            ok = np.isfinite(z_s) & (z_s > tau_s)
                            m2[i] = m[i] & ok
                        m = m2
                    if edge_gate and np.any(m):
                        for i in range(N):
                            if not m[i].any():
                                continue
                            eb = _edge_band_mask(Tt[i],  q = edge_q)
                            if edge_dilate > 0:
                                eb = binary_dilation(eb, iterations=edge_dilate)
                            m[i] = m[i]  & (~eb)
                    # ---- Connected-component pruning: reject edge-like chains
                    if np.any(m) and cc_max_area > 0 :
                        m3 = np.zeros_like(m, dtype=bool)
                        for i in range(N):
                            if not m[i].any():
                                continue
                            m3[i] = _keep_small_components(m[i], max_area= cc_max_area, connectivity=cc_connectivity)
  

                    # store final mask
                    gmask[:, y0:y1, x0:x1] = m

                    # replace with temporal median (robust inpainting in angle)
                    Tc = Tt.copy()
                    Tc[m] = med_t[m]
                    Tcorr[:, y0:y1, x0:x1] = Tc

                    pbar.update(1)

    return Tcorr.astype(np.float32), gmask
