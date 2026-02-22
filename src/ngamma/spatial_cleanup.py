import numpy as np
from scipy.ndimage import median_filter, label, generate_binary_structure


def _gradient_band_mask(img: np.ndarray, good: np.ndarray, q: float = 0.995):
    """
    Mask pixels in the top q quantile of gradient magnitude (proxy for edges).
    Excludes NaNs and uses only 'good' pixels for threshold.
    """
    img = img.astype(np.float32, copy=False)

    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
    gy[1:, :] = np.abs(img[1:, :] - img[:-1, :])
    g = gx + gy

    m = good & np.isfinite(g)
    if np.count_nonzero(m) == 0:
        return np.zeros_like(img, dtype=bool)

    thr = np.quantile(g[m], q)
    return m & (g >= thr)


def spatial_impulse_cleanup(
    img: np.ndarray,
    good_mask: np.ndarray,
    *,
    win: int = 5,            # local window for median/MAD
    tau_s: float = 8.0,      # spatial spike threshold (z-score)
    grad_q: float = 0.995,   # reject top 0.5% gradient pixels (edges)
    max_area: int = 25,      # keep only small connected components
    eps: float = 1e-6,
):
    """
    Second-pass cleanup for residual gamma 'snow' + small streaks.

    Steps:
      1) local median (robust baseline)
      2) local MAD (robust scale)
      3) spike candidates where (img - med)/(1.4826*MAD+eps) > tau_s
      4) reject candidates near strong gradients (edge band)
      5) keep only small connected components (<= max_area)
      6) replace those pixels with local median

    Returns:
      img_clean, mask_final
    """
    if win % 2 == 0 or win < 3:
        raise ValueError("win must be odd and >= 3")

    img = img.astype(np.float32, copy=True)
    good = good_mask & np.isfinite(img)

    # Robust baseline/scale
    med = median_filter(img, size=win, mode="reflect")
    resid = img - med
    mad = median_filter(np.abs(resid), size=win, mode="reflect")
    s = 1.4826 * mad + eps

    z = resid / s

    # candidates: positive spikes only (gamma hits typically add counts -> higher T)
    cand = good & (z > tau_s)

    # reject strong gradients (protect real edges)
    edgeband = _gradient_band_mask(img, good, q=grad_q)
    cand = cand & (~edgeband)

    # connected components (keep only small blobs)
    st = generate_binary_structure(2, 2)  # 8-connectivity
    lab, nlab = label(cand, structure=st)
    if nlab == 0:
        return img, cand

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
    keep_labels = np.where(sizes <= max_area)[0]
    keep = np.isin(lab, keep_labels)

    # inpaint using local median
    img[keep] = med[keep]

    return img, keep
