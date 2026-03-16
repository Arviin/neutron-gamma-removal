from pathlib import Path
import sys
import numpy as np
from astropy.io import fits

# ------------------------------------------------------------
# Make ngamma importable
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from ngamma.io_fits import list_sorted, read_fits
from ngamma.masks import make_detector_stability_mask
from ngamma.preprocess import compute_transmission
from ngamma.detect import gamma_remove_trend_tiled


# ============================================================
# USER SETTINGS
# ============================================================

RAW_FOLDER = Path(r"D:\mpc3091-Qianru\0_data\01_ballpacking_snr\02_rawdata2\05_ct10s")
EXPOSURE_PREFIX = "10s"

# SAME ROI as current pipeline
ROI = (106, 1942, 188, 1860)

# Benchmark projection
TARGET_IDX = 606   # file *_00606.fits

# Injection amplitudes in transmission units
AMPLITUDES = [0.02, 0.05, 0.10, 0.20]

# Injection shapes to test
SHAPES = ["pixel", "blob2", "blob3"]

# Number of injected events per region per amplitude per shape
N_INJECT = 100

# Region definitions
EDGE_Q = 0.995
BORDER_EXCLUDE = 20
NBINS = 512

# Define "flat/smooth" subregions using LOW local-variability quantiles
BG_GRAD_Q = 0.50
BG_MAD_Q = 0.50
BULK_GRAD_Q = 0.50
BULK_MAD_Q = 0.50

# Local window size for region smoothness definition
LOCAL_SIZE = 9

# Detector parameters (same as current pipeline)
DET_K = 4
DET_TAU_T = 6.0
DET_S_FLOOR_T = 1e-6
DET_TILE = 256
DET_SPATIAL_SIZE = 9
DET_TAU_S = 6.0
DET_S_FLOOR_S = 1e-6
DET_EDGE_Q = 0.99
DET_EDGE_GATE = True
DET_EDGE_DILATE = 3
DET_EDGE_TAU_BOOST = 2.0


# ============================================================
# HELPERS
# ============================================================

def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
    gy[1:, :] = np.abs(img[1:, :] - img[:-1, :])
    return gx + gy


def make_edge_band(img: np.ndarray, q: float = 0.995) -> np.ndarray:
    g = gradient_magnitude(img)
    gf = g[np.isfinite(g)]
    if gf.size == 0:
        return np.zeros_like(img, dtype=bool)
    thr = np.quantile(gf, q)
    return np.isfinite(g) & (g >= thr)


def make_border_mask(shape: tuple[int, int], border: int) -> np.ndarray:
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    m[border:H-border, border:W-border] = True
    return m


def otsu_threshold(values: np.ndarray, nbins: int = 512) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values for Otsu threshold.")

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return vmin

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)

    prob = hist / hist.sum()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan

    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = np.nanargmax(sigma_b2)
    return float(bin_centers[idx])


def nan_local_median_mad_2d(img: np.ndarray, size: int, eps: float = 1e-6):
    """
    NaN-aware local median and MAD on a single 2D image.
    Used ONLY to define smooth/flat benchmark regions.
    """
    if size % 2 == 0 or size < 3:
        raise ValueError("size must be odd and >= 3")

    from numpy.lib.stride_tricks import sliding_window_view

    pad = size // 2
    img_pad = np.pad(img, pad_width=pad, mode="constant", constant_values=np.nan)
    Wv = sliding_window_view(img_pad, (size, size))
    Wf = Wv.reshape(Wv.shape[0], Wv.shape[1], -1)

    med = np.nanmedian(Wf, axis=-1)
    mad = np.nanmedian(np.abs(Wf - med[..., None]), axis=-1)
    madn = 1.4826 * mad + eps

    return med.astype(np.float32), madn.astype(np.float32)


def qthr(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, q))


def sample_centers(mask: np.ndarray, n: int, rng: np.random.Generator, margin: int = 2):
    """
    Sample center coordinates safely away from image border so blobs fit.
    """
    H, W = mask.shape
    safe = mask.copy()
    safe[:margin, :] = False
    safe[-margin:, :] = False
    safe[:, :margin] = False
    safe[:, -margin:] = False

    ys, xs = np.where(safe)
    if len(ys) < n:
        raise ValueError(
            f"Mask has only {len(ys)} safe pixels, cannot sample {n}. "
            f"Reduce N_INJECT or relax region criteria."
        )

    choose = rng.choice(len(ys), size=n, replace=False)
    return list(zip(ys[choose], xs[choose]))


def inject_shape(img: np.ndarray, y: int, x: int, amp: float, shape: str):
    """
    Add a synthetic positive impulse to img in-place.
    """
    if shape == "pixel":
        img[y, x] += amp

    elif shape == "blob2":
        img[y:y+2, x:x+2] += amp

    elif shape == "blob3":
        img[y-1:y+2, x-1:x+2] += amp

    else:
        raise ValueError(f"Unknown shape: {shape}")


def support_mask_from_centers(shape_hw: tuple[int, int], centers: list[tuple[int, int]], shape: str) -> np.ndarray:
    """
    Build a binary mask of injected support.
    """
    H, W = shape_hw
    m = np.zeros((H, W), dtype=bool)

    for y, x in centers:
        if shape == "pixel":
            if 0 <= y < H and 0 <= x < W:
                m[y, x] = True

        elif shape == "blob2":
            y1, y2 = max(0, y), min(H, y+2)
            x1, x2 = max(0, x), min(W, x+2)
            m[y1:y2, x1:x2] = True

        elif shape == "blob3":
            y1, y2 = max(0, y-1), min(H, y+2)
            x1, x2 = max(0, x-1), min(W, x+2)
            m[y1:y2, x1:x2] = True

        else:
            raise ValueError(f"Unknown shape: {shape}")

    return m


def dilate_binary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Simple square dilation using NumPy only.
    """
    if radius <= 0:
        return mask.copy()

    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        y1 = max(0, y - radius)
        y2 = min(H, y + radius + 1)
        x1 = max(0, x - radius)
        x2 = min(W, x + radius + 1)
        out[y1:y2, x1:x2] = True

    return out


def support_recall(pred_mask: np.ndarray, inj_support: np.ndarray, dilate_radius: int = 1) -> float:
    """
    Event-level recall:
    count an injected event as recovered if prediction overlaps a dilated support.
    For simplicity with many injections, this computes support overlap fraction.
    """
    supp = dilate_binary(inj_support, radius=dilate_radius)
    n = int(inj_support.sum())
    if n == 0:
        return 0.0

    # fraction of injected support pixels "covered" by prediction in a tolerant way
    covered = np.sum(inj_support & supp & dilate_binary(pred_mask, radius=dilate_radius))
    return float(covered) / float(n)


# ============================================================
# LOAD STACK
# ============================================================

def build_stack():
    raw_files = list_sorted(RAW_FOLDER, f"ct{EXPOSURE_PREFIX}")
    ob_files  = list_sorted(RAW_FOLDER, f"ob{EXPOSURE_PREFIX}")
    dc_files  = list_sorted(RAW_FOLDER, f"dc{EXPOSURE_PREFIX}")

    if len(raw_files) == 0:
        raise ValueError("No raw projections found.")
    if len(ob_files) == 0 or len(dc_files) == 0:
        raise ValueError("Need OB and DC files.")

    # same subset strategy as current pipeline
    Nsub = 40
    idxs = np.linspace(0, len(raw_files) - 1, Nsub, dtype=int)
    raw_files = [raw_files[i] for i in idxs]

    names = [p.name for p in raw_files]

    target_name = f"ct{EXPOSURE_PREFIX}_{TARGET_IDX:05d}.fits"
    if target_name not in names:
        raise ValueError(
            f"Target file {target_name} not in current subset.\n"
            f"Subset start/end examples: {names[:5]} ... {names[-5:]}"
        )
    target_pos = names.index(target_name)

    OB_stack = np.stack([read_fits(p, roi=ROI) for p in ob_files], axis=0)
    DC_stack = np.stack([read_fits(p, roi=ROI) for p in dc_files], axis=0)

    good_mask, diag = make_detector_stability_mask(
        OB_stack, DC_stack,
        z_reject=6.0,
        den_eps=50.0,
        rstd_ob_max=0.05,
        rstd_dc_max=0.02,
        border_px=64,
    )

    H, W = diag["OB_mean"].shape
    Tstack = np.empty((len(raw_files), H, W), dtype=np.float32)

    for i, rp in enumerate(raw_files):
        I = read_fits(rp, roi=ROI)
        Tstack[i] = compute_transmission(I, diag["OB_mean"], diag["DC_mean"], good_mask)

    return Tstack, good_mask, target_pos, names[target_pos]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    Tstack, good_mask, target_pos, target_name = build_stack()
    Tref = Tstack[target_pos].copy()

    finite = np.isfinite(Tref)
    border_ok = make_border_mask(Tref.shape, BORDER_EXCLUDE)
    usable = finite & border_ok & good_mask

    # --------------------------------------------------------
    # Base segmentation: sample / background / edge
    # --------------------------------------------------------
    edge = make_edge_band(Tref, q=EDGE_Q) & usable
    grad = gradient_magnitude(Tref)
    _, local_mad = nan_local_median_mad_2d(Tref, size=LOCAL_SIZE, eps=1e-6)

    otsu_thr = otsu_threshold(Tref[usable], nbins=NBINS)

    sample = usable & (Tref < otsu_thr)
    background = usable & (Tref >= otsu_thr)

    edge_sample = edge & sample
    nonedge_bg = background & (~edge)
    nonedge_bulk = sample & (~edge)

    # --------------------------------------------------------
    # Refined benchmark regions
    # --------------------------------------------------------
    bg_grad_thr = qthr(grad[nonedge_bg], BG_GRAD_Q)
    bg_mad_thr  = qthr(local_mad[nonedge_bg], BG_MAD_Q)

    bulk_grad_thr = qthr(grad[nonedge_bulk], BULK_GRAD_Q)
    bulk_mad_thr  = qthr(local_mad[nonedge_bulk], BULK_MAD_Q)

    flat_background = (
        nonedge_bg
        & (grad <= bg_grad_thr)
        & (local_mad <= bg_mad_thr)
    )

    smooth_bulk = (
        nonedge_bulk
        & (grad <= bulk_grad_thr)
        & (local_mad <= bulk_mad_thr)
    )

    region_masks = {
        "flat_background": flat_background,
        "smooth_bulk": smooth_bulk,
        "edge_sample": edge_sample,
    }

    print("=== Stack Injection Benchmark Setup ===")
    print("Target file:", target_name)
    print("Target stack position:", target_pos)
    print("ROI:", ROI)
    print(f"Otsu threshold: {otsu_thr:.6f}")
    print("")
    print(f"Usable pixels        : {int(usable.sum()):,}")
    print(f"Background pixels    : {int(background.sum()):,}")
    print(f"Sample pixels        : {int(sample.sum()):,}")
    print(f"Non-edge background  : {int(nonedge_bg.sum()):,}")
    print(f"Non-edge bulk        : {int(nonedge_bulk.sum()):,}")
    print(f"Edge-sample pixels   : {int(edge_sample.sum()):,}")
    print("")
    print("Refined benchmark regions:")
    print(f"flat_background px   : {int(flat_background.sum()):,}")
    print(f"smooth_bulk px       : {int(smooth_bulk.sum()):,}")
    print(f"edge_sample px       : {int(edge_sample.sum()):,}")
    print("")
    print("Region thresholds:")
    print(f"BG grad thr          : {bg_grad_thr:.6g}")
    print(f"BG local MAD thr     : {bg_mad_thr:.6g}")
    print(f"Bulk grad thr        : {bulk_grad_thr:.6g}")
    print(f"Bulk local MAD thr   : {bulk_mad_thr:.6g}")
    print("")

    # --------------------------------------------------------
    # Benchmark loop
    # --------------------------------------------------------
    for shape_name in SHAPES:
        print(f"================ SHAPE = {shape_name} ================")

        for amp in AMPLITUDES:
            print(f"--- Amplitude = {amp:.4f} ---")

            for region_name, region_mask in region_masks.items():
                centers = sample_centers(region_mask, N_INJECT, rng, margin=2)

                Tinj = Tstack.copy()

                # inject only into target projection
                for y, x in centers:
                    if np.isfinite(Tinj[target_pos, y, x]):
                        inject_shape(Tinj[target_pos], y, x, amp, shape_name)

                inj_support = support_mask_from_centers(
                    Tinj[target_pos].shape,
                    centers,
                    shape_name,
                )

                _, gmask, debug = gamma_remove_trend_tiled(
                    Tinj,
                    k=DET_K,
                    tau_t=DET_TAU_T,
                    s_floor_t=DET_S_FLOOR_T,
                    tile=DET_TILE,
                    spatial_size=DET_SPATIAL_SIZE,
                    tau_s=DET_TAU_S,
                    s_floor_s=DET_S_FLOOR_S,
                    edge_q=DET_EDGE_Q,
                    edge_gate=DET_EDGE_GATE,
                    edge_dilate=DET_EDGE_DILATE,
                    edge_tau_boost=DET_EDGE_TAU_BOOST,
                    return_debug=True,
                )

                temporal_pred = debug["temporal_mask"][target_pos]
                spatial_pred  = debug["spatial_mask"][target_pos]
                final_pred    = debug["final_mask"][target_pos]

                rec_temporal = support_recall(temporal_pred, inj_support, dilate_radius=1)
                rec_spatial  = support_recall(spatial_pred, inj_support, dilate_radius=1)
                rec_final    = support_recall(final_pred, inj_support, dilate_radius=1)

                print(
                    f"{region_name:16s}  "
                    f"temporal={rec_temporal:.3f}  "
                    f"spatial={rec_spatial:.3f}  "
                    f"final={rec_final:.3f}"
                )

            print("")