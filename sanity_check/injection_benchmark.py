from pathlib import Path
import numpy as np
from astropy.io import fits


# ============================================================
# USER SETTINGS
# ============================================================

T_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260309_153425\transmission_corr\Tcorr_00606.fits"
)

OUT_DIR = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260309_153425\benchmark_masks"
)

EDGE_Q = 0.995          # top 0.5% gradients = edge band
BORDER_EXCLUDE = 20     # ignore outer border pixels from benchmarking
NBINS = 512             # histogram bins for Otsu


# ============================================================
# HELPERS
# ============================================================

def read_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)


def write_fits(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Simple gradient proxy:
        g = |dI/dx| + |dI/dy|
    """
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    gx[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
    gy[1:, :] = np.abs(img[1:, :] - img[:-1, :])

    return gx + gy


def make_edge_band(img: np.ndarray, q: float = 0.995) -> np.ndarray:
    """
    Edge band = top (1-q) fraction of gradient magnitude.
    """
    g = gradient_magnitude(img)
    gf = g[np.isfinite(g)]

    if gf.size == 0:
        return np.zeros_like(img, dtype=bool)

    thr = np.quantile(gf, q)
    return np.isfinite(g) & (g >= thr)


def make_border_mask(shape: tuple[int, int], border: int) -> np.ndarray:
    """
    True inside the usable area, False near the border.
    """
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    m[border:H-border, border:W-border] = True
    return m


def otsu_threshold(values: np.ndarray, nbins: int = 512) -> float:
    """
    Pure NumPy Otsu threshold on 1D finite values.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values available for Otsu threshold.")

    vmin = float(values.min())
    vmax = float(values.max())

    if vmax <= vmin:
        return vmin

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)

    prob = hist / hist.sum()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    omega = np.cumsum(prob)                         # class probabilities
    mu = np.cumsum(prob * bin_centers)             # class means * probs
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan

    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = np.nanargmax(sigma_b2)

    return float(bin_centers[idx])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    T = read_fits(T_PATH)

    finite = np.isfinite(T)
    border_ok = make_border_mask(T.shape, BORDER_EXCLUDE)

    usable = finite & border_ok
    edge = make_edge_band(T, q=EDGE_Q) & usable

    # --------------------------------------------------------
    # OTSU-BASED SAMPLE / BACKGROUND SPLIT
    # --------------------------------------------------------
    Tu = T[usable]
    otsu_thr = otsu_threshold(Tu, nbins=NBINS)

    # High transmission -> likely open/background
    # Low transmission  -> likely sample/material
    sample = usable & (T < otsu_thr)
    background = usable & (T >= otsu_thr)

    # Remove edge band from both, for clean region classes
    background = background & (~edge)
    bulk = sample & (~edge)

    # optional: define "sample_all" too
    sample_all = sample.copy()

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------
    n_total = T.size
    n_finite = int(finite.sum())
    n_usable = int(usable.sum())
    n_edge = int(edge.sum())
    n_sample = int(sample_all.sum())
    n_bg = int(background.sum())
    n_bulk = int(bulk.sum())

    # --------------------------------------------------------
    # SAVE MASKS
    # --------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_fits(OUT_DIR / "edge_mask.fits", edge.astype(np.float32))
    write_fits(OUT_DIR / "sample_mask.fits", sample_all.astype(np.float32))
    write_fits(OUT_DIR / "background_mask.fits", background.astype(np.float32))
    write_fits(OUT_DIR / "bulk_mask.fits", bulk.astype(np.float32))

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------
    print("=== Injection Benchmark: Region Class Setup ===")
    print("Image:", T_PATH)
    print(f"Shape: {T.shape[0]} x {T.shape[1]}")
    print(f"Finite pixels: {n_finite:,} / {n_total:,}")
    print(f"Usable pixels (excluding {BORDER_EXCLUDE}px border): {n_usable:,}")
    print("")
    print(f"Otsu transmission threshold: {otsu_thr:.6f}")
    print("")
    print(f"Edge band pixels   : {n_edge:,}  ({100.0*n_edge/max(n_usable,1):.3f}%)")
    print(f"Sample pixels      : {n_sample:,}  ({100.0*n_sample/max(n_usable,1):.3f}%)")
    print(f"Background pixels  : {n_bg:,}  ({100.0*n_bg/max(n_usable,1):.3f}%)")
    print(f"Bulk pixels        : {n_bulk:,}  ({100.0*n_bulk/max(n_usable,1):.3f}%)")
    print("")
    print("Checks:")
    print(f"edge ∩ background = {int(np.sum(edge & background))}")
    print(f"edge ∩ bulk       = {int(np.sum(edge & bulk))}")
    print(f"background ∩ bulk = {int(np.sum(background & bulk))}")
    print("")
    print("Saved masks to:")
    print(OUT_DIR)
    print("")
    print("Interpretation:")
    print("- Edge band = strongest gradients (where edge-triggering risk is highest).")
    print("- Sample mask = lower-transmission region from Otsu thresholding.")
    print("- Background mask = higher-transmission region from Otsu thresholding.")
    print("- Bulk = sample interior away from strong edges.")
    print("- This is still a benchmark setup step. No injections yet.")