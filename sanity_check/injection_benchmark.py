from pathlib import Path
import numpy as np
from astropy.io import fits


# ============================================================
# USER SETTINGS
# ============================================================

T_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260309_153425\transmission_corr\Tcorr_00606.fits"
)

EDGE_Q = 0.995          # top 0.5% gradients = edge band
BORDER_EXCLUDE = 20     # ignore outer border pixels from benchmarking


# ============================================================
# HELPERS
# ============================================================

def read_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)


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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    T = read_fits(T_PATH)

    finite = np.isfinite(T)
    border_ok = make_border_mask(T.shape, BORDER_EXCLUDE)

    usable = finite & border_ok
    edge = make_edge_band(T, q=EDGE_Q) & usable

    # For now:
    # - background = usable pixels below median intensity
    # - bulk       = usable pixels above median intensity but not in edge band
    #
    # This is a temporary first split.
    # Later we may improve "background vs sample interior" using masks or thresholding.
    Tf = T[usable]
    med_T = np.median(Tf)

    background = usable & (T <= med_T) & (~edge)
    bulk = usable & (T > med_T) & (~edge)

    n_total = T.size
    n_finite = int(finite.sum())
    n_usable = int(usable.sum())
    n_edge = int(edge.sum())
    n_bg = int(background.sum())
    n_bulk = int(bulk.sum())

    print("=== Injection Benchmark: Region Class Setup ===")
    print("Image:", T_PATH)
    print(f"Shape: {T.shape[0]} x {T.shape[1]}")
    print(f"Finite pixels: {n_finite:,} / {n_total:,}")
    print(f"Usable pixels (excluding {BORDER_EXCLUDE}px border): {n_usable:,}")
    print("")
    print(f"Edge band pixels   : {n_edge:,}  ({100.0*n_edge/max(n_usable,1):.3f}%)")
    print(f"Background pixels  : {n_bg:,}  ({100.0*n_bg/max(n_usable,1):.3f}%)")
    print(f"Bulk pixels        : {n_bulk:,}  ({100.0*n_bulk/max(n_usable,1):.3f}%)")
    print("")
    print("Checks:")
    print(f"edge ∩ background = {int(np.sum(edge & background))}")
    print(f"edge ∩ bulk       = {int(np.sum(edge & bulk))}")
    print(f"background ∩ bulk = {int(np.sum(background & bulk))}")
    print("")
    print("Interpretation:")
    print("- Edge band = strongest gradients (where edge-triggering risk is highest).")
    print("- Background = lower-intensity usable region, provisional proxy for open/background area.")
    print("- Bulk = higher-intensity usable region away from strong edges.")
    print("- This is only Step 1 of the benchmark. No injections yet.")