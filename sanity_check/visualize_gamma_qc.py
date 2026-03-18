from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ============================================================
# USER INPUT
# ============================================================

T_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\transmission_corr\Tcorr_00606.fits"
)

M_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\masks\gamma\gamma_00606.fits"
)

EDGE_Q = 0.995
OUT_PNG = Path("sanity_check/gamma_qc_visual.png")


# ============================================================
# HELPERS
# ============================================================

def read_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)


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


def bounding_box_from_mask(mask: np.ndarray, pad: int = 40):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(0, ys.min() - pad)
    y1 = min(mask.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask.shape[1], xs.max() + pad + 1)
    return y0, y1, x0, x1


# ============================================================
# MAIN
# ============================================================

T = read_fits(T_PATH)
M = read_fits(M_PATH) > 0.5
E = make_edge_band(T, q=EDGE_Q)

finite = np.isfinite(T)

# crop around where detections exist
y0, y1, x0, x1 = bounding_box_from_mask(M, pad=60)

Tcrop = T[y0:y1, x0:x1]
Mcrop = M[y0:y1, x0:x1]
Ecrop = E[y0:y1, x0:x1]

# summary numbers
n_mask = int(np.sum(M & finite))
n_edge_mask = int(np.sum(M & E & finite))
n_nonedge_mask = int(np.sum(M & (~E) & finite))

print("=== Visual Gamma QC Summary ===")
print("T file :", T_PATH)
print("M file :", M_PATH)
print(f"Finite pixels           : {int(np.sum(finite)):,}")
print(f"Flagged mask pixels     : {n_mask:,}")
print(f"Flagged on edge band    : {n_edge_mask:,}")
print(f"Flagged off edge band   : {n_nonedge_mask:,}")

# figure
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 1) raw transmission
im0 = ax[0, 0].imshow(T, cmap="gray")
ax[0, 0].set_title("Transmission")
ax[0, 0].axis("off")
plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

# 2) transmission + gamma mask
ax[0, 1].imshow(T, cmap="gray")
ax[0, 1].imshow(np.where(M, 1.0, np.nan), cmap="autumn", alpha=0.8)
ax[0, 1].set_title("Transmission + Gamma Mask")
ax[0, 1].axis("off")

# 3) transmission + edge band
ax[1, 0].imshow(T, cmap="gray")
ax[1, 0].imshow(np.where(E, 1.0, np.nan), cmap="cool", alpha=0.6)
ax[1, 0].set_title("Transmission + Edge Band")
ax[1, 0].axis("off")

# 4) zoomed crop with both overlays
ax[1, 1].imshow(Tcrop, cmap="gray")
ax[1, 1].imshow(np.where(Ecrop, 1.0, np.nan), cmap="cool", alpha=0.35)
ax[1, 1].imshow(np.where(Mcrop, 1.0, np.nan), cmap="autumn", alpha=0.85)
ax[1, 1].set_title(f"Zoomed Crop: mask + edge\n[y:{y0}:{y1}, x:{x0}:{x1}]")
ax[1, 1].axis("off")

plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.show()

print("")
print("Saved figure to:")
print(OUT_PNG)