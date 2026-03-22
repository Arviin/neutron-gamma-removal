from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# =========================
# USER INPUT
# =========================

ROI = (106, 1942, 188, 1860)

T_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\transmission\T_00606.fits"
)

TCORR_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\transmission_corr\Tcorr_00606.fits"
)

TCLEAN_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\transmission_corr_clean\Tclean_00606.fits"
)

MASK_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\masks\gamma\gamma_00606.fits"
)

OUT_PNG = Path("sanity_check/cleaning_effect_compare.png")

# =========================
# LOAD
# =========================

def read_fits(p):
    with fits.open(p, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)

T = read_fits(T_PATH)
y0, y1, x0, x1 = ROI
T = T[y0:y1, x0:x1]

Tcorr = read_fits(TCORR_PATH)
Tclean = read_fits(TCLEAN_PATH)
M = read_fits(MASK_PATH) > 0.5

finite_corr = np.isfinite(T) & np.isfinite(Tcorr)
finite_clean = np.isfinite(T) & np.isfinite(Tclean)
finite_both = np.isfinite(Tcorr) & np.isfinite(Tclean) & np.isfinite(T)

diff_corr = T - Tcorr
diff_clean_extra = Tcorr - Tclean
diff_total = T - Tclean
masked_diff_corr = np.where(M, diff_corr, np.nan)

# =========================
# PRINT STATS
# =========================

print("=== Cleaning Effect Comparison ===")
print(f"Mean |T - Tcorr|      : {np.nanmean(np.abs(diff_corr)):.6f}")
print(f"Max  |T - Tcorr|      : {np.nanmax(np.abs(diff_corr)):.6f}")
print(f"Mean |Tcorr - Tclean| : {np.nanmean(np.abs(diff_clean_extra)):.6f}")
print(f"Max  |Tcorr - Tclean| : {np.nanmax(np.abs(diff_clean_extra)):.6f}")
print(f"Mean |T - Tclean|     : {np.nanmean(np.abs(diff_total)):.6f}")
print(f"Max  |T - Tclean|     : {np.nanmax(np.abs(diff_total)):.6f}")
print("")
print(f"Mask pixels           : {np.sum(M)}")
print(f"Changed at mask       : {np.sum((np.abs(diff_corr) > 1e-6) & M)}")
print(f"Changed outside mask  : {np.sum((np.abs(diff_corr) > 1e-6) & (~M))}")

# =========================
# FIGURE PLOTTING
# =========================

fig, ax = plt.subplots(3, 3, figsize=(15, 12))

# row 1
ax[0, 0].imshow(T, cmap="gray")
ax[0, 0].set_title("Original T")
ax[0, 0].axis("off")

ax[0, 1].imshow(Tcorr, cmap="gray")
ax[0, 1].set_title("Corrected Tcorr")
ax[0, 1].axis("off")

ax[0, 2].imshow(Tclean, cmap="gray")
ax[0, 2].set_title("Cleaned Tclean")
ax[0, 2].axis("off")

# row 2
ax[1, 0].imshow(diff_corr, cmap="bwr")
ax[1, 0].set_title("Difference: T - Tcorr")
ax[1, 0].axis("off")

ax[1, 1].imshow(diff_clean_extra, cmap="bwr")
ax[1, 1].set_title("Difference: Tcorr - Tclean")
ax[1, 1].axis("off")

ax[1, 2].imshow(diff_total, cmap="bwr")
ax[1, 2].set_title("Difference: T - Tclean")
ax[1, 2].axis("off")

# row 3
ax[2, 0].imshow(masked_diff_corr, cmap="bwr")
ax[2, 0].set_title("T - Tcorr ONLY at mask")
ax[2, 0].axis("off")

ax[2, 1].hist(T[finite_corr].ravel(), bins=200, alpha=0.5, label="T")
ax[2, 1].hist(Tcorr[finite_corr].ravel(), bins=200, alpha=0.5, label="Tcorr")
ax[2, 1].hist(Tclean[finite_clean].ravel(), bins=200, alpha=0.5, label="Tclean")
ax[2, 1].set_title("Histogram")
ax[2, 1].legend()

h = ax[2, 2].hist2d(
    T[finite_clean].ravel(),
    Tclean[finite_clean].ravel(),
    bins=200
)
ax[2, 2].set_title("Joint T vs Tclean")

plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.show()

print("")
print("Saved figure to:")
print(OUT_PNG)

