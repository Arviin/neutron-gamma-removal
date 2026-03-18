from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# =========================
# USER INPUT
# =========================

T_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\transmission\T_00606.fits"
)

TCORR_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\transmission_corr\Tcorr_00606.fits"
)

MASK_PATH = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260318_113803\masks\gamma\gamma_00606.fits"
)


# =========================
# LOAD
# =========================

def read_fits(p):
    with fits.open(p, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)


T = read_fits(T_PATH)
Tc = read_fits(TCORR_PATH)
M = read_fits(MASK_PATH) > 0.5

finite = np.isfinite(T) & np.isfinite(Tc)

diff = T - Tc


# =========================
# PRINT STATS
# =========================

print("=== Cleaning Effect Summary ===")
print(f"Mean |T - Tcorr| : {np.nanmean(np.abs(diff)):.6f}")
print(f"Max  |T - Tcorr| : {np.nanmax(np.abs(diff)):.6f}")
print(f"Changed pixels   : {np.sum(np.abs(diff) > 1e-6)}")

print("")
print("Mask consistency:")
print(f"Mask pixels      : {np.sum(M)}")
print(f"Changed & mask   : {np.sum((np.abs(diff) > 1e-6) & M)}")
print(f"Changed & NOT mask: {np.sum((np.abs(diff) > 1e-6) & (~M))}")


# =========================
# FIGURES
# =========================

fig, ax = plt.subplots(2, 3, figsize=(14, 8))

# original
im0 = ax[0, 0].imshow(T, cmap="gray")
ax[0, 0].set_title("Original T")
ax[0, 0].axis("off")

# corrected
im1 = ax[0, 1].imshow(Tc, cmap="gray")
ax[0, 1].set_title("Corrected Tcorr")
ax[0, 1].axis("off")

# difference
im2 = ax[0, 2].imshow(diff, cmap="bwr")
ax[0, 2].set_title("Difference (T - Tcorr)")
ax[0, 2].axis("off")

# difference + mask
ax[1, 0].imshow(diff, cmap="bwr")
ax[1, 0].imshow(np.where(M, 1.0, np.nan), cmap="autumn", alpha=0.6)
ax[1, 0].set_title("Difference + Mask")
ax[1, 0].axis("off")

# histogram
ax[1, 1].hist(T[finite].ravel(), bins=200, alpha=0.5, label="T")
ax[1, 1].hist(Tc[finite].ravel(), bins=200, alpha=0.5, label="Tcorr")
ax[1, 1].set_title("Histogram")
ax[1, 1].legend()

# joint
ax[1, 2].hist2d(
    T[finite].ravel(),
    Tc[finite].ravel(),
    bins=200,
)
ax[1, 2].set_title("Joint T vs Tcorr")

plt.tight_layout()
plt.show()