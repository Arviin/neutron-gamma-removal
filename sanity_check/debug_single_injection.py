from pathlib import Path
import sys
import numpy as np
from astropy.io import fits

# ------------------------------------------------------------
# make ngamma importable
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from ngamma.io_fits import list_sorted, read_fits
from ngamma.masks import make_detector_stability_mask
from ngamma.preprocess import compute_transmission
from ngamma.detect import gamma_remove_trend_tiled


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

RAW_FOLDER = Path(r"D:\mpc3091-Qianru\0_data\01_ballpacking_snr\02_rawdata2\05_ct10s")
EXPOSURE_PREFIX = "10s"

ROI = (106, 1942, 188, 1860)

TARGET_IDX = 606

AMPLITUDE = 0.20

OUT_DIR = Path(r"debug_injection_outputs")


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def write_fits(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)


# ------------------------------------------------------------
# build stack exactly like pipeline
# ------------------------------------------------------------

raw_files = list_sorted(RAW_FOLDER, f"ct{EXPOSURE_PREFIX}")
ob_files  = list_sorted(RAW_FOLDER, f"ob{EXPOSURE_PREFIX}")
dc_files  = list_sorted(RAW_FOLDER, f"dc{EXPOSURE_PREFIX}")

Nsub = 40
idxs = np.linspace(0, len(raw_files)-1, Nsub, dtype=int)
raw_files = [raw_files[i] for i in idxs]

names = [p.name for p in raw_files]

target_name = f"ct{EXPOSURE_PREFIX}_{TARGET_IDX:05d}.fits"
target_pos = names.index(target_name)

print("Target file:", target_name)
print("Stack index:", target_pos)

OB_stack = np.stack([read_fits(p, roi=ROI) for p in ob_files], axis=0)
DC_stack = np.stack([read_fits(p, roi=ROI) for p in dc_files], axis=0)

good_mask, diag = make_detector_stability_mask(
    OB_stack,
    DC_stack,
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

    Tstack[i] = compute_transmission(
        I,
        diag["OB_mean"],
        diag["DC_mean"],
        good_mask,
    )


# ------------------------------------------------------------
# choose safe background pixel
# ------------------------------------------------------------

Tref = Tstack[target_pos]

finite = np.isfinite(Tref) & good_mask

ys, xs = np.where(finite)

rng = np.random.default_rng(0)

k = rng.integers(len(ys))

y0 = int(ys[k])
x0 = int(xs[k])

print("Injected pixel:", y0, x0)

print("Original T:", Tref[y0, x0])


# ------------------------------------------------------------
# inject event
# ------------------------------------------------------------

Tinj = Tstack.copy()

Tinj[target_pos, y0, x0] += AMPLITUDE

print("Injected T:", Tinj[target_pos, y0, x0])


# ------------------------------------------------------------
# run detector
# ------------------------------------------------------------

Tcorr, gmask, debug = gamma_remove_trend_tiled(
    Tinj,
    k=4,
    tau_t=6.0,
    s_floor_t=1e-6,
    tile=256,
    spatial_size=9,
    tau_s=6.0,
    s_floor_s=1e-6,
    edge_q=0.99,
    edge_gate=True,
    edge_dilate=3,
    return_debug=True,
)

pred = gmask[target_pos]
temp_pred = debug["temporal_mask"][target_pos]
spat_pred = debug["spatial_mask"][target_pos]

print("Temporal mask at pixel:", temp_pred[y0, x0])
print("Spatial-confirmed mask at pixel:", spat_pred[y0, x0])
print("Final mask at pixel:", pred[y0, x0])

# ------------------------------------------------------------
# save debug outputs
# ------------------------------------------------------------

write_fits(OUT_DIR / "original_projection.fits", Tref)
write_fits(OUT_DIR / "injected_projection.fits", Tinj[target_pos])
write_fits(OUT_DIR / "detector_mask.fits", pred.astype(np.float32))

write_fits(OUT_DIR / "temporal_mask.fits", temp_pred.astype(np.float32))
write_fits(OUT_DIR / "spatial_mask.fits", spat_pred.astype(np.float32))
write_fits(OUT_DIR / "final_mask.fits", pred.astype(np.float32))


print("Saved debug outputs to:", OUT_DIR)