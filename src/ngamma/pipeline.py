import numpy as np
from tqdm import tqdm

from .io_fits import list_sorted, read_fits, write_fits
from .config import ScanConfig
from .masks import make_detector_stability_mask
from .preprocess import compute_transmission, log_transform
from .detect import gamma_remove_trend_tiled
from .postprocess import fill_nans_nearest
from .spatial_cleanup import spatial_impulse_cleanup



def process_scan(cfg: ScanConfig):

    print("Raw folder:", cfg.raw_folder)
    print("Output folder:", cfg.output_folder.resolve())
    out = cfg.output_folder
    print("Run output folder:", out.resolve())

    raw_files = list_sorted(cfg.raw_folder, f"ct{cfg.exposure_prefix}")
    ob_files  = list_sorted(cfg.raw_folder, f"ob{cfg.exposure_prefix}")
    dc_files  = list_sorted(cfg.raw_folder, f"dc{cfg.exposure_prefix}")

    print("ROI:", cfg.roi)

    print("Number of raw projections:", len(raw_files))
    print("Number of OB frames:", len(ob_files))
    print("Number of DC frames:", len(dc_files))

    if len(raw_files) == 0:
        raise ValueError("No raw projection files found")
    if len(ob_files) == 0 or len(dc_files) == 0:
        raise ValueError("Need OB and DC frames")
    Nsub = 40
    idxs = np.linspace(0, len(raw_files)-1, Nsub, dtype=int)
    raw_files = [raw_files[i] for i in idxs]
    print("Using subset projections:", Nsub)
    print("Selected indices (0-based):", idxs[:10], "...", idxs[-10:])
    print("Selected file examples:", [p.name for p in raw_files[:5]])



    # ---- Load OB/DC stacks
    OB_stack = np.stack([read_fits(p, roi = cfg.roi) for p in ob_files], axis=0)  # (10,H,W)
    DC_stack = np.stack([read_fits(p, roi = cfg.roi) for p in dc_files], axis=0)

    # ---- Build stability mask (starting parameters)
    good_mask, diag = make_detector_stability_mask(
        OB_stack, DC_stack,
        z_reject=6.0,
        den_eps=50.0,
        rstd_ob_max=0.05,
        rstd_dc_max=0.02,
        border_px=64,
    )

    good_frac = 100.0 * float(np.mean(good_mask))
    print(f"Good pixels: {good_frac:.2f}% (masked {100.0-good_frac:.2f}%)")

    # Save diagnostics immediately (so you see outputs even if you interrupt later)
    write_fits(out/ "masks" / "good_mask.fits", good_mask.astype(np.float32))
    write_fits(out / "masks" / "den.fits", diag["den"])
    write_fits(out / "masks" / "ob_mean.fits", diag["OB_mean"])
    write_fits(out / "masks" / "ob_std.fits", diag["OB_std"])

    # ---- Build transmission stack in RAM
    print("Loading and normalizing all projections into transmission stack...")
    N = len(raw_files)
    H, W = diag["OB_mean"].shape
    Tstack = np.empty((N, H, W), dtype=np.float32)

    for i, rp in enumerate(tqdm(raw_files)):
        I = read_fits(rp, roi = cfg.roi)
        T = compute_transmission(I, diag["OB_mean"], diag["DC_mean"], good_mask)
        Tstack[i] = T

    finite = np.isfinite(Tstack).sum()
    nan_count = np.size(Tstack) - finite
    print(f"NaN pixels in Tstack (masked/bad): {nan_count} ({100.0*nan_count/np.size(Tstack):.3f}% of all pixels)")

    # ---- Gamma removal: temporal Hampel (tiled, circular, NaN-safe)
    print("Running temporal Hampel gamma removal...")
    Tcorr, gmask = gamma_remove_trend_tiled(
    Tstack,
    k=4,
    tau_t=6.0,
    s_floor_t=1e-6,
    tile=256,
    spatial_size=9,
    tau_s=6.0,
    s_floor_s=1e-6,
    edge_q=0.99,
    edge_gate=True,
    edge_dilate=3,)

    flagged = int(np.sum(gmask))
    finite2 = int(np.sum(np.isfinite(Tstack)))
    print(f"Flagged pixels: {flagged} ({100.0*flagged/max(finite2,1):.4f}% of finite pixels)")

    # ---- Save corrected transmission/log and gamma masks
    # We save BOTH:
    #  (1) NaN-preserving corrected outputs (scientifically honest)
    #  (2) NaN-filled corrected outputs (for ImageJ / recon compatibility)
    print("Saving corrected transmission/log and gamma masks...")

    for i, rp in enumerate(tqdm(raw_files)):
        idx = rp.name.split("_")[-1]  # "00001.fits"

        # 1) NaN-preserving
        write_fits(out / "transmission_corr" / f"Tcorr_{idx}", Tcorr[i])
        write_fits(out / "log_corr" / f"logcorr_{idx}", log_transform(Tcorr[i]))
        write_fits(out / "masks" / "gamma" / f"gamma_{idx}", gmask[i].astype(np.float32))

        # 2) NaN-filled (for viewing/recon)
        Tfill = fill_nans_nearest(Tcorr[i])

        # 3) Spatial cleanup (presentation/recon quality)
        # Use the SAME good_mask to avoid inventing values in masked detector regions.
        Tclean, smask = spatial_impulse_cleanup(Tfill,good_mask,win=5,tau_s=8.0,grad_q=0.995,max_area=25,)

        write_fits(out / "transmission_corr_filled" / f"TcorrFill_{idx}", Tfill)
        write_fits(out / "log_corr_filled" / f"logcorrFill_{idx}", log_transform(Tfill))

        # Save the cleaned version separately (this is what you show in ImageJ / use for recon)
        write_fits(out / "transmission_corr_clean" / f"Tclean_{idx}", Tclean)
        write_fits(out / "log_corr_clean" / f"logclean_{idx}", log_transform(Tclean))
        write_fits(out / "masks" / "spatial" / f"spatial_{idx}", smask.astype(np.float32))


    print("Done.")
