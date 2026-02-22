import numpy as np

def robust_mean_std(stack: np.ndarray, z_thresh: float = 6.0, eps: float = 1e-6):
    """
    Robust per-pixel mean/std with MAD-based frame rejection.
    stack: (N,H,W) float
    Returns: mean(H,W), std(H,W), valid_count(H,W)
    """
    stack = np.asarray(stack, dtype=np.float32)
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0)
    scale = 1.4826 * mad + eps

    z = (stack - med) / scale
    keep = np.abs(z) < z_thresh  # (N,H,W)

    valid = np.sum(keep, axis=0).astype(np.int32)
    valid_safe = np.where(valid == 0, 1, valid)

    mean = np.sum(stack * keep, axis=0) / valid_safe

    # robust std computed on kept frames
    diff2 = (stack - mean) ** 2
    var = np.sum(diff2 * keep, axis=0) / np.maximum(valid_safe - 1, 1)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32), valid

def make_detector_stability_mask(
    OB_stack: np.ndarray,
    DC_stack: np.ndarray,
    *,
    z_reject: float = 6.0,
    den_eps: float = 50.0,
    rstd_ob_max: float = 0.05,
    rstd_dc_max: float = 0.02,
    border_px: int = 0,
):
    """
    Build a boolean mask of RELIABLE pixels (True = good).

    Parameters (defaults are starting points; we will tune with QC):
    - den_eps: minimum acceptable (OB-DC) in raw counts.
    - rstd_ob_max: max relative std in OB: std_OB / mean_OB.
    - rstd_dc_max: max DC std relative to mean OB: std_DC / mean_OB.
    - border_px: optionally exclude outer ring.
    """
    OBm, OBs, OBn = robust_mean_std(OB_stack, z_thresh=z_reject)
    DCm, DCs, DCn = robust_mean_std(DC_stack, z_thresh=z_reject)

    den = OBm - DCm

    # Conditions for "bad"
    bad_den = den <= den_eps
    bad_ob  = (OBs / (np.abs(OBm) + 1e-6)) > rstd_ob_max
    bad_dc  = (DCs / (np.abs(OBm) + 1e-6)) > rstd_dc_max

    bad = bad_den | bad_ob | bad_dc

    if border_px and border_px > 0:
        H, W = OBm.shape
        border = np.zeros((H, W), dtype=bool)
        b = int(border_px)
        border[:b, :] = True
        border[-b:, :] = True
        border[:, :b] = True
        border[:, -b:] = True
        bad = bad | border

    good = ~bad
    diagnostics = {
        "OB_mean": OBm, "OB_std": OBs,
        "DC_mean": DCm, "DC_std": DCs,
        "den": den,
        "bad_den": bad_den, "bad_ob": bad_ob, "bad_dc": bad_dc,
        "good": good,
    }
    return good, diagnostics
