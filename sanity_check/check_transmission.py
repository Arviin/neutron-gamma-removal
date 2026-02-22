from pathlib import Path
import numpy as np
from astropy.io import fits

def read_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)

def edge_mask(shape, edge=64):
    """Boolean mask for an 'edge ring' of width=edge pixels."""
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    m[:edge, :] = True
    m[-edge:, :] = True
    m[:, :edge] = True
    m[:, -edge:] = True
    return m

def stats_for_thresholds(T: np.ndarray, t_max=2.0):
    eps = 1e-6
    finite = np.isfinite(T)
    total = finite.sum()

    out = {}
    out["total_pixels"] = int(total)
    out["min"] = float(np.nanmin(T))
    out["max"] = float(np.nanmax(T))
    out["mean"] = float(np.nanmean(T))
    out["median"] = float(np.nanmedian(T))

    # key diagnostics
    out["pct_T_gt_1.0"]  = 100.0 * float(np.sum((T > 1.0) & finite)) / total
    out["pct_T_gt_1.1"]  = 100.0 * float(np.sum((T > 1.1) & finite)) / total
    out["pct_T_gt_1.2"]  = 100.0 * float(np.sum((T > 1.2) & finite)) / total
    out["pct_T_lt_1e-3"] = 100.0 * float(np.sum((T < 1e-3) & finite)) / total

    # "hitting clip" (exact 2.0 is common if we wrote it that way)
    out["count_T_eq_tmax"] = int(np.sum((T == t_max) & finite))
    out["pct_T_eq_tmax"]   = 100.0 * out["count_T_eq_tmax"] / total

    # near-clip (more robust than equality)
    out["count_T_ge_0p999tmax"] = int(np.sum((T >= 0.999 * t_max) & finite))
    out["pct_T_ge_0p999tmax"]   = 100.0 * out["count_T_ge_0p999tmax"] / total

    return out

def split_center_edge(T: np.ndarray, edge=64, t_max=2.0):
    em = edge_mask(T.shape, edge=edge)
    cm = ~em

    def frac(mask, cond):
        denom = np.sum(mask & np.isfinite(T))
        if denom == 0:
            return 0.0
        return 100.0 * float(np.sum(mask & cond & np.isfinite(T))) / float(denom)

    return {
        "edge_pct_T_ge_0p999tmax": frac(em, T >= 0.999 * t_max),
        "center_pct_T_ge_0p999tmax": frac(cm, T >= 0.999 * t_max),
        "edge_pct_T_gt_1p1": frac(em, T > 1.1),
        "center_pct_T_gt_1p1": frac(cm, T > 1.1),
    }

if __name__ == "__main__":
    # CHANGE THIS to one of your corrected transmission files:
    t_path = Path(r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\transmission_corr\Tcorr_00001.fits")

    T = read_fits(t_path)

    print("File:", t_path)
    s = stats_for_thresholds(T, t_max=2.0)
    for k, v in s.items():
        print(f"{k}: {v}")

    ce = split_center_edge(T, edge=64, t_max=2.0)
    print("\nCenter vs edge (edge ring width = 64 px):")
    for k, v in ce.items():
        print(f"{k}: {v:.6f}%")
