"""
speckle_score.py

Purpose
-------
Quantify residual "white snow" (impulse-like bright speckles) in an image.

We want an objective metric to show that cleaning helps, instead of relying only on eyes.

Scientific idea
---------------
Gamma speckles are high-frequency positive outliers (salt noise).
A robust way to detect them:

1) Remove low-frequency structure by subtracting a local median:
     R = I - median_filter(I, win)

   This keeps "spike-like" residuals and suppresses smooth structure.

2) Estimate the typical scale of residuals robustly using MAD:
     MAD = median(|R - median(R)|)
     sigma_robust ~ 1.4826 * MAD

3) Count extreme positive residuals:
     R > n_sigma * sigma_robust

To avoid falsely counting real edges/structure, we exclude the "edge band"
(top q fraction of gradient magnitudes).

Outputs
-------
- Extreme bright residual count per megapixel (MP)
- Percent of finite pixels that are extreme
- Summary comparison for BEFORE vs AFTER images

Interpretation
--------------
- If cleaning works, the extreme count/MP should drop strongly.
- If it does not drop, you are not removing the right artifact class.
- If it drops but your edges get blurred, you over-cleaned (not measured here;
  you check visually or with edge metrics separately).

Typical thresholds
------------------
- win = 5 (or 7) for median filter
- n_sigma = 6 to 10 (start with 8)
- q = 0.995 (exclude top 0.5% gradients)

Usage examples (PowerShell)
---------------------------
pixi run python sanity_check/speckle_score.py `
  --before "D:\...\transmission_corr_filled\TcorrFill_00001.fits" `
  --after  "D:\...\transmission_corr_clean\Tclean_00001.fits"

You can run this on several indices and report mean±std in a paper.
"""

from pathlib import Path
import argparse
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter


def read_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)


def gradient_band_mask(I: np.ndarray, q: float = 0.995) -> np.ndarray:
    """Top (1-q) fraction gradient pixels, used as an "edge exclusion" band."""
    gx = np.zeros_like(I)
    gy = np.zeros_like(I)
    gx[:, 1:] = np.abs(I[:, 1:] - I[:, :-1])
    gy[1:, :] = np.abs(I[1:, :] - I[:-1, :])
    g = gx + gy

    gf = g[np.isfinite(g)]
    if gf.size == 0:
        return np.zeros_like(I, dtype=bool)

    thr = np.quantile(gf, q)
    return np.isfinite(g) & (g >= thr)


def robust_sigma(x: np.ndarray) -> float:
    """Robust sigma estimate via MAD (1.4826*MAD)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def speckle_metrics(I: np.ndarray, *, win: int, n_sigma: float, q: float):
    """
    Compute speckle score on one image.
    Returns dict with counts and rates.
    """
    finite = np.isfinite(I)
    H, W = I.shape

    # Exclude strongest edges to avoid counting real structure as "speckle"
    edge_band = gradient_band_mask(I, q=q)
    eval_mask = finite & (~edge_band)

    # Local-median residual (high-pass but robust)
    medloc = median_filter(I, size=win, mode="reflect")
    R = I - medloc

    sig = robust_sigma(R[eval_mask])
    if not np.isfinite(sig) or sig == 0:
        sig = np.nan

    # Extreme positive spikes (white snow)
    extreme = eval_mask & (R > (n_sigma * sig))

    n_ext = int(np.sum(extreme))
    n_eval = int(np.sum(eval_mask))
    mp = (H * W) / 1e6
    per_mp = n_ext / max(mp, 1e-12)
    pct_eval = 100.0 * n_ext / max(n_eval, 1)

    return {
        "H": H,
        "W": W,
        "eval_pixels": n_eval,
        "extreme_pixels": n_ext,
        "extreme_per_MP": per_mp,
        "extreme_pct_eval": pct_eval,
        "robust_sigma_R": sig,
        "n_sigma": n_sigma,
        "win": win,
        "q": q,
    }


def print_report(name: str, path: Path, m: dict):
    print(f"\n--- {name} ---")
    print("File:", path)
    print(f"Image size: {m['H']} x {m['W']}  (~{(m['H']*m['W']/1e6):.2f} MP)")
    print(f"Edge exclusion: top {(1-m['q'])*100:.2f}% gradients excluded (q={m['q']})")
    print(f"Residual: R = I - median_filter(I, win={m['win']})")
    print(f"Robust sigma(R) ~ {m['robust_sigma_R']:.6g}")
    print(f"Threshold: R > {m['n_sigma']} * sigma(R)")
    print(f"Extreme bright residuals: {m['extreme_pixels']:,}")
    print(f"Extreme per MP: {m['extreme_per_MP']:.2f} / MP")
    print(f"Extreme fraction of evaluated pixels: {m['extreme_pct_eval']:.4f}%")
    print("Interpretation: lower is cleaner (less 'white snow').")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quantify residual bright speckle noise (white snow).")
    ap.add_argument("--before", required=True, help="Path to BEFORE image (e.g., TcorrFill)")
    ap.add_argument("--after", required=True, help="Path to AFTER image (e.g., Tclean)")
    ap.add_argument("--win", type=int, default=5, help="Median-filter window size (odd int). Default=5")
    ap.add_argument("--n_sigma", type=float, default=8.0, help="Threshold in robust sigmas. Default=8")
    ap.add_argument("--q", type=float, default=0.995, help="Gradient quantile to exclude edges. Default=0.995")
    args = ap.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)

    I0 = read_fits(before_path)
    I1 = read_fits(after_path)

    m0 = speckle_metrics(I0, win=args.win, n_sigma=args.n_sigma, q=args.q)
    m1 = speckle_metrics(I1, win=args.win, n_sigma=args.n_sigma, q=args.q)

    print("\n=== Speckle Score (Bright Residuals) ===")
    print("Goal: show that AFTER has far fewer extreme bright residuals than BEFORE.")
    print("If it drops strongly -> cleaning removed impulse-like speckles.")
    print("If it does not drop -> cleaning is not targeting the right artifact class.\n")

    print_report("BEFORE", before_path, m0)
    print_report("AFTER", after_path, m1)

    # Improvement factor
    if m1["extreme_per_MP"] > 0:
        factor = m0["extreme_per_MP"] / m1["extreme_per_MP"]
    else:
        factor = np.inf

    print("\n=== Summary ===")
    print(f"Extreme/MP BEFORE: {m0['extreme_per_MP']:.2f}")
    print(f"Extreme/MP AFTER : {m1['extreme_per_MP']:.2f}")
    print(f"Improvement factor (BEFORE/AFTER): {factor:.2f}x")
    print("\nRule of thumb:")
    print("- >2x drop is noticeable, >5x is strong, >10x is excellent (context-dependent).")
