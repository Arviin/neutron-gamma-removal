"""
check_gamma_mask.py

Purpose
-------
This script answers a specific (and very important) scientific question:

  "Is my mask (gamma mask or spatial-cleanup mask) biased toward real edges
   (high-gradient regions), or is it more uniformly distributed like true
   impulse noise would be?"

Why this matters
----------------
1) Gamma/impulse hits are (ideally) *not* correlated with object edges.
   They are sparse, high-frequency events that should appear roughly
   uniformly across the detector (modulo shielding, beam profile, etc.).

2) If a "gamma mask" is heavily concentrated on high-gradient pixels,
   then the detector is probably firing on *real structure* (moving edges,
   partial-volume effects, registration drift, or reconstruction-like effects).
   This is scientifically dangerous because it means you may be modifying
   real line-integral information at boundaries.

3) For the *spatial cleanup mask*, some mild correlation with edges can happen
   if your edge-protection is too weak, but a strong edge bias means the cleanup
   is eroding real structure, not just removing impulse speckles.

What it does
------------
Given:
  --T : a transmission-like image (T, Tcorr, or Tclean)
  --M : a mask image (gamma mask or spatial mask; nonzero = flagged)

It computes:
1) Overall flagged fraction among finite pixels.
2) An "edge band" defined as the top (1-q) fraction of gradient magnitudes.
   This is a proxy for strong edges.
3) Flagged fraction INSIDE the edge band vs OUTSIDE the band (background).
4) An edge-bias ratio = (edge-band flagged %) / (background flagged %).

Interpretation (key)
--------------------
- If edge-band% >> background%  (ratio >> 1):
    The mask is edge-biased. For gamma detection this is BAD:
    you are likely touching real object boundaries.

- If edge-band% ~ background% (ratio ~ 1):
    The mask is not strongly edge-biased. This is more consistent with impulse noise.

Practical recommendations
-------------------------
- For a "gamma mask":
    ratio ~ 1–3x : generally acceptable (depends on sample + motion).
    ratio > ~5x  : strong evidence of edge-triggering; improve spatial confirmation.

- For a "spatial cleanup mask":
    ratio ~ 1–2x : good (cleanup is not edge-destructive).
    ratio > ~3x  : cleanup may be eating edges; strengthen edge protection (increase q),
                  reduce max_area, or raise tau_s.

Notes / limitations (be honest)
-------------------------------
- The "gradient band" is a *proxy* for edges. It can include texture inside the sample.
  Still, a huge ratio is a red flag.
- This test is single-image. Run it on multiple indices (e.g., 1, 200, 600, 1000)
  to avoid cherry-picking.
"""

from pathlib import Path
import numpy as np
from astropy.io import fits
import argparse


def read_fits(path: Path) -> np.ndarray:
    """Read FITS image as float32."""
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)


def band_mask_from_gradient(T: np.ndarray, q: float = 0.995) -> np.ndarray:
    """
    Make an "edge band" mask from a simple gradient magnitude proxy.

      g = |dT/dx| + |dT/dy|

    Then define the edge band as pixels with g in the top (1-q) fraction.

    Example:
      q=0.995 -> top 0.5% gradients -> strong edges only
      q=0.990 -> top 1.0% gradients -> more permissive
    """
    gx = np.zeros_like(T)
    gy = np.zeros_like(T)
    gx[:, 1:] = np.abs(T[:, 1:] - T[:, :-1])
    gy[1:, :] = np.abs(T[1:, :] - T[:-1, :])
    g = gx + gy

    gf = g[np.isfinite(g)]
    if gf.size == 0:
        return np.zeros_like(T, dtype=bool)

    thr = np.quantile(gf, q)
    return np.isfinite(g) & (g >= thr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Check whether a mask is concentrated on high-gradient (edge) regions."
    )
    ap.add_argument(
        "--T",
        required=True,
        help="Path to transmission-like FITS (T, Tcorr, or Tclean).",
    )
    ap.add_argument(
        "--M",
        required=True,
        help="Path to mask FITS (gamma mask or spatial mask; nonzero = flagged).",
    )
    ap.add_argument(
        "--q",
        type=float,
        default=0.995,
        help="Gradient quantile defining edge band (0.995 = top 0.5%% gradients).",
    )
    args = ap.parse_args()

    T_path = Path(args.T)
    M_path = Path(args.M)

    T = read_fits(T_path)
    M = read_fits(M_path) > 0.5  # boolean mask

    # Finite pixels in T define the evaluation domain (avoid NaNs/inf)
    finite = np.isfinite(T)
    total_finite = int(finite.sum())
    total_flag = int(np.sum(M & finite))
    overall_pct = 100.0 * total_flag / max(total_finite, 1)

    print("\n=== Gamma/Mask Edge-Bias Check ===")
    print("T file   :", T_path)
    print("Mask file:", M_path)
    print(f"Finite pixels: {total_finite:,}")
    print(f"Flagged pixels (finite only): {total_flag:,}  ({overall_pct:.4f}%)")

    # Edge band from gradient proxy
    band = band_mask_from_gradient(T, q=args.q)
    band &= finite  # only evaluate where T is finite

    n_b = int(np.sum(band & M))
    d_b = int(np.sum(band))
    p_b = 100.0 * n_b / max(d_b, 1)

    # Background: finite pixels not in the edge band
    bg = finite & (~band)
    n_bg = int(np.sum(bg & M))
    d_bg = int(np.sum(bg))
    p_bg = 100.0 * n_bg / max(d_bg, 1)

    # Ratio: how many times more likely to be flagged on edges vs background
    ratio = p_b / max(p_bg, 1e-12)

    print(f"\nEdge band definition: top {(1-args.q)*100:.2f}% gradient pixels (q={args.q})")
    print(f"Edge-band flagged: {n_b:,}/{d_b:,}  = {p_b:.4f}%")
    print(f"Background flagged: {n_bg:,}/{d_bg:,} = {p_bg:.4f}%")
    print(f"Edge-bias ratio (edge% / bg%): {ratio:.2f}x")

    print("\n--- How to interpret (practical) ---")
    print("A) For GAMMA masks (temporal detector):")
    print("   - ratio ~ 1–3x  : usually acceptable")
    print("   - ratio > ~5x   : strong edge-triggering -> improve edge protection / spatial confirmation")
    print("\nB) For SPATIAL-cleanup masks (presentation step):")
    print("   - ratio ~ 1–2x  : good (not eating edges)")
    print("   - ratio > ~3x   : cleanup likely eroding real boundaries -> tighten parameters")
    print("\nGeneral:")
    print(" - If edge-band% is huge (e.g., tens of %) while bg% is small, the mask is biased.")
    print(" - Run this on multiple indices (e.g., 00001, 00200, 00600, 01000) to avoid cherry-picking.")

    print("\n--- Example commands (PowerShell) ---")
    print(r'pixi run python sanity_check/check_gamma_mask.py --T "...\transmission\T_00001.fits" --M "...\masks\gamma\gamma_00001.fits"')
    print(r'pixi run python sanity_check/check_gamma_mask.py --T "...\transmission_corr_clean\Tclean_00001.fits" --M "...\masks\spatial\spatial_00001.fits"')

    print("\nDone.")
