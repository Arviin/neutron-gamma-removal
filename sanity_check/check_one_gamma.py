from pathlib import Path
import sys
import subprocess

# ---- CHANGE ONLY THIS NUMBER ----
IDX = "01010"   # e.g. "00001", "00600", "01000"
# ----------------------------------

run_folder = Path(
    r"D:\Projects\neutron-gamma-removal\data\processed\ct10s\run_20260224_150631"
)

Tfile = run_folder / "transmission_corr" / f"Tcorr_{IDX}.fits"
Mfile = run_folder / "masks" / "gamma" / f"gamma_{IDX}.fits"

cmd = [
    "python",
    "sanity_check/check_gamma_mask.py",
    "--T", str(Tfile),
    "--M", str(Mfile),
]

subprocess.run(cmd)