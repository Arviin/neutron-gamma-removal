# scripts/run_one_scan.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
# scripts/run_one_scan.py

import sys
from pathlib import Path

# --- Make sure "src/" is on PYTHONPATH so `import ngamma` works ---
ROOT = Path(__file__).resolve().parents[1]   # repo root
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
else:
    raise FileNotFoundError(f"Expected src/ folder at: {SRC}")

from ngamma.config import ScanConfig
from ngamma.pipeline import process_scan


def main():
    # ---- EDIT THESE 3 LINES ONLY ----
    raw_folder = Path(r"D:\mpc3091-Qianru\0_data\01_ballpacking_snr\02_rawdata2\05_ct10s")
    output_folder = Path(r"D:\Projects\neutron-gamma-removal\data\processed\ct10s")
    exposure_prefix = "10s"
    # ---------------------------------

    cfg = ScanConfig(
        raw_folder=raw_folder,
        output_folder=output_folder,
        exposure_prefix=exposure_prefix,
    )

    # Make a run-specific output folder so you never mix runs again
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_folder = cfg.output_folder / f"run_{run_tag}"

    process_scan(cfg)


if __name__ == "__main__":
    main()