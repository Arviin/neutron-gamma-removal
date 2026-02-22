import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from ngamma.config import ScanConfig
from ngamma.pipeline import process_scan


if __name__ == "__main__":

    raw_folder = Path(
        r"D:\mpc3091-Qianru\0_data\01_ballpacking_snr\02_rawdata2\05_ct10s"
    )

    output_folder = Path(
        r"D:\Projects\neutron-gamma-removal\data\processed\ct10s"
    )

    cfg = ScanConfig(
        raw_folder=raw_folder,
        output_folder=output_folder,
        exposure_prefix="10s"
    )

    process_scan(cfg)
