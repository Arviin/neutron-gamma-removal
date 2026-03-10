from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

@dataclass
class ScanConfig:
    raw_folder: Path
    output_folder: Path
    exposure_prefix: str   # e.g. "10s"
    # ROI = (y0, y1, x0, x1)
    roi: Optional[Tuple[int, int, int, int]] = None