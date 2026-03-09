from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

@dataclass
class ScanConfig:
    raw_folder: Path
    output_folder: Path
    exposure_prefix: str   # e.g. "10s"
    roi: Optional[Tuple[int, int, int, int]] = None