from dataclasses import dataclass
from pathlib import Path

@dataclass
class ScanConfig:
    raw_folder: Path
    output_folder: Path
    exposure_prefix: str   # e.g. "10s"
