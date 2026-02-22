from pathlib import Path
import re
import numpy as np
from astropy.io import fits

INDEX_RE = re.compile(r".*_(\d+)\.fits$")

def extract_index(path: Path) -> int:
    m = INDEX_RE.match(path.name)
    return int(m.group(1)) if m else -1

def list_sorted(folder: Path, prefix: str):
    files = list(folder.glob(f"{prefix}_*.fits"))
    files = sorted(files, key=extract_index)
    return files

def read_fits(path: Path):
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data.astype(np.float32)

def write_fits(path: Path, data: np.ndarray):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
