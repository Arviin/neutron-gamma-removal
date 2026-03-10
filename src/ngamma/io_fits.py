from pathlib import Path
import re
import numpy as np
from astropy.io import fits
from typing import Optional, Tuple


INDEX_RE = re.compile(r".*_(\d+)\.fits$")

def extract_index(path: Path) -> int:
    m = INDEX_RE.match(path.name)
    return int(m.group(1)) if m else -1

def list_sorted(folder: Path, prefix: str):
    files = list(folder.glob(f"{prefix}_*.fits"))
    files = sorted(files, key=extract_index)
    return files

# def read_fits(path: Path):
#     with fits.open(path, memmap=False) as hdul:
#         return hdul[0].data.astype(np.float32)

# def write_fits(path: Path, data: np.ndarray):
#     hdu = fits.PrimaryHDU(data.astype(np.float32))
#     path.parent.mkdir(parents=True, exist_ok=True)
#     hdu.writeto(path, overwrite=True)




def read_fits(path: Path, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    if roi is not None:
        y0, y1, x0, x1 = roi

        if not (0 <= y0 < y1 <= data.shape[0] and 0 <= x0 < x1 <= data.shape[1]):
            raise ValueError(f"Invalid ROI {roi} for image shape {data.shape}")

        data = data[y0:y1, x0:x1]

    return data


def write_fits(path: Path, data: np.ndarray):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)