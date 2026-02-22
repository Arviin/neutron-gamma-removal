import numpy as np
from scipy.ndimage import distance_transform_edt


def fill_nans_nearest(A: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by nearest-neighbor interpolation using distance transform.

    This is ONLY for:
      - visualization (ImageJ)
      - reconstruction tools that cannot handle NaNs

    It does NOT change the scientific gamma detection (which uses NaNs correctly).
    """
    A = np.asarray(A, dtype=np.float32).copy()
    nan = ~np.isfinite(A)
    if not nan.any():
        return A

    # CRITICAL: return_distances must be False, otherwise SciPy returns (dist, indices)
    indices = distance_transform_edt(nan, return_indices=True, return_distances=False)

    # indices shape is (ndim, H, W) -> (2, H, W)
    iy = indices[0]
    ix = indices[1]

    A[nan] = A[iy[nan], ix[nan]]
    return A
