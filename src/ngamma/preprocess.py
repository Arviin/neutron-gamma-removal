import numpy as np

def compute_transmission(I, OBm, DCm, good_mask,
                         t_min=1e-6, t_max=2.0):
    """
    Physically constrained transmission.
    Pixels outside good_mask -> NaN (do not pretend they are valid).
    """
    I = I.astype(np.float32)
    num = I - DCm
    den = OBm - DCm

    # physical: counts can't be negative
    num = np.clip(num, 0.0, None)

    T = np.empty_like(num, dtype=np.float32)
    T[:] = np.nan

    # only compute where detector is stable
    m = good_mask
    T[m] = num[m] / den[m]

    # allow slight >1 but prevent absurd values
    T[m] = np.clip(T[m], t_min, t_max)

    return T

def log_transform(T):
    out = np.full_like(T, np.nan, dtype=np.float32)
    m = np.isfinite(T)
    out[m] = -np.log(T[m])
    return out
