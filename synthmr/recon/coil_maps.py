"""Backend-free helpers for coil-map estimation."""

import numpy as np
from scipy.signal.windows import tukey


def tukey_window_nd(shape, widths, alpha=0.5):
    """Separable centred Tukey window ``shape`` whose support along axis ``i`` is
    ``round(widths[i] * shape[i])`` samples (zero outside) with taper ``alpha``.

    The taper is sampled without its zero end points, so a full-width window never
    zeroes the outermost samples (important for the few kz partitions of a stack).

    Used as a k-space low-pass: ``widths=(1.0, 0.25, 0.25)`` keeps all kz partitions and the
    central quarter in-plane.
    """
    win = np.ones(shape, dtype=np.float32)
    for ax, (n, wfrac) in enumerate(zip(shape, widths)):
        m = max(1, int(round(wfrac * n)))
        w1 = np.zeros(n, dtype=np.float32)
        start = (n - m) // 2
        w1[start:start + m] = tukey(m + 2, alpha)[1:-1]
        shp = [1] * len(shape)
        shp[ax] = n
        win = win * w1.reshape(shp)
    return win
