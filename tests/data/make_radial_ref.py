"""Regenerate ``radial_ref_tfmri.npz``: tfmri radial trajectories + radial density for every ordering.

Run in the TF container: ``python tests/data/make_radial_ref.py``. Used by ``tests/unit/test_backend.py`` so
backends without tfmri (sigpy) are checked against the reference implementation.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from synthmr.backend import set_backend  # noqa: E402

ORDERINGS = ("linear", "golden", "golden_half", "sorted", "sorted_half", "tiny", "tiny_half")
BASE_RESOLUTION, VIEWS, PHASES = 32, 13, 3


def main():
    B = set_backend("tf")
    out = {}
    for o in ORDERINGS:
        traj = B.radial_trajectory(BASE_RESOLUTION, VIEWS, PHASES, o)
        out[f"traj_{o}"] = B.to_numpy(traj)
        out[f"dens_{o}"] = B.to_numpy(B.estimate_radial_density(B.flatten_trajectory(traj)))
    np.savez_compressed(Path(__file__).with_name("radial_ref_tfmri.npz"), **out)


if __name__ == "__main__":
    main()
