"""Cross-backend regression: the sigpy backend reproduces captured TF outputs (``capture_xbackend.py``).

Both backends use ``RNG(kind="numpy")``, so every random draw is identical; remaining differences are
float rounding, NUFFT accuracy and the arbitrary SVD / Walsh eigenvector phases (rules in
``tolerances.json["xbackend"]``). Runs wherever sigpy is installed (TF not needed).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sigpy")

from synthmr.backend import set_backend  # noqa: E402
from synthmr.pipeline import Pipeline  # noqa: E402
from synthmr.recon import reconstruct  # noqa: E402
from tests.regression.compare import TOLERANCES, check_stage  # noqa: E402
from tests.regression.xbackend_cases import CASES, STAGES, XBACKEND_DIR, case_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RULES = {k: v for k, v in TOLERANCES["xbackend"].items() if not k.startswith("_")}


def _reference(name):
    path = ROOT / XBACKEND_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"missing {path}: run tests/regression/capture_xbackend.py in the TF container")
    return np.load(path)


def _assert_all(pairs):
    results = {stage: check_stage(stage, ref, new, RULES) for stage, (ref, new) in pairs.items()}
    failures = [f"{s}: {msg}" for s, (ok, msg) in results.items() if not ok]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("name", list(CASES))
def test_pipeline_matches_tf(name):
    ref = _reference(name)
    sample = Pipeline(case_config(name, "sigpy")).run(ref["volume"], return_stages=True)
    pairs = {k: (ref[k], sample.stages[k]) for k in STAGES if k in ref.files and k in sample.stages}
    pairs.update(input=(ref["input"], sample.input), target=(ref["target"], sample.target))
    _assert_all(pairs)


@pytest.mark.parametrize("name", list(CASES))
def test_recon_from_tf_kspace_matches_tf(name):
    """Recon alone, fed TF's simulated k-space and trajectory: isolates recon errors from simulation ones."""
    ref = _reference(name)
    B = set_backend("sigpy")
    cfg = case_config(name, "sigpy")
    rec = reconstruct(B.from_numpy(ref["sim_kspace"]), B.from_numpy(ref["traj"]), B.from_numpy(ref["dcw"]),
                      cfg.recon, through_plane=cfg.trajectory.through_plane,
                      kz_rotation=cfg.trajectory.kz_rotation, backend=B)
    _assert_all({k: (ref[k], B.to_numpy(rec[k])) for k in ("uskspace", "zfill", "image")})
