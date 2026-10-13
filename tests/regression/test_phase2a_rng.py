"""2a: legacy code driven by synthmr.rng streams reproduces golden exactly.

Proves the RNG wrapper (backend Generator + named streams) is a drop-in for the
legacy module-level ``rg`` / ``noise_rg`` generators.
"""

import pytest

from synthmr.rng import RNG
from tests.regression.legacy_runner import FIXTURES, run_fixture
from tests.regression.harness import assert_matches_golden


@pytest.mark.parametrize("fixture", FIXTURES)
def test_legacy_with_injected_rng_matches_golden(fixture, golden_dir, artifacts_dir):
    cases, _ = run_fixture(fixture, rng=RNG())
    for case, stages in cases.items():
        assert_matches_golden(golden_dir, case, stages, gif_dir=artifacts_dir / "phase2/2a")
