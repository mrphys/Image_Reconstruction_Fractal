"""2d: the refactored chain (phantom -> simulate -> trajectory -> recon -> augment) matches golden
at every stage (R5 traj/dcw, R6 zfill/image, R7 rolled pair)."""

import pytest

from tests.regression.harness import assert_matches_golden
from tests.regression.legacy_runner import FIXTURES
from tests.regression.new_runner import run_fixture


@pytest.mark.parametrize("fixture", FIXTURES)
def test_new_chain_matches_golden(fixture, golden_dir, artifacts_dir):
    for case, stages in run_fixture(fixture).items():
        assert_matches_golden(golden_dir, case, stages, gif_dir=artifacts_dir / "phase2/2d")
