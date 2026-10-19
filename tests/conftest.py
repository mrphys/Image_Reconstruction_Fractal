import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # share the GPU with subprocesses
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

try:  # TF-free environments (sigpy backend) run the backend-agnostic tests only
    import tensorflow as tf  # noqa: E402
except ImportError:
    tf = None

# The env var alone is not honoured once TF-MRI/NUFFT are imported; set growth explicitly
# before any GPU context exists so tests do not monopolise the GPU.
for _gpu in (tf.config.list_physical_devices("GPU") if tf is not None else []):
    tf.config.experimental.set_memory_growth(_gpu, True)

GOLDEN = ROOT / "tests/artifacts/golden/gpu"
ARTIFACTS = ROOT / "tests/artifacts"


def _has_gpu():
    return tf is not None and bool(tf.config.list_physical_devices("GPU"))


def pytest_ignore_collect(collection_path, config):
    """Without TensorFlow, skip the modules that drive the legacy TF code or the TF backend directly."""
    if tf is None and collection_path.name in TF_ONLY_MODULES:
        return True
    return None


TF_ONLY_MODULES = {
    "capture_golden.py", "legacy_runner.py", "new_runner.py", "test_phase2a_rng.py", "test_phase2b_fractal.py",
    "test_phase2c_simulation.py", "test_phase2d_end_to_end.py", "test_phase2e_pipeline.py", "test_phase3_smoke.py",
    "test_3d.py",
}


@pytest.fixture(autouse=True)
def _restore_active_backend():
    """Tests that call ``set_backend`` must not change the default backend seen by later tests."""
    import synthmr.backend as backend
    previous = backend._active
    yield
    backend._active = previous


@pytest.fixture(scope="session")
def golden_dir():
    if not GOLDEN.exists():
        pytest.skip("golden missing: run tests/regression/capture_golden.py --device gpu")
    if not _has_gpu():
        pytest.skip("golden regression requires a GPU (TF CPU conv3d segfaults in tfmri Walsh)")
    return GOLDEN


@pytest.fixture(scope="session")
def artifacts_dir():
    return ARTIFACTS


@pytest.fixture(scope="session")
def golden_np_dir():
    """Golden directory for numpy-only stages (no GPU needed to compare)."""
    if tf is None:
        pytest.skip("golden comparisons need TensorFlow")
    if not GOLDEN.exists():
        pytest.skip("golden missing: run tests/regression/capture_golden.py --device gpu")
    return GOLDEN
