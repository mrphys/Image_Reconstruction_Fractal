import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # share the GPU with subprocesses
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

# The env var alone is not honoured once TF-MRI/NUFFT are imported; set growth explicitly
# before any GPU context exists so tests do not monopolise the GPU.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

GOLDEN = ROOT / "tests/artifacts/golden/gpu"
ARTIFACTS = ROOT / "tests/artifacts"


def _has_gpu():
    import tensorflow as tf
    return bool(tf.config.list_physical_devices("GPU"))


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
    if not GOLDEN.exists():
        pytest.skip("golden missing: run tests/regression/capture_golden.py --device gpu")
    return GOLDEN
