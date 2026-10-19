"""Capture TF-backend reference outputs for the cross-backend (TF vs sigpy) regression.

Run in the TF container (GPU, like the golden), from the repo root::

    python tests/regression/capture_xbackend.py            # -> tests/artifacts/xbackend/<case>.npz
    python tests/regression/capture_xbackend.py --cases new_2dt sos_3dt

Each ``<case>.npz`` holds the input volume, the pipeline stages listed in ``xbackend_cases.STAGES``,
the augmented ``input``/``target`` and the config (JSON). ``tests/regression/test_xbackend.py`` replays
them on the sigpy backend (TF-free container) with the same numpy random streams.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from tests.regression.xbackend_cases import CASES, STAGES, XBACKEND_DIR, case_config, case_volume  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / XBACKEND_DIR))
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    return p.parse_args(argv)


def capture(name, out_dir):
    import numpy as np

    from synthmr.pipeline import Pipeline

    cfg = case_config(name, "tf")
    volume = case_volume(name)
    sample = Pipeline(cfg).run(volume, return_stages=True)
    arrays = {k: sample.stages[k] for k in STAGES if k in sample.stages}
    np.savez_compressed(Path(out_dir) / f"{name}.npz", volume=volume, input=sample.input, target=sample.target,
                        config=np.array(json.dumps(cfg.to_dict(), default=list)), **arrays)
    print(f"{name}: input {sample.input.shape}, target {sample.target.shape}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        capture(name, args.out)


if __name__ == "__main__":
    main()
