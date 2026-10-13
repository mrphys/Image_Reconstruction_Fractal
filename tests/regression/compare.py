"""Compare two directories of captured stages (golden vs rerun, or golden vs new code).

Usage::

    python tests/regression/compare.py tests/artifacts/golden/gpu tests/artifacts/golden_rerun/gpu

Comparison rules per stage live in ``tolerances.json`` next to this file:

* ``exact``: bit-identical (dtype and values).
* ``coilmap_phase``: for stages combined with tfmri Walsh maps. The eigenvector
  phase is not anchored and varies run to run on GPU at isolated pixels, so we
  compare magnitudes tightly (``mag_atol``) and complex values modulo sign
  (``min(|a-b|, |a+b|) <= atol``).
"""

import argparse
import json
from pathlib import Path

import numpy as np

TOLERANCES = json.loads((Path(__file__).with_name("tolerances.json")).read_text())


def as_complex(x):
    """Trailing real/imag channel pair -> complex; anything else unchanged."""
    if not np.iscomplexobj(x) and x.ndim and x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]
    return x


def check_stage(stage, ref, new):
    """Return ``(ok, message)`` comparing ``new`` against ``ref`` under the stage's rule."""
    if ref.shape != new.shape:
        return False, f"shape {new.shape} != {ref.shape}"
    rule = TOLERANCES["stages"].get(stage, {"mode": "exact"})
    if rule["mode"] == "exact":
        ok = ref.dtype == new.dtype and np.array_equal(ref, new)
        diff = np.abs(ref.astype(np.complex128) - new.astype(np.complex128)).max(initial=0.0)
        return ok, f"exact={ok} max|diff|={diff:.3e}"
    if rule["mode"] == "coilmap_phase":
        a, b = as_complex(ref), as_complex(new)
        mag = np.abs(np.abs(a) - np.abs(b)).max(initial=0.0)
        signed = np.minimum(np.abs(a - b), np.abs(a + b)).max(initial=0.0)
        ok = mag <= rule["mag_atol"] and signed <= rule["atol"]
        return ok, f"max|mag diff|={mag:.3e} (<= {rule['mag_atol']}) max sign-inv diff={signed:.3e} (<= {rule['atol']})"
    raise ValueError(f"unknown mode {rule['mode']!r}")


def compare_dirs(ref_root, other_root):
    """``{case: {stage: (ok, message)}}`` for every case/stage in ``ref_root``."""
    out = {}
    for case_dir in sorted(p for p in Path(ref_root).iterdir() if p.is_dir()):
        other = Path(other_root) / case_dir.name
        manifest = json.loads((case_dir / "manifest.json").read_text())
        out[case_dir.name] = {}
        for stage in manifest["stages"]:
            path = other / f"{stage}.npy"
            out[case_dir.name][stage] = (
                check_stage(stage, np.load(case_dir / f"{stage}.npy"), np.load(path))
                if path.exists() else (False, "missing"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("other")
    a = p.parse_args()
    all_ok = True
    for case, stages in compare_dirs(a.ref, a.other).items():
        print(case)
        for stage, (ok, msg) in stages.items():
            all_ok &= ok
            print(f"  {'PASS' if ok else 'FAIL'} {stage:15s} {msg}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
