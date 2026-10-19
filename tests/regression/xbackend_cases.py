"""Cross-backend (TF vs sigpy) regression cases, shared by ``capture_xbackend.py`` and ``test_xbackend.py``.

Every case runs a preset with ``rng="numpy"`` so both backends draw the same random numbers.
Small sizes keep the CPU-only sigpy run fast; ``new_2dt_full`` exercises the full-size 2D preset.
"""

import numpy as np

XBACKEND_DIR = "tests/artifacts/xbackend"
STAGES = ("sim_kspace", "sim_smaps", "sim_image", "traj", "dcw", "uskspace", "zfill", "image")


def _small(n, phases=8, slices=5):
    return {"rng": "numpy", "simulation": {"base_resolution": n, "phases": phases, "slices": slices},
            "trajectory": {"base_resolution": n, "phases": phases},
            "recon": {"base_resolution": n, "phases": phases}, "augment": {"phases": phases - 2},
            "input": {"legacy_global_np": False}}


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


CASES = {
    "legacy_2dt": ("fractal_2dt_legacy", (8, 64, 64), _small(64)),
    "new_2dt": ("fractal_new_2dt", (8, 64, 64), _small(64)),
    "new_2dt_nocc_uncombined": ("fractal_new_2dt_nocc", (8, 64, 64),
                                _merge(_small(64), {"recon": {"input_combine": "none"}})),
    "sos_3dt": ("fractal_3dt_sos", (8, 5, 64, 64), _small(64)),
    "sos_3dt_golden_mapstarget": ("fractal_3dt_sos", (8, 5, 64, 64),
                                  _merge(_small(64), {"trajectory": {"kz_rotation": "golden"},
                                                      "recon": {"target_combine": "maps"}})),
    "new_2dt_full": ("fractal_new_2dt", (32, 256, 256), {"rng": "numpy"}),
}


def case_volume(name):
    """Deterministic complex test volume in [-1, 1] (same for every backend)."""
    shape = CASES[name][1]
    rng = np.random.default_rng(sum(map(ord, name)))
    v = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return (v / np.abs(v).max()).astype(np.complex64)


def case_config(name, backend):
    from synthmr.pipeline import load_config
    preset, _, overrides = CASES[name]
    return load_config(preset, overrides={**overrides, "backend": backend})
