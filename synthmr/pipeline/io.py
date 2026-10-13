"""Saving/loading samples."""

import json
from pathlib import Path

import numpy as np


def save_sample(sample, out_dir, name, fmt="npy", save_meta=True):
    """Write ``sample`` as ``{name}_input.npy``/``{name}_output.npy`` (legacy) or ``{name}.h5``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(out_dir / f"{name}_input.npy", sample.input)
        np.save(out_dir / f"{name}_output.npy", sample.target)
    elif fmt == "h5":
        import h5py
        with h5py.File(out_dir / f"{name}.h5", "w") as f:
            f.create_dataset("input", data=sample.input)
            f.create_dataset("output", data=sample.target)
    else:
        raise ValueError(f"unknown output format {fmt!r}")
    if save_meta:
        (out_dir / f"{name}.json").write_text(json.dumps(sample.meta, indent=2, default=_jsonable))


def load_sample(out_dir, name, fmt="npy"):
    """Return ``(input, target)``."""
    out_dir = Path(out_dir)
    if fmt == "npy":
        return np.load(out_dir / f"{name}_input.npy"), np.load(out_dir / f"{name}_output.npy")
    import h5py
    with h5py.File(out_dir / f"{name}.h5", "r") as f:
        return f["input"][()], f["output"][()]


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
