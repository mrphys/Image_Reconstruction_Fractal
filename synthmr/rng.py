"""Injected random-number streams.

Replaces the legacy module-level generators. Each named stream is an independent
backend ``Generator``; the legacy code had three, which map to:

=============  ============================================  ===========
stream         legacy generator                              legacy seed
=============  ============================================  ===========
``transforms``  ``preprocessing_natural_videos.rg``           1
``noise``       ``preprocessing_natural_videos.noise_rg``     2
``augment``     ``preprocessing_multicoil_noselect.rg``       1
=============  ============================================  ===========

Keeping noise separate means re-seeding ``transforms``/``augment`` (to share
augmentations) leaves noise realisations independent.

``RNG(kind="numpy")`` uses :class:`NumpyGenerator` streams instead of the backend's own
generator. Every backend accepts them, so the same seeds give the same draws on TF and sigpy
(used for cross-backend comparison; the TF default stays philox for the legacy golden).
"""

import numpy as np

from synthmr.backend import get_backend

LEGACY_SEEDS = {"transforms": 1, "noise": 2, "augment": 1}


def _np_dtype(dtype):
    return np.dtype(getattr(dtype, "as_numpy_dtype", dtype))


def _shape(shape):
    return tuple(int(s) for s in np.asarray(shape, dtype=np.int64).reshape(-1))


class NumpyGenerator:
    """Seeded numpy (philox) generator with the ``backend.Generator`` interface.

    Accepts TF-style arguments (``tf.shape(...)`` shapes, tensor bounds, ``tf.int32``) so TF stage
    code can draw from it unchanged. Returns host numpy arrays.
    """

    def __init__(self, seed):
        self.seed = seed
        self._gen = np.random.Generator(np.random.Philox(seed))

    def reset(self, seed=None):
        self.seed = self.seed if seed is None else seed
        self._gen = np.random.Generator(np.random.Philox(self.seed))

    def uniform(self, shape=(), minval=0, maxval=None, dtype="float32"):
        dtype, shape = _np_dtype(dtype), _shape(shape)
        if np.issubdtype(dtype, np.integer):
            return self._gen.integers(int(np.asarray(minval)), int(np.asarray(maxval)), size=shape).astype(dtype)
        hi = 1.0 if maxval is None else np.asarray(maxval, dtype=np.float64)
        return self._gen.uniform(np.asarray(minval, dtype=np.float64), hi, size=shape).astype(dtype)

    def normal(self, shape=(), mean=0.0, stddev=1.0, dtype="float32"):
        z = self._gen.standard_normal(size=_shape(shape))
        return (z * float(np.asarray(stddev)) + float(np.asarray(mean))).astype(_np_dtype(dtype))


class RNG:
    """A set of named, independently seeded random streams.

    ``kind="backend"`` (default) uses ``backend.Generator``; ``kind="numpy"`` uses :class:`NumpyGenerator`.
    """

    def __init__(self, seeds=None, backend=None, kind="backend"):
        if kind == "numpy":
            make = NumpyGenerator
        elif kind == "backend":
            make = (backend or get_backend()).Generator
        else:
            raise ValueError(f"unknown rng kind {kind!r}; use 'backend' or 'numpy'")
        self.kind = kind
        self._streams = {name: make(seed)
                         for name, seed in {**LEGACY_SEEDS, **(seeds or {})}.items()}

    def __getitem__(self, name):
        return self._streams[name]

    @property
    def transforms(self):
        return self._streams["transforms"]

    @property
    def noise(self):
        return self._streams["noise"]

    @property
    def augment(self):
        return self._streams["augment"]

    def reset(self, **seeds):
        """Re-seed the given streams (``rng.reset(transforms=5, augment=5)``)."""
        for name, seed in seeds.items():
            self._streams[name].reset(seed)

    def seeds(self):
        return {name: g.seed for name, g in self._streams.items()}
