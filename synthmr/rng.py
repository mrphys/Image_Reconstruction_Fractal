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
"""

from synthmr.backend import get_backend

LEGACY_SEEDS = {"transforms": 1, "noise": 2, "augment": 1}


class RNG:
    """A set of named, independently seeded random streams."""

    def __init__(self, seeds=None, backend=None):
        backend = backend or get_backend()
        self._streams = {name: backend.Generator(seed)
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
