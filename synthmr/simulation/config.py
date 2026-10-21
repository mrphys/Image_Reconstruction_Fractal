"""Configuration of the image -> multicoil k-space simulation stage."""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SimulationConfig:
    """Mirrors the legacy ``preprocessing_natural_videos.config_default`` keys.

    ``num_coils``, ``regsnr`` and ``sigma_coil`` accept one value (fixed) or two
    (uniform random range), as in the legacy code.
    """

    base_resolution: int = 256
    phases: int = 32
    masking: bool = True
    regsnr: Sequence[float] = (12, 22)
    sigma_coil: Sequence[float] = (2, 6)
    complex_transform: float = 1
    num_coils: Sequence[int] = (15,)
    add_phase: float = 2
    image_adjustment: Sequence[float] = (0, 1)
    coil_size_factor: int = 2      # coil kernel support = factor * base_resolution
    coil_phase_ngrid: int = 2      # background-phase grid of each coil map
    object_phase_ngrid: int = 6    # background-phase grid of the object
    # "2d": per-frame 2D simulation (legacy). "3d": volume-consistent 3D (Phase 3).
    mode: str = "2d"
    slices: int = 1                # Z (3d mode)
    slice_spacing_px: float = 10.0 # 3d: slice spacing in in-plane pixel units (coil/mask/phase z-extent)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Normalise list/tuple inputs (legacy dicts, YAML) so configs compare and serialise consistently.
        for name in ("regsnr", "sigma_coil", "num_coils", "image_adjustment"):
            setattr(self, name, tuple(getattr(self, name)))

    @classmethod
    def from_legacy(cls, d):
        """Build from a legacy ``config_default`` dict."""
        keys = {"base_resolution", "phases", "masking", "regsnr", "sigma_coil",
                "complex_transform", "num_coils", "add_phase", "image_adjustment"}
        unknown = set(d) - keys - {"return_gt"}
        if unknown:
            raise KeyError(f"unknown legacy simulation keys {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in keys})
