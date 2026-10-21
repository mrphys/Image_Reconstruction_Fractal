"""Sampling trajectories and operators.

A trajectory is an in-plane encoder (``radial`` | ``spiral`` | ``file``) combined with a
through-plane encoder (``none`` for 2D, ``cartesian_kz`` for stack-of-stars /
stack-of-spirals). Each in-plane encoder has its own config class (``TRAJECTORY_KINDS``).
New encoders: see .claude/skills/add-trajectory.
"""

from dataclasses import dataclass, field, fields

from synthmr.backend import get_backend


@dataclass
class TrajectoryConfig:
    """Fields shared by every in-plane encoder. Build a subclass (see ``TRAJECTORY_KINDS``)."""

    kind: str = field(init=False, default="")   # fixed by the subclass
    base_resolution: int = 256
    phases: int = 32
    ordering: str = "tiny_half"
    reverse: bool = True
    through_plane: str = "none"          # none | cartesian_kz
    kz_rotation: str = "aligned"         # aligned | golden (per-partition angular offset)

    @staticmethod
    def from_dict(d):
        """Build the subclass named by ``d["kind"]`` (default ``radial``)."""
        d = dict(d)
        kind = d.pop("kind", "radial")
        if kind not in TRAJECTORY_KINDS:
            raise ValueError(f"unknown trajectory kind {kind!r}; available: {sorted(TRAJECTORY_KINDS)}")
        cls = TRAJECTORY_KINDS[kind]
        unknown = set(d) - {f.name for f in fields(cls) if f.init}
        if unknown:
            raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
        return cls(**d)

    @staticmethod
    def common_fields():
        """Names of the fields every kind shares (kept when an override switches ``kind``)."""
        return [f.name for f in fields(TrajectoryConfig)]


@dataclass
class RadialTrajectoryConfig(TrajectoryConfig):
    kind: str = field(init=False, default="radial")
    spokes: int = 13

    @classmethod
    def from_legacy(cls, d):
        """From a legacy ``config_radial_traj()`` dict (radial only)."""
        d = dict(d)
        if d.get("radial_spokes", 0) <= 0:
            raise ValueError("legacy spiral configs: build SpiralTrajectoryConfig(...) directly")
        kwargs = {k: d[k] for k in ("base_resolution", "phases", "ordering", "reverse") if k in d}
        return cls(spokes=d["radial_spokes"], **kwargs)


@dataclass
class SpiralTrajectoryConfig(TrajectoryConfig):
    kind: str = field(init=False, default="spiral")
    field_of_view: float = 400
    tiny_number: int = 7
    max_tempres: int = 50
    min_max_arm_time: list = field(default_factory=lambda: [3, 8])
    vd_spiral_arms: int = 26
    vd_inner_cutoff: float = 0.2
    pre_vd_outer_cutoff: float = 0.5
    vd_outer_density: float = 0.25
    vd_type: str = "linear"
    max_grad_ampl: float = 24.0
    min_rise_time: float = 8.125
    dwell_time: float = 1.4
    gradient_delay: float = 0.56
    readoutOS: int = 2
    deadtime: float = 1.9
    larmor_const: float = 42.577478518
    flow: int = 0

    def waveform_params(self):
        """Spiral design parameters (everything not shared with other kinds, minus ``field_of_view``)."""
        common = set(TrajectoryConfig.common_fields()) | {"field_of_view"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in common}


@dataclass
class FileTrajectoryConfig(TrajectoryConfig):
    """Trajectory + density loaded from ``.npy`` files (see :mod:`synthmr.sampling.files`).

    ``traj_units``: ``radians`` ([-pi, pi], synthmr/tfmri) or ``pixels`` ([-N/2, N/2], sigpy/extreme_mri,
    N = ``base_resolution``). ``density_kind``: ``density`` (dcw = 1/density, as for generated
    trajectories) or ``dcw`` (used as is). ``views_axis``: read a 3-D array as ``(V, S, 2)`` instead
    of ``(T, R, 2)``. Fewer frames than ``phases`` are repeated cyclically.
    """

    kind: str = field(init=False, default="file")
    reverse: bool = False
    traj_path: str = ""
    density_path: str = ""
    traj_units: str = "radians"
    density_kind: str = "density"
    views_axis: bool = False


TRAJECTORY_KINDS = {"radial": RadialTrajectoryConfig, "spiral": SpiralTrajectoryConfig,
                    "file": FileTrajectoryConfig}


def make_trajectory(config, backend=None):
    """Return ``(traj (T, R, 2), dcw (T, R))`` backend tensors for the in-plane encoder."""
    backend = backend or get_backend()
    if config.kind == "file":
        from .files import load_trajectory_files
        traj, dcw = load_trajectory_files(config)
        return backend.from_numpy(traj), backend.from_numpy(dcw)
    return backend.sampling.make_trajectory(config)


__all__ = ["TrajectoryConfig", "RadialTrajectoryConfig", "SpiralTrajectoryConfig", "FileTrajectoryConfig",
           "TRAJECTORY_KINDS", "make_trajectory"]
