"""Sampling trajectories and operators.

A trajectory is an in-plane encoder (``radial`` | ``spiral``) combined with a
through-plane encoder (``none`` for 2D, ``cartesian_kz`` for stack-of-stars /
stack-of-spirals). New encoders: see .claude/skills/add-trajectory.
"""

from dataclasses import dataclass, field

from synthmr.backend import get_backend


def _spiral_defaults():
    return dict(tiny_number=7, max_tempres=50, min_max_arm_time=[3, 8], vd_spiral_arms=26,
                vd_inner_cutoff=0.2, pre_vd_outer_cutoff=0.5, vd_outer_density=0.25, vd_type='linear',
                max_grad_ampl=24.0, min_rise_time=8.125, dwell_time=1.4, gradient_delay=0.56,
                readoutOS=2, deadtime=1.9, larmor_const=42.577478518, flow=0)


@dataclass
class TrajectoryConfig:
    kind: str = "radial"                 # in-plane encoder: radial | spiral
    radial_spokes: int = 13
    base_resolution: int = 256
    field_of_view: float = 400
    phases: int = 32
    ordering: str = "tiny_half"
    reverse: bool = True
    spiral: dict = field(default_factory=_spiral_defaults)
    through_plane: str = "none"          # none | cartesian_kz
    kz_rotation: str = "aligned"         # aligned | golden (per-partition angular offset)

    @classmethod
    def from_legacy(cls, d):
        """From a legacy ``config_radial_traj()`` dict (radial only)."""
        d = dict(d)
        if d.get("radial_spokes", 0) <= 0:
            raise ValueError("legacy spiral configs: build TrajectoryConfig(kind='spiral', spiral=...) directly")
        return cls(kind="radial", **{k: d[k] for k in
                                     ("radial_spokes", "base_resolution", "field_of_view", "phases",
                                      "ordering", "reverse") if k in d})


def make_trajectory(config, backend=None):
    """Return ``(traj (T, R, 2), dcw (T, R))`` backend tensors for the in-plane encoder."""
    return (backend or get_backend()).sampling.make_trajectory(config)


__all__ = ["TrajectoryConfig", "make_trajectory"]
