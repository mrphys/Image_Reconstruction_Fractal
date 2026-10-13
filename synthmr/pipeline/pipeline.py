"""Composable pipeline: phantom volume -> (network input, target) pair."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from synthmr.backend import get_backend
from synthmr.phantoms import complexify
from synthmr.recon import augment, reconstruct
from synthmr.rng import RNG
from synthmr.sampling import make_trajectory
from synthmr.simulation import simulate


@dataclass
class Sample:
    input: np.ndarray                 # 2D+t: (T, H, W, ch); 3D+t: (T, Z, H, W, ch)
    target: np.ndarray                # same leading dims, ch = 1 (abs) or 2 (complex)
    meta: dict = field(default_factory=dict)
    stages: Optional[dict] = None     # intermediate arrays when run(..., return_stages=True)


class Pipeline:
    """Runs the stages configured by a :class:`PipelineConfig`.

    Random state lives in the pipeline (``self.rng`` backend streams and
    ``self.np_rng`` for complexify) and continues across ``run`` calls, like the
    legacy scripts processing a directory in one process.
    """

    def __init__(self, config, rng=None, backend=None):
        self.config = config
        self.backend = backend or get_backend()
        seeds = {k: v for k, v in config.seeds.items() if k != "numpy"}
        self.rng = rng or RNG(seeds, backend=self.backend)
        self.np_rng = np.random if config.input.legacy_global_np else np.random.default_rng(
            config.seeds.get("numpy", 0))
        self._traj = None

    def to_video(self, volume):
        method = self.config.input.complexify
        if method == "video":
            return volume
        if method == "complex_to_video":
            return complexify.complex_to_video(volume)
        return complexify.METHODS[method](volume, rng=self.np_rng)

    def trajectory(self):
        if self._traj is None:      # deterministic: build once
            self._traj = make_trajectory(self.config.trajectory, backend=self.backend)
        return self._traj

    def run(self, volume, return_stages=False):
        """Process one phantom volume (``(T, H, W)`` for 2D+t, ``(T, Z, H, W)`` for 3D+t)."""
        cfg, B = self.config, self.backend
        video = self.to_video(np.asarray(volume))
        video_u8 = np.clip(video, 0, 255).astype(np.uint8)
        with B.device(cfg.device):
            sim = simulate(video_u8, cfg.simulation, self.rng, backend=B)
            traj, dcw = self.trajectory()
            rec = reconstruct(sim["kspace"], traj, dcw, cfg.recon, through_plane=cfg.trajectory.through_plane,
                              kz_rotation=cfg.trajectory.kz_rotation, backend=B)
            zfill, image = augment(rec["zfill"], rec["image"], cfg.augment, self.rng, backend=B)
        stages = None
        if return_stages:
            stages = {"video": video, "video_uint8": video_u8,
                      **{f"sim_{k}": B.to_numpy(v) for k, v in sim.items() if v is not None},
                      **{k: B.to_numpy(v) for k, v in rec.items() if v is not None}}
        return Sample(input=B.to_numpy(zfill), target=B.to_numpy(image),
                      meta={"preset": cfg.name, "seeds": self.rng.seeds()}, stages=stages)
