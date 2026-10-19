"""Pipeline configuration: nested dataclasses, serialisable to/from dicts, JSON and YAML."""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from synthmr.recon import AugmentConfig, ReconConfig
from synthmr.rng import LEGACY_SEEDS
from synthmr.sampling import RadialTrajectoryConfig, TrajectoryConfig
from synthmr.simulation import SimulationConfig


@dataclass
class InputConfig:
    """How a stored phantom volume becomes the 2-channel [0, 255] video fed to simulation.

    ``complexify``: ``complex_to_video`` (complex volume in [-1, 1]), ``sine_wave`` /
    ``sine_wave_smooth`` / ``sine_wave_spiral`` (escape-time volume), or ``video`` (already a video).
    """

    complexify: str = "complex_to_video"
    legacy_global_np: bool = False   # draw complexify randomness from the global np.random (legacy)


@dataclass
class OutputConfig:
    format: str = "npy"              # npy (legacy *_input/_output.npy) | h5 (input/output datasets)
    gifs: int = 0                    # write GIFs for the first N samples
    save_meta: bool = True


@dataclass
class PipelineConfig:
    name: str = "custom"
    input: InputConfig = field(default_factory=InputConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    trajectory: TrajectoryConfig = field(default_factory=RadialTrajectoryConfig)
    recon: ReconConfig = field(default_factory=ReconConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seeds: dict = field(default_factory=lambda: {**LEGACY_SEEDS, "numpy": 0})
    device: Optional[str] = None     # e.g. "gpu:0"; None = backend default placement

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)

    def updated(self, overrides):
        """New config with a (possibly nested, partial) dict of overrides applied.

        Switching ``trajectory.kind`` drops the old kind's params, so the new kind starts from its defaults.
        """
        overrides = overrides or {}
        base = self.to_dict()
        kind = overrides.get("trajectory", {}).get("kind")
        if kind is not None and kind != base["trajectory"]["kind"]:
            base["trajectory"] = {k: base["trajectory"][k] for k in TrajectoryConfig.common_fields()}
        return self.from_dict(_deep_merge(base, overrides))


def _from_dict(cls, d):
    kwargs = {}
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    for k, v in d.items():
        t = names[k].type
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            v = t.from_dict(v) if hasattr(t, "from_dict") else _from_dict(t, v)
        kwargs[k] = v
    return cls(**kwargs)


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_overrides(path):
    """Read a JSON or YAML override file."""
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(preset=None, path=None, overrides=None):
    """Preset (by name) <- file overrides <- dict overrides."""
    from .presets import get_preset
    cfg = get_preset(preset) if preset else PipelineConfig()
    if path:
        cfg = cfg.updated(read_overrides(path))
    return cfg.updated(overrides) if overrides else cfg
