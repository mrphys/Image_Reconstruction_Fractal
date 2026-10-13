"""2e: presets equal the legacy configs; Pipeline + make_dataset CLI reproduce golden files."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from synthmr.pipeline import PipelineConfig, get_preset, load_config
from tests.regression.compare import check_stage
from tests.regression.new_runner import PRESET_FOR, configs

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fixture", list(PRESET_FOR))
def test_preset_matches_legacy_config(fixture):
    sim, traj, rec, aug = configs(fixture)
    p = get_preset(PRESET_FOR[fixture])
    assert p.simulation == sim and p.trajectory == traj and p.augment == aug
    legacy_cc = rec.input_format == "coil_compressed"
    assert (p.recon.input_format == "coil_compressed") == legacy_cc
    assert (p.recon.base_resolution, p.recon.phases, p.recon.input_coils, p.recon.output_format,
            p.recon.normalize_input) == (rec.base_resolution, rec.phases, rec.input_coils, rec.output_format,
                                         rec.normalize_input)


def test_config_roundtrip_and_overrides(tmp_path):
    cfg = get_preset("fractal_new_2dt")
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg
    f = tmp_path / "o.yaml"
    f.write_text("simulation:\n  num_coils: [12]\ntrajectory:\n  radial_spokes: 21\n")
    c2 = load_config("fractal_new_2dt", path=f)
    assert c2.simulation.num_coils == (12,) and c2.trajectory.radial_spokes == 21
    assert c2.recon == cfg.recon
    with pytest.raises(KeyError):
        cfg.updated({"simulation": {"bogus": 1}})


def test_make_dataset_cli_reproduces_golden(golden_dir, tmp_path):
    """CLI with preset fractal_new_2dt on one input file == golden B_cc final pair.

    Run in-process (the Pipeline owns fresh RNG streams, so state equals a fresh process);
    a subprocess would compete with this process's TF GPU memory pool.
    """
    src = tmp_path / "in"
    src.mkdir()
    np.save(src / "julia_00_slice_1.npy", np.load(golden_dir / "B_cc/fractal_volume.npy"))
    out = tmp_path / "out"
    cli = runpy.run_path(str(ROOT / "scripts/make_dataset.py"))
    cli["main"](["--preset", "fractal_new_2dt", "--input", str(src), "--output", str(out), "--gifs", "1"])
    assert (out / "gifs/julia_00_slice_1/pair.gif").exists()
    for stage, suffix in (("zfill_rolled", "input"), ("image_rolled", "output")):
        ok, msg = check_stage(stage, np.load(golden_dir / f"B_cc/{stage}.npy"),
                              np.load(out / f"julia_00_slice_1_{suffix}.npy"))
        assert ok, f"{stage}: {msg}"
