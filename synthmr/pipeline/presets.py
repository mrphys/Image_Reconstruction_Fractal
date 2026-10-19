"""Named pipeline presets. The 2D presets reproduce the legacy entry points exactly."""

from synthmr.recon import AugmentConfig, ReconConfig
from synthmr.sampling import RadialTrajectoryConfig
from synthmr.simulation import SimulationConfig

from .config import InputConfig, OutputConfig, PipelineConfig


def fractal_2dt_legacy():
    """Base repo ``train_network.ipynb``: escape-time fractal -> sine-wave complexify -> 13 sorted spokes."""
    return PipelineConfig(
        name="fractal_2dt_legacy",
        input=InputConfig(complexify="sine_wave", legacy_global_np=True),
        simulation=SimulationConfig(num_coils=(15,), complex_transform=1),
        trajectory=RadialTrajectoryConfig(spokes=13, base_resolution=256, phases=32, ordering="sorted_half"),
        recon=ReconConfig(input_format="coil_compressed", input_coils=10, output_format="abs"),
        augment=AugmentConfig(phases=24, roll=1, rotation=1),
    )


def fractal_new_2dt():
    """``process_fractals_new_grasp``: new-framework complex volume, 10 coils, 13 golden spokes."""
    return PipelineConfig(
        name="fractal_new_2dt",
        input=InputConfig(complexify="complex_to_video"),
        simulation=SimulationConfig(num_coils=(10,), complex_transform=1),
        trajectory=RadialTrajectoryConfig(spokes=13, base_resolution=256, phases=32, ordering="golden_half"),
        recon=ReconConfig(input_format="coil_compressed", input_coils=10, output_format="abs"),
        augment=AugmentConfig(phases=24, roll=1, rotation=1),
    )


def fractal_new_2dt_nocc():
    """``process_fractals_new_grasp_grouped`` settings (its ``input_format='blah'`` = no coil compression).

    The per-group RNG re-seeding of that script is superseded by the 3D+t preset.
    """
    cfg = fractal_new_2dt()
    cfg.name = "fractal_new_2dt_nocc"
    cfg.recon.input_format = "complex"
    return cfg


def fractal_3dt_sos():
    """3D+t: ``(T, Z, H, W)`` fractal volume, volume-consistent simulation, stack-of-stars
    (13 golden-angle spokes per partition, Cartesian kz), 3D coil maps from the time-averaged
    Tukey-filtered data. Replaces the per-slice ``process_fractals_new_grasp_grouped`` hack."""
    return PipelineConfig(
        name="fractal_3dt_sos",
        input=InputConfig(complexify="complex_to_video"),
        simulation=SimulationConfig(mode="3d", slices=10, slice_spacing_px=10.0, num_coils=(10,),
                                    complex_transform=1),
        trajectory=RadialTrajectoryConfig(spokes=13, base_resolution=256, phases=32, ordering="golden_half",
                                          through_plane="cartesian_kz", kz_rotation="aligned"),
        recon=ReconConfig(input_format="complex", output_format="abs", coil_maps="time_avg_tukey_3d",
                          tukey_width=(1.0, 0.25, 0.25), tukey_alpha=0.5, anchor_phase=True),
        augment=AugmentConfig(phases=24, roll=1, rotation=1, z_flip=0),
        output=OutputConfig(format="h5"),
    )


PRESETS = {f.__name__: f for f in (fractal_2dt_legacy, fractal_new_2dt, fractal_new_2dt_nocc, fractal_3dt_sos)}


def register_preset(fn):
    PRESETS[fn.__name__] = fn
    return fn


def get_preset(name):
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()
