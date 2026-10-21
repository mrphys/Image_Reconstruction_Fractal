"""Run the legacy (old) pipelines on the fixtures and return every stage.

Importing this module imports TensorFlow; callers that need CPU-only placement
must set ``CUDA_VISIBLE_DEVICES`` before importing it.
"""

import contextlib

import numpy as np
import tensorflow as tf

from legacy.app import fractal_framework as ff
from legacy.app import julia_generator_new_grasp as gen
from legacy.app.inter4k_utils import preprocessing_multicoil_noselect as pmc
from legacy.app.inter4k_utils import preprocessing_natural_videos as pnv
from legacy.app.inter4k_utils import preprocessing_trajectory_gen as ptg
from legacy.base_repo import fractal_utils as fr
from tests.regression import fixtures as fx

FIXTURES = ("A", "B_cc", "B_grouped")


@contextlib.contextmanager
def injected_rng(rng):
    """Temporarily replace the legacy module-level generators with ``rng``'s streams."""
    if rng is None:
        yield
        return
    saved = pnv.rg, pnv.noise_rg, pmc.rg
    pnv.rg, pnv.noise_rg, pmc.rg = rng.transforms, rng.noise, rng.augment
    try:
        yield
    finally:
        pnv.rg, pnv.noise_rg, pmc.rg = saved


def _reset(transforms=fx.LEGACY_RG_SEED, noise=None, augment=None):
    """Reset legacy generators (works for tf Generators and injected streams alike)."""
    reset = lambda g, s: g.reset_from_seed(s) if hasattr(g, "reset_from_seed") else g.reset(s)
    reset(pnv.rg, transforms)
    reset(pmc.rg, transforms if augment is None else augment)
    if noise is not None:
        reset(pnv.noise_rg, noise)


def run_pipeline(video, fixture):
    """Body of the legacy ``process_fractals*`` loop, returning every stage."""
    config_ni, config_pp, config_traj = fx.legacy_configs(fixture, pnv, pmc, ptg)
    preproc_natural_image = pnv.preprocessing_fn(**config_ni)
    # return_kspace only repackages outputs (zfill -> dict); computation is unchanged.
    preproc_function = pmc.preprocessing_fn(**config_pp, return_kspace=True)
    roll_function = pmc.rolling_fn(**fx.ROLL_KWARGS, input_format=config_pp['input_format'],
                                   output_format=config_pp['output_format'])
    traj_function = ptg.create_traj_fn(**config_traj)

    video_u8 = tf.cast(tf.clip_by_value(video, 0, 255), tf.uint8)
    kspace = preproc_natural_image({'video': video_u8})
    stages = {'video_uint8': video_u8.numpy(), 'kspace': kspace['kspace'].numpy()}
    for element in traj_function(tf.data.Dataset.from_tensors(kspace)):
        zf, image = preproc_function(element)
        zfill_rolled, image_rolled = roll_function(zf['zfill'], image)
        stages.update(
            traj=element['traj']['traj'].numpy(),
            dcw=element['traj']['dcw'].numpy(),
            uskspace=zf['kspace'].numpy(),
            zfill=zf['zfill'].numpy(),
            image=image.numpy(),
            zfill_rolled=zfill_rolled.numpy(),
            image_rolled=image_rolled.numpy(),
        )
    return stages


def run_fixture(name, rng=None):
    """Return ``({case_name: stages}, meta)``; grouped fixtures have one case per slice.

    With ``rng`` (a ``synthmr.rng.RNG``) the legacy generators are replaced by its streams.
    """
    with injected_rng(rng):
        np.random.seed(fx.A_NP_SEED)
        _reset(noise=fx.LEGACY_NOISE_RG_SEED)
        if name == 'A':
            escape = fx.fixture_a_escape(fr)
            np.random.seed(fx.A_NP_SEED)
            video = fr.make_complex_fractal(escape)
            stages = run_pipeline(video, name)
            return {'A': {'fractal_escape': escape, 'fractal_video': video, **stages}}, {'c': fx.A_C}

        volumes, recipe = fx.fixture_b_volumes(gen, ff)
        if name == 'B_cc':
            stages = run_pipeline(fx.new_fractal_to_video(volumes[0]), name)
            return {'B_cc': {'fractal_volume': volumes[0], **stages}}, {'recipe': recipe}

        # B_grouped: every slice of the group starts from the same transform RNG state.
        out = {}
        seed = fx.group_seed('julia_00')
        for s, volume in enumerate(volumes, start=1):
            _reset(transforms=seed)
            stages = run_pipeline(fx.new_fractal_to_video(volume), name)
            out[f'B_grouped_slice_{s}'] = {'fractal_volume': volume, **stages}
        return out, {'recipe': recipe, 'group_seed': seed}
