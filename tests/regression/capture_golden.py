"""Capture golden stage outputs of the OLD (legacy) pipelines on fixed fractals.

Usage (from the repo root)::

    python tests/regression/capture_golden.py --device cpu
    python tests/regression/capture_golden.py --device gpu --out tests/artifacts/golden_rerun

Every stage of every fixture (see ``fixtures.py``) is written as ``<stage>.npy``
under ``<out>/<device>/<fixture>/`` together with ``manifest.json`` (shapes,
dtypes, sha256) and GIFs for visual inspection.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--out", default=str(ROOT / "tests/artifacts/golden"))
    p.add_argument("--fixtures", nargs="+", default=["A", "B_cc", "B_grouped"])
    p.add_argument("--no-gifs", action="store_true")
    return p.parse_args()


args = parse_args()
if args.device == "cpu":
    # Must happen before TensorFlow is imported. The legacy code wraps itself in
    # tf.device('/gpu:0'); soft placement then falls back to the CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from synthmr.viz.gifs import save_gif, save_panel_gif  # noqa: E402
from tests.regression.legacy_runner import run_fixture  # noqa: E402


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_case(case_dir, stages, meta, gifs):
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'meta': meta, 'stages': {}}
    for stage, arr in stages.items():
        np.save(case_dir / f'{stage}.npy', arr)
        manifest['stages'][stage] = {'shape': list(arr.shape), 'dtype': str(arr.dtype),
                                     'sha256': sha256(arr)}
    (case_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=float))
    if gifs:
        write_gifs(case_dir / 'gifs', stages)


def coil_sos(kspace_ctyx):
    """Root-sum-of-squares image ``(T, H, W)`` from ``(T, C, H, W)`` Cartesian k-space."""
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_ctyx, axes=(-2, -1))), axes=(-2, -1))
    return np.sqrt((np.abs(img) ** 2).sum(axis=1))


def complex_from_channels(x):
    """``(..., 2)`` real/imag channels -> complex; single channel -> as is."""
    return x[..., 0] + 1j * x[..., 1] if x.shape[-1] == 2 else x[..., 0]


def write_gifs(gif_dir, s):
    video = s['video_uint8'].astype(np.float32)
    save_panel_gif([video[..., 0], video[..., 1]], ['video ch0 (real)', 'video ch1 (imag)'],
                   gif_dir / '01_video.gif')
    save_gif(coil_sos(s['kspace'][0]), gif_dir / '02_kspace_sos.gif', 'fully-sampled multicoil (SoS)')
    zfill = complex_from_channels(s['zfill'])          # (H, W, T)
    image = s['image'] if s['image'].ndim == 3 else complex_from_channels(s['image'])
    save_panel_gif([np.moveaxis(zfill, -1, 0), np.moveaxis(image, -1, 0)],
                   ['zero-filled input |.|', 'target |.|'], gif_dir / '03_zfill_vs_image.gif')
    zr = complex_from_channels(s['zfill_rolled'])      # (T, H, W)
    ir = complex_from_channels(s['image_rolled'])
    save_panel_gif([zr, ir], ['input (rolled/rotated)', 'target (rolled/rotated)'],
                   gif_dir / '04_final_pair.gif')


def main():
    if args.device == 'gpu':
        assert tf.config.list_physical_devices('GPU'), 'no GPU visible'
    out_root = Path(args.out) / args.device
    for name in args.fixtures:
        print(f'[{args.device}] fixture {name} ...', flush=True)
        cases, meta = run_fixture(name)
        for case, stages in cases.items():
            write_case(out_root / case, stages, meta, gifs=not args.no_gifs)
            print(f'  wrote {out_root / case}: ' + ', '.join(
                f'{k}{list(v.shape)}' for k, v in stages.items()), flush=True)


if __name__ == '__main__':
    main()
