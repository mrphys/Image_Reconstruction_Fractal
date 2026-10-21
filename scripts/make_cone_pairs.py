"""Paired 3D+t samples on the Extreme MRI DCE cone trajectory (docs/plans/CONE_PAIRS_PLAN.md).

Phantoms come from the random ``3dt`` recipe protocol on an isotropic n^3 grid (``--size``, default 64). The demo's
cone trajectory is rescaled to n^3 and binned into ``--frames-total`` contiguous interleave bins, of which the first
``--frames`` are used (one per phantom frame). Simulated 3D coils, k-space AWGN, gridding zero-fill
(``nufft_adjoint(k * dcf)``) combined with maps estimated from the data (``time_avg_tukey_3d``, as the stack-of-stars
3D+t pairs); target = SoS of the coil images. Everything is saved as float16.

Example::

    python scripts/make_cone_pairs.py --n 1 --seed 0 --plots --device 0
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmr.backend.sigpy.recon3d import combine_with_maps, time_avg_tukey_maps  # noqa: E402
from synthmr.phantoms.generate import FractalDomain, _write_meta, iter_recipes_3dt  # noqa: E402
from synthmr.recon import ReconConfig  # noqa: E402
from synthmr.rng import RNG  # noqa: E402
from synthmr.simulation.coils import simulate_coils_3d  # noqa: E402
from synthmr.simulation.masks import ellipsoid_mask_3d  # noqa: E402
from synthmr.simulation.phase import background_phase_3d  # noqa: E402

RES = Path(__file__).resolve().parents[2] / "app" / "resources"
CONE = RES / "4d_extreme_mri" / "3D_cone_data"
# Stack-of-stars 3D+t recon settings (new_julia_grasp_paired); Tukey width made isotropic for the isotropic grid.
MAPS_CFG = ReconConfig(coil_maps="time_avg_tukey_3d", tukey_width=(0.25, 0.25, 0.25), tukey_alpha=0.5,
                       anchor_phase=True)


def cone_domain(extent, frames, n=64):
    """Isotropic n^3 domain: voxel ``v = extent / n``; bounds +-(n-1)/2 v make the linspace step equal ``v``."""
    v = extent / n
    half = (n - 1) / 2 * v
    return FractalDomain(shape=(n, n), slices=n, slice_step=v, bounds=((-half, half),) * 2, frames=frames,
                         time_range=(-0.3, 0.3), max_iter=30)


def load_trajectory(frames_total, frames, n=64):
    """Rescaled coord bins ``(frames, M, 3)`` and dcf ``(frames, M)`` plus the trajectory meta."""
    import sigpy as sp
    coord = np.load(CONE / "coord.npy", mmap_mode="r")
    dcf = np.load(CONE / "dcf.npy", mmap_mode="r")
    native = sp.estimate_shape(coord)
    scale = np.array([n / m for m in native], dtype=np.float32)
    per_bin = coord.shape[0] // frames_total
    used = per_bin * frames
    c = (np.asarray(coord[:used]) * scale).reshape(frames, per_bin * coord.shape[1], 3)
    d = np.asarray(dcf[:used]).reshape(frames, per_bin * dcf.shape[1])
    meta = {"source": str(CONE), "native_shape": list(native), "scale": scale.tolist(),
            "interleaves": coord.shape[0], "readout": coord.shape[1], "frames_total": frames_total,
            "frames_used": frames, "interleaves_per_bin": per_bin, "samples_per_frame": int(c.shape[1]),
            "nufft": {"oversamp": 1.25, "width": 4}}
    return c, d, meta


def simulate_pair(phantom, coord, dcf, rng, device, regsnr=(12, 22)):
    """``phantom`` (T, Z, Y, X) complex -> dict(input (T,Z,Y,X) complex, target (T,Z,Y,X), maps, mask, ...).

    Input: coil zero-fill combined with maps estimated from the data (not masked); target: SoS of the noise-free coil
    images, as the stack-of-stars 3D+t pairs (``coil_maps='time_avg_tukey_3d'``, ``target_combine='sos'``).
    """
    import sigpy as sp
    xp = device.xp
    gen = rng.transforms
    shape = phantom.shape[1:]
    # Complex object as in prepare_video (image_adjustment (0, 1), complex_transform 1): (re, im) min-max scaled to
    # [0, 1] jointly, then a random channel roll. The raw block's +-pi phase speckle would dominate outer k-space.
    v = np.stack([phantom.real, phantom.imag], -1)
    v = np.roll((v - v.min()) / (v.max() - v.min()), int(gen.uniform(shape=(), minval=0, maxval=2, dtype="int32")), -1)
    phantom = v[..., 0] + 1j * v[..., 1]
    mask = ellipsoid_mask_3d(shape, gen, slice_spacing=1)
    maps = simulate_coils_3d(gen, shape, sigma_coil=(2, 6), num_coils=(12,), add_phase=2, ngrid=2,
                             slice_spacing=1) * mask[None]
    obj = (phantom * mask[None] * np.exp(1j * background_phase_3d(shape, gen, ngrid=6, slice_spacing=1))[None])
    obj = obj.astype(np.complex64)
    snr = float(np.asarray(gen.uniform(shape=(), minval=regsnr[0], maxval=regsnr[1])))

    with device:
        S = sp.to_device(maps, device)
        sos_max = max(float(xp.sqrt((xp.abs(S * sp.to_device(f, device)) ** 2).sum(0)).max()) for f in obj)
        # SNR relative to the mean coil-image power, as in sim3d. sigpy's NUFFT is unitary-scaled, so this is the
        # Cartesian per-sample noise level (k-space sample power would be dominated by the dense cone centre).
        ksp, power = [], 0.0
        target = np.empty(obj.shape, np.float32)
        for t in range(obj.shape[0]):
            x = S * sp.to_device(obj[t], device) / sos_max
            power += float((xp.abs(x) ** 2).mean()) / obj.shape[0]
            target[t] = sp.to_device(xp.sqrt((xp.abs(x) ** 2).sum(0)), sp.cpu_device)      # SoS (target_combine sos)
            ksp.append(sp.nufft(x, sp.to_device(coord[t], device)))
        std = float(np.sqrt(power / 10 ** (snr / 10)))
        oshape = (maps.shape[0],) + shape

        def zero_fill(t):
            return sp.nufft_adjoint(ksp[t] * sp.to_device(dcf[t], device), sp.to_device(coord[t], device),
                                    oshape=oshape)

        zsum = 0
        for t, k in enumerate(ksp):
            noise = (np.asarray(rng.noise.normal(shape=k.shape, stddev=std))
                     + 1j * np.asarray(rng.noise.normal(shape=k.shape, stddev=std))).astype(np.complex64)
            ksp[t] = k + sp.to_device(noise, device)
            zsum = zsum + zero_fill(t)
        est = estimate_maps(zsum, device)                                                 # (Z, Y, X, C)
        combined = np.empty(obj.shape, np.complex64)
        for t in range(len(ksp)):
            combined[t] = sp.to_device(combine_with_maps(xp.moveaxis(zero_fill(t), 0, -1), est), sp.cpu_device)
    t_scale, i_scale = float(target.max()), float(np.abs(combined).max())
    return {"input": combined / i_scale, "target": target / t_scale, "maps": maps, "est_maps": sp.to_device(est),
            "mask": mask, "object": obj, "snr_db": snr, "noise_std": std, "coil_sos_max": sos_max,
            "target_scale": t_scale, "input_scale": i_scale}


def estimate_maps(zsum, device):
    """``time_avg_tukey_3d`` maps (as in the stack-of-stars 3D+t data) from the time-summed zero-fill ``(C, Z, Y, X)``.

    Walsh needs ``eigh``; falls back to the CPU when cuSOLVER is unusable (CUDA 11.2 libs on sm_89 in the TF image).
    """
    import sigpy as sp
    from synthmr.backend.sigpy import core
    try:
        with core.device(f"gpu:{device.id}" if device.id >= 0 else "cpu"):
            return time_avg_tukey_maps(zsum[None], MAPS_CFG)
    except Exception as e:  # noqa: BLE001
        if "CUSOLVER" not in type(e).__name__.upper():
            raise
    with core.device("cpu"):
        return sp.to_device(time_avg_tukey_maps(sp.to_device(zsum)[None], MAPS_CFG), device)


def nrmse(x, ref):
    """NRMSE of ``|x|`` (least-squares scaled onto ``ref``) w.r.t. ``ref``."""
    x = np.abs(x).ravel().astype(np.float64)
    ref = ref.ravel().astype(np.float64)
    a = (x @ ref) / max(x @ x, 1e-30)
    return float(np.linalg.norm(a * x - ref) / np.linalg.norm(ref))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extent", type=float, default=2.4, help="fractal-domain side length (voxel = extent / size)")
    p.add_argument("--frames-total", type=int, default=100)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--size", type=int, default=64, help="isotropic grid size n (n^3 voxels)")
    p.add_argument("--out", default=None, help="default: new_julia_grasp_cone_paired[_<size> if size != 64]")
    p.add_argument("--phantom-out", default=None, help="default: julia_cone_3dt[_<size> if size != 64]")
    p.add_argument("--plots", action="store_true", help="write review plots for the first sample")
    p.add_argument("--device", type=int, default=0, help="cupy device id; -1 = CPU")
    a = p.parse_args(argv)

    import h5py
    import sigpy as sp
    device = sp.cpu_device
    if a.device >= 0:
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > a.device:
                device = sp.Device(a.device)
        except Exception as e:  # noqa: BLE001
            print(f"no GPU ({e}); using CPU")
    print(f"device: {device}")

    N = a.size
    suffix = "" if N == 64 else f"_{N}"
    out = Path(a.out or RES / f"new_julia_grasp_cone_paired{suffix}")
    ph_out = Path(a.phantom_out or RES / f"julia_cone_3dt{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    ph_out.mkdir(parents=True, exist_ok=True)
    coord, dcf, traj_meta = load_trajectory(a.frames_total, a.frames, N)
    print(f"trajectory: native {traj_meta['native_shape']} -> {N}^3, {traj_meta['samples_per_frame']} samples/frame "
          f"({traj_meta['samples_per_frame'] / N ** 3:.2f} x {N}^3)")
    domain = cone_domain(a.extent, a.frames, N)

    for k, recipe, block in iter_recipes_3dt(domain, a.n, a.seed, norm="volume"):
        t0 = time.time()
        name = f"julia_{k:03d}"
        ph_path = ph_out / f"{name}.npy"
        np.save(ph_path, np.stack([block.real, block.imag], -1).astype(np.float16))
        meta = {"mode": "3dt", "recipe": recipe, "domain": asdict(domain), "norm": "volume", "extent": a.extent,
                "size": N, "voxel": a.extent / N, "seed": a.seed}
        _write_meta(ph_path, meta)

        seeds = {"transforms": 2 * (a.seed * 1_000_000 + k) + 1, "noise": 2 * (a.seed * 1_000_000 + k) + 2}
        sim = simulate_pair(block, coord, dcf, RNG(seeds=seeds, kind="numpy"), device)
        inp = np.stack([sim["input"].real, sim["input"].imag], -1).astype(np.float16)
        tgt = sim["target"][..., None].astype(np.float16)
        with h5py.File(out / f"{name}.h5", "w") as f:
            f.create_dataset("input", data=inp)
            f.create_dataset("output", data=tgt)
        (out / f"{name}.json").write_text(json.dumps({
            **meta, "phantom": str(ph_path), "trajectory": traj_meta, "coil_seeds": seeds,
            "simulation": {"num_coils": 12, "sigma_coil": [2, 6], "add_phase": 2, "coil_phase_ngrid": 2,
                           "object_phase_ngrid": 6,
                           "object": "prepare_video min-max (re, im), complex_transform 1", "regsnr": [12, 22],
                           "noise": "complex AWGN on k-space samples, SNR vs mean coil-image power",
                           "coil_maps": "time_avg_tukey_3d", "tukey_width": list(MAPS_CFG.tukey_width),
                           "tukey_alpha": MAPS_CFG.tukey_alpha, "anchor_phase": MAPS_CFG.anchor_phase,
                           "input_combine": "maps", "target_combine": "sos"},
            "snr_db": sim["snr_db"], "noise_std": sim["noise_std"], "coil_sos_max": sim["coil_sos_max"],
            "target_scale": sim["target_scale"], "input_scale": sim["input_scale"], "dtype": "float16",
            "layout": {"input": "(T, Z, Y, X, 2) re/im", "output": "(T, Z, Y, X, 1) magnitude"},
        }, indent=2, default=lambda o: getattr(o, "tolist", str)(o)))
        single = float(np.mean([nrmse(sim["input"][t], sim["target"][t]) for t in range(a.frames)]))
        avg = nrmse(sim["input"].mean(0), sim["target"].mean(0))
        print(f"{name}: snr {sim['snr_db']:.1f} dB, NRMSE single frame (mean) {single:.3f}, "
              f"32-frame average {avg:.3f}, {time.time() - t0:.1f}s")
        if a.plots and k == 0:
            from review_cone_pairs import write_review
            write_review(out / "review", block, sim, coord)
            print(f"review plots -> {out / 'review'}")


if __name__ == "__main__":
    main()
