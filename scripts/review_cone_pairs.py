"""Review plots for make_cone_pairs.py (one sample)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from synthmr.viz.gifs import save_panel_gif  # noqa: E402

PLANES = ("axial (Z)", "coronal (Y)", "sagittal (X)")


def ortho(v):
    """Central (Z, Y, X) planes of a ``(Z, Y, X)`` volume."""
    z, y, x = (s // 2 for s in v.shape)
    return v[z], v[:, y], v[:, :, x]


def _show(ax, img, cmap, vmin, vmax, title=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return im


def phantom_ortho(path, block, frames=(0, 16, 31)):
    mag_max = float(np.abs(block).max())
    fig, axes = plt.subplots(2 * len(frames), 3, figsize=(7.5, 2.4 * 2 * len(frames)), constrained_layout=True)
    for i, t in enumerate(frames):
        for j, plane in enumerate(ortho(block[t])):
            im_m = _show(axes[2 * i, j], np.abs(plane), "gray", 0, mag_max, f"t={t} |x| {PLANES[j]}")
            im_p = _show(axes[2 * i + 1, j], np.angle(plane), "twilight", -np.pi, np.pi, f"t={t} ∠x {PLANES[j]}")
        fig.colorbar(im_m, ax=axes[2 * i], shrink=0.8)
        fig.colorbar(im_p, ax=axes[2 * i + 1], shrink=0.8)
    fig.suptitle("Phantom (raw fractal block, before mask / background phase)", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pair_ortho(path, sim, frames=(0, 16), dpi=220):
    tgt, inp = sim["target"], sim["input"]
    fig, axes = plt.subplots(3 * len(frames), 4, figsize=(14, 3.4 * 3 * len(frames)), constrained_layout=True)
    for i, t in enumerate(frames):
        mag = np.abs(inp[t])
        a = float((mag * tgt[t]).sum() / max((mag * mag).sum(), 1e-30))      # LS scale of |input| onto target
        cols = [(tgt[t], "gray", 0, 1, "target"), (mag, "gray", 0, float(mag.max()), "|input|"),
                (np.angle(inp[t]), "twilight", -np.pi, np.pi, "∠input"),
                (np.abs(a * mag - tgt[t]), "magma", 0, 0.5, f"|{a:.2f}·|input| − target|")]
        for j, (vol, cmap, lo, hi, name) in enumerate(cols):
            for r, plane in enumerate(ortho(vol)):
                im = _show(axes[3 * i + r, j], plane, cmap, lo, hi, f"t={t} {name} {PLANES[r]}")
            fig.colorbar(im, ax=axes[3 * i:3 * i + 3, j], shrink=0.6, location="bottom")
    fig.suptitle(f"Pair: target vs gridding zero-fill (SNR {sim['snr_db']:.1f} dB)", fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def coils(path, maps, title="Simulated coil maps"):
    c = maps.shape[0]
    z = maps.shape[1] // 2
    fig, axes = plt.subplots(2, c + 1, figsize=(1.6 * (c + 1), 3.6), constrained_layout=True)
    mmax = float(np.abs(maps[:, z]).max())
    for i in range(c):
        im_m = _show(axes[0, i], np.abs(maps[i, z]), "gray", 0, mmax, f"|S{i}|")
        im_p = _show(axes[1, i], np.angle(maps[i, z]), "twilight", -np.pi, np.pi, f"∠S{i}")
    sos = np.sqrt((np.abs(maps[:, z]) ** 2).sum(0))
    _show(axes[0, c], sos, "gray", 0, float(sos.max()), "SoS")
    axes[1, c].axis("off")
    fig.colorbar(im_m, ax=axes[0, :c], shrink=0.8)
    fig.colorbar(im_p, ax=axes[1, :c], shrink=0.8)
    fig.suptitle(f"{title}, central axial plane (z={z})", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def trajectory(path, coord, readout=711, every=20):
    one = coord[0].reshape(-1, readout, 3)
    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    for il in one[::every]:
        ax3.plot(il[:, 2], il[:, 1], il[:, 0], lw=0.6, color="#3b6fb6", alpha=0.7)
    ax3.set_xlabel("kx")
    ax3.set_ylabel("ky")
    ax3.set_zlabel("kz")
    kmax = float(np.round(np.abs(coord[0]).max()))
    ax3.set_title(f"frame 0: every {every}th of\n{one.shape[0]} interleaves ({2 * kmax:.0f}³ units)", fontsize=8)
    bins = np.linspace(-kmax, kmax, 129)
    pairs = [((2, 1), "kx–ky"), ((2, 0), "kx–kz"), ((1, 0), "ky–kz")]
    for r, (pts, label) in enumerate([(coord[0], "frame 0"), (coord.reshape(-1, 3), f"all {coord.shape[0]} frames")]):
        for j, ((u, v), name) in enumerate(pairs):
            ax = fig.add_subplot(gs[r, j + 1])
            h, _, _ = np.histogram2d(pts[:, v], pts[:, u], bins=[bins, bins])
            im = ax.imshow(np.log10(1 + h), origin="lower", extent=(-kmax, kmax, -kmax, kmax), cmap="viridis")
            ax.set_title(f"{label}: {name} projection", fontsize=8)
            ax.set_xlabel(name.split("–")[0], fontsize=7)
            ax.set_ylabel(name.split("–")[1], fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, shrink=0.8, label="log10(1 + samples)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pair_time(path, sim, dpi=150):
    tgt, mag = sim["target"], np.abs(sim["input"])
    z, y, x = (s // 2 for s in tgt.shape[1:])
    panels = [tgt[:, z], mag[:, z], tgt[:, :, y], mag[:, :, y], tgt[:, :, :, x], mag[:, :, :, x]]
    titles = ["target axial", "|input| axial", "target coronal", "|input| coronal", "target sagittal",
              "|input| sagittal"]
    vlims = [(0, 1), (0, float(mag.max()))] * 3
    save_panel_gif([np.flip(p, 1) for p in panels], titles, path, vlims=vlims, fps=6, dpi=dpi)


def write_review(out_dir, block, sim, coord):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phantom_ortho(out_dir / "phantom_ortho.png", block)
    pair_ortho(out_dir / "pair_ortho.png", sim)
    coils(out_dir / "coils.png", sim["maps"])
    coils(out_dir / "coils_est.png", np.moveaxis(sim["est_maps"], -1, 0),
          "Estimated time_avg_tukey_3d maps (input combine)")
    trajectory(out_dir / "trajectory.png", coord)
    pair_time(out_dir / "pair_time.gif", sim)
