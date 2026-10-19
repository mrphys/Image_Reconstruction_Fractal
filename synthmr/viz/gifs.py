"""GIF writers for human inspection of volumes and stage comparisons.

All functions take numpy arrays with time on axis 0 and are backend-free.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


def to_display(x):
    """Magnitude of a complex array, or of a trailing real/imag channel pair."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.abs(x)
    return x.astype(np.float32)


def save_gif(frames, path, title=None, cmap="gray", fps=10, vlim=None):
    """Animate ``frames`` ``(T, H, W)`` (complex frames are shown as magnitude)."""
    save_panel_gif([frames], [title or ""], path, cmap=cmap, fps=fps, vlims=[vlim])


def save_panel_gif(panels, titles, path, cmap="gray", fps=10, dpi = 72, vlims=None, suptitle=None):
    """Animate several ``(T, H, W)`` arrays side by side with a shared time axis.

    Each panel gets its own fixed colour limits (``vlims[i]`` or its min/max).
    """
    panels = [to_display(p) for p in panels]
    n_frames = max(p.shape[0] for p in panels)
    vlims = vlims or [None] * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4), squeeze=False)
    images = []
    for ax, panel, title, vlim in zip(axes[0], panels, titles, vlims):
        lo, hi = vlim if vlim is not None else (float(panel.min()), float(panel.max()))
        images.append(ax.imshow(panel[0], cmap=cmap, vmin=lo, vmax=hi))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    header = fig.suptitle("", fontsize=8)
    fig.tight_layout()

    def update(i):
        for im, panel in zip(images, panels):
            im.set_data(panel[min(i, panel.shape[0] - 1)])
        header.set_text(f"{suptitle + '  ' if suptitle else ''}frame {i + 1}/{n_frames}")
        return [*images, header]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps).save(
        path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_comparison_gif(old, new, path, title=""):
    """``old | new | |old - new|`` for two ``(T, H, W)`` arrays."""
    old, new = to_display(old), to_display(new)
    diff = np.abs(old - new)
    lo, hi = float(min(old.min(), new.min())), float(max(old.max(), new.max()))
    save_panel_gif(
        [old, new, diff], ["old", "new", f"|diff| max={diff.max():.2e}"], path,
        vlims=[(lo, hi), (lo, hi), None], suptitle=title)


def _channels_to_display(x, complex_channels=True):
    """Magnitude of ``(..., ch)``: ``[re_1..re_C, im_1..im_C]`` if ``complex_channels`` (ch even), else
    C magnitude channels. Several coils are root-sum-of-squares combined."""
    x = np.asarray(x)
    if x.shape[-1] == 1:
        return x[..., 0]
    if complex_channels:
        k = x.shape[-1] // 2
        return np.sqrt(np.sum(x[..., :k] ** 2 + x[..., k:] ** 2, axis=-1))
    return np.sqrt(np.sum(x ** 2, axis=-1))


def save_slice_mosaic_gif(volume, path, title="", cols=None, vlim=None, fps=8, dpi=72):
    """Animate ``(T, Z, H, W)`` over time with all Z slices tiled in a grid."""
    v = to_display(volume)
    t, z, h, w = v.shape
    cols = cols or int(np.ceil(np.sqrt(z)))
    rows = int(np.ceil(z / cols))
    mosaic = np.zeros((t, rows * h, cols * w), dtype=np.float32)
    for s in range(z):
        r, c = divmod(s, cols)
        mosaic[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = v[:, s]
    save_panel_gif([mosaic], [title or f"{z} slices"], path, fps=fps, vlims=[vlim], dpi=dpi)


def save_ortho_gif(volume, path, title="", fps=8, dpi=72):
    """Animate ``(T, Z, H, W)`` over time as three orthogonal views through the centre."""
    v = to_display(volume)
    t, z, h, w = v.shape
    lim = (float(v.min()), float(v.max()))
    rep = max(1, h // (4 * z))   # stretch Z so thin stacks remain visible
    axial = v[:, z // 2]
    coronal = np.repeat(v[:, :, h // 2, :], rep, axis=1)
    sagittal = np.repeat(v[:, :, :, w // 2], rep, axis=1)
    save_panel_gif([axial, coronal, sagittal], ["axial (z mid)", "XZ (y mid)", "YZ (x mid)"], path,
                   vlims=[lim] * 3, suptitle=title, fps=fps, dpi=dpi)


def save_sample_gifs(sample, out_dir):
    """Input vs target GIFs for a :class:`synthmr.pipeline.Sample` (2D+t or 3D+t)."""
    out_dir = Path(out_dir)
    complex_input = sample.meta.get("input_format") != "abs"
    inp, tgt = _channels_to_display(sample.input, complex_input), _channels_to_display(sample.target)
    if inp.ndim == 3:
        save_panel_gif([inp, tgt], ["input |.|", "target |.|"], out_dir / "pair.gif")
    else:
        save_slice_mosaic_gif(inp, out_dir / "input_mosaic.gif", "input |.|")
        save_slice_mosaic_gif(tgt, out_dir / "target_mosaic.gif", "target |.|")
        save_ortho_gif(tgt, out_dir / "target_ortho.gif", "target")
