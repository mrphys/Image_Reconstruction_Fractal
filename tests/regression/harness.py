"""Compare in-memory stages against golden and write old|new|diff GIFs."""

import numpy as np

from synthmr.viz.gifs import save_comparison_gif
from tests.regression.compare import as_complex, check_stage


def _frames(stage, x):
    """Time-first ``(T, H, W)`` view of a stage array for GIFs (None if not an image stage)."""
    x = as_complex(x)
    if stage in ("zfill", "image"):           # (H, W, T)
        return np.moveaxis(x, -1, 0)
    if stage in ("zfill_rolled", "image_rolled"):
        return x[..., 0] if x.ndim == 4 else x
    if stage == "video_uint8":
        return x.astype(np.float32)
    return None


def assert_matches_golden(golden_dir, case, stages, gif_dir=None, gif_stages=("zfill_rolled", "image_rolled")):
    """Check every stage present in both ``stages`` and the golden case; raise listing all failures."""
    case_dir = golden_dir / case
    failures, report = [], []
    for stage, new in stages.items():
        path = case_dir / f"{stage}.npy"
        if not path.exists():
            continue
        ref = np.load(path)
        ok, msg = check_stage(stage, ref, np.asarray(new))
        report.append(f"{'PASS' if ok else 'FAIL'} {case}/{stage}: {msg}")
        if not ok:
            failures.append(stage)
        if gif_dir is not None and stage in gif_stages:
            old_f, new_f = _frames(stage, ref), _frames(stage, np.asarray(new))
            if old_f is not None and old_f.shape == new_f.shape:
                save_comparison_gif(old_f, new_f, gif_dir / f"{case}_{stage}.gif", title=f"{case}/{stage}")
    print("\n".join(report))
    assert not failures, f"{case}: stages differ from golden: {failures}\n" + "\n".join(report)
