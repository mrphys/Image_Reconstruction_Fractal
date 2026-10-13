"""Turn a directory of phantom volumes (``*.npy``) into paired training data.

Examples::

    python scripts/make_dataset.py --preset fractal_new_2dt --input data/julia --output data/pairs
    python scripts/make_dataset.py --preset fractal_3dt_sos --config my_overrides.yaml \\
        --input data/julia4d --output data/pairs3d --gifs 3
    python scripts/make_dataset.py --list-presets
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # share the GPU with subprocesses


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", help="named preset (see --list-presets)")
    p.add_argument("--config", help="JSON/YAML overrides applied on top of the preset")
    p.add_argument("--input", help="directory of input volumes (*.npy)")
    p.add_argument("--output", help="output directory")
    p.add_argument("--limit", type=int, default=None, help="process at most N files")
    p.add_argument("--gifs", type=int, default=None, help="write GIFs for the first N samples")
    p.add_argument("--format", choices=["npy", "h5"], default=None)
    p.add_argument("--device", default=None, help="e.g. gpu:0 or cpu")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--print-config", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from synthmr.pipeline import PRESETS, Pipeline, load_config, save_sample

    if args.list_presets:
        print("\n".join(sorted(PRESETS)))
        return
    overrides = {k: v for k, v in {"device": args.device}.items() if v is not None}
    out_over = {k: v for k, v in {"gifs": args.gifs, "format": args.format}.items() if v is not None}
    if out_over:
        overrides["output"] = out_over
    cfg = load_config(args.preset, args.config, overrides)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return
    if not (args.input and args.output):
        raise SystemExit("--input and --output are required")

    import numpy as np
    files = sorted(Path(args.input).glob("*.npy"))[: args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    pipeline = Pipeline(cfg)
    for i, f in enumerate(files):
        sample = pipeline.run(np.load(f))
        sample.meta.update(source=str(f))
        save_sample(sample, out, f.stem, fmt=cfg.output.format, save_meta=cfg.output.save_meta)
        if i < cfg.output.gifs:
            from synthmr.viz.gifs import save_sample_gifs
            save_sample_gifs(sample, out / "gifs" / f.stem)
        print(f"[{i + 1}/{len(files)}] {f.name} -> input {sample.input.shape}, target {sample.target.shape}",
              flush=True)


if __name__ == "__main__":
    main()
