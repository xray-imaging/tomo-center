"""tomo-center CLI: `find` (inference) and `train` (fine-tune) subcommands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from tomo_center import logging as tca_logging
from tomo_center.ai.inference import inference_pipeline
from tomo_center.io_tiff import load_folder

log = tca_logging.getLogger(__name__)


# ---------- top-level parser --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tomo-center",
        description=(
            "DINOv2-based rotation-axis center picker for pre-reconstructed TIFF "
            "slices. Use `find` to pick the best center from a sweep, or `train` "
            "to fine-tune the classifier on labeled slices."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{find,train}")

    _add_find_parser(sub)
    _add_train_parser(sub)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tca_logging.setup_custom_logger()
    return args.func(args)


# ---------- find --------------------------------------------------------------

def _add_find_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "find",
        help="Pick the best center from a folder of reconstructed slices.",
        description=(
            "Score each TIFF in the folder (one slice per candidate center) and "
            "report the candidate(s) with the highest 'correct center' probability."
        ),
    )
    p.add_argument("folder", type=Path,
                   help="Folder containing one TIFF per candidate center.")
    p.add_argument("--model-path", type=Path, required=True,
                   help="Classifier checkpoint (.pt). Download from "
                        "https://anl.box.com/s/4o8qcig6pl9k8p7x4z3qqbrpgnjipolq")
    p.add_argument("--centers-file", type=Path, default=None,
                   help="One float per line, in TIFF-sorted order. If omitted, "
                        "centers are parsed from the filenames (last numeric token).")
    p.add_argument("--out-dir", type=Path, default=Path.cwd(),
                   help="Output directory (default: cwd).")
    p.add_argument("--downsample-factor", type=int, nargs="+", default=[1])
    p.add_argument("--num-windows", type=int, nargs="+", default=[1])
    p.add_argument("--window-size", type=int, nargs="+", default=[224])
    p.add_argument("--use-8bits", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-intermediate", action="store_true")
    p.add_argument("--plot", nargs="?", type=Path, const="__default__", default=None,
                   metavar="PATH",
                   help="Save (and display, if a GUI is available) a score-vs-center "
                        "PNG. Without a value, writes to <out-dir>/scores.png.")
    p.set_defaults(func=cmd_find)


def _validate_scale_lists(args: argparse.Namespace) -> None:
    n = len(args.downsample_factor)
    for name, lst in (("--num-windows", args.num_windows),
                      ("--window-size", args.window_size)):
        if len(lst) != n:
            raise SystemExit(
                f"{name} has {len(lst)} values but --downsample-factor has {n}; "
                "they must match (one entry per scale)."
            )


def cmd_find(args: argparse.Namespace) -> int:
    _validate_scale_lists(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading TIFFs from %s ...", args.folder)
    stack, centers, paths = load_folder(args.folder, args.centers_file)
    log.info("  %d slices, shape %s, dtype %s", len(paths), stack.shape[1:], stack.dtype)
    log.info("  center range: %s .. %s", min(centers), max(centers))

    infer_args = SimpleNamespace(
        infer_use_8bits=args.use_8bits,
        infer_downsample_factor=list(args.downsample_factor),
        infer_num_windows=list(args.num_windows),
        infer_window_size=list(args.window_size),
        infer_seed_number=args.seed,
        infer_model_path=str(args.model_path),
        infer_save_intermediate_data=args.save_intermediate,
    )
    result = inference_pipeline(infer_args, stack, centers, str(args.out_dir))
    chosen = result["centers"]

    log.info("Best center(s):")
    for c in chosen:
        log.info("  %.1f", c)

    if args.plot is not None:
        plot_path = (args.out_dir / "scores.png"
                     if str(args.plot) == "__default__" else args.plot)
        _plot_scores(result["candidates"], result["scores"], chosen, plot_path)
        log.info("Wrote score plot to %s", plot_path)

    return 0


def _plot_scores(candidates, scores, chosen, out_path: Path) -> None:
    """Save (and display, if a GUI is available) the score-vs-center plot."""
    import os
    try:
        import matplotlib
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "--plot requires matplotlib. Install with: pip install -e '.[plot]' "
            "(or: pip install matplotlib)"
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = sorted(range(len(candidates)), key=lambda i: candidates[i])
    xs = [candidates[i] for i in order]
    ys = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, "-o", markersize=4, linewidth=1, label="P(correct center)")
    for c in chosen:
        ax.axvline(c, linestyle="--", linewidth=1, alpha=0.7,
                   label=f"picked: {c:.2f}")
    ax.set_xlabel("Candidate center of rotation (px)")
    ax.set_ylabel("Score (softmax of class-1 logit)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-slice probability of being the correct center")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ---------- train -------------------------------------------------------------

def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "train",
        help="Fine-tune the CoR classifier on labeled slices.",
        description=(
            "Fine-tune the DINOv2 + attention-pool classifier on a folder of "
            "labeled tomographic slices. Input layout: <LABELS_DIR>/centered/*.tif "
            "and <LABELS_DIR>/off_centered/*.tif."
        ),
    )
    p.add_argument("labels_dir", type=Path,
                   help="Directory containing centered/ and off_centered/ subfolders of TIFFs.")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write the best checkpoint (.pt).")
    p.add_argument("--resume", type=Path, default=None,
                   help="Existing checkpoint to fine-tune from. If omitted, the "
                        "backbone is downloaded from Meta's torch.hub (requires internet).")
    p.add_argument("--base-model", default="dinov2_vitb14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
                   help="DINOv2 variant for the backbone (default: dinov2_vitb14). "
                        "Only used when --resume is not given.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="Linear warmup steps before cosine decay (default: 50).")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of labeled slices held out for validation (default: 0.1).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu (default: cuda; falls back to cpu if unavailable).")
    p.add_argument("--window-size", type=int, default=224,
                   help="Crop size; must match the value used at inference time.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-augment", action="store_true",
                   help="Disable random flip + small crop offset on the training set.")
    p.set_defaults(func=cmd_train)


def cmd_train(args: argparse.Namespace) -> int:
    # Lazy import — train module pulls in torch.utils.data etc., not needed for `find`.
    from tomo_center.train import run_training
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
