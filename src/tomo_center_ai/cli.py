"""CLI: run the AI rotation-center picker over a folder of pre-reconstructed TIFFs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from tomo_center_ai.io_tiff import load_folder
from tomo_center_ai.ai.inference import inference_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tomo-center-ai",
        description=(
            "Pick the best rotation-axis center from a folder of pre-reconstructed "
            "TIFF slices using the tomocupy DINOv2 classifier."
        ),
    )
    p.add_argument(
        "folder",
        type=Path,
        help="Folder containing one TIFF per candidate center.",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help=(
            "Path to the classifier checkpoint (.pt). Download from "
            "https://anl.box.com/s/4o8qcig6pl9k8p7x4z3qqbrpgnjipolq"
        ),
    )
    p.add_argument(
        "--centers-file",
        type=Path,
        default=None,
        help=(
            "Text file with one candidate center per line, in the same sorted order "
            "as the TIFFs. If omitted, centers are parsed from the filenames "
            "(last numeric token in each stem)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for center_of_rotation.txt (default: cwd).",
    )

    # Inference knobs — kept identical to the upstream `args.infer_*` schema.
    p.add_argument("--downsample-factor", type=int, nargs="+", default=[1],
                   help="Downsample factor(s) per scale (default: [1]).")
    p.add_argument("--num-windows", type=int, nargs="+", default=[1],
                   help="Number of random patches per scale (default: [1]).")
    p.add_argument("--window-size", type=int, nargs="+", default=[224],
                   help="Patch size(s) per scale (default: [224]).")
    p.add_argument("--use-8bits", action="store_true",
                   help="Requantize to 8 bits before inference.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    p.add_argument("--save-intermediate", action="store_true",
                   help="Also save predicts_all.npz to --out-dir.")
    p.add_argument(
        "--plot",
        nargs="?",
        type=Path,
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save a PNG of per-slice probability vs. candidate center. "
            "Without a value, writes to <out-dir>/scores.png. "
            "Requires the [plot] extra: pip install -e '.[plot]'."
        ),
    )
    return p


def _validate_scale_lists(args: argparse.Namespace) -> None:
    n = len(args.downsample_factor)
    for name, lst in (("--num-windows", args.num_windows),
                      ("--window-size", args.window_size)):
        if len(lst) != n:
            raise SystemExit(
                f"{name} has {len(lst)} values but --downsample-factor has {n}; "
                "they must match (one entry per scale)."
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_scale_lists(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TIFFs from {args.folder} ...")
    stack, centers, paths = load_folder(args.folder, args.centers_file)
    print(f"  {len(paths)} slices, shape {stack.shape[1:]}, dtype {stack.dtype}")
    print(f"  center range: {min(centers)} .. {max(centers)}")

    infer_args = SimpleNamespace(
        infer_use_8bits=args.use_8bits,
        infer_downsample_factor=list(args.downsample_factor),
        infer_num_windows=list(args.num_windows),
        infer_window_size=list(args.window_size),
        infer_seed_number=args.seed,
        infer_model_path=str(args.model_path),
        infer_save_intermediate_data=args.save_intermediate,
    )

    result = inference_pipeline(
        infer_args,
        stack,
        centers,
        str(args.out_dir),
    )
    chosen = result["centers"]

    print("\nBest center(s):")
    for c in chosen:
        print(f"  {c:.1f}")
    print(f"\nAppended to {args.out_dir / 'center_of_rotation.txt'}")

    if args.plot is not None:
        plot_path = (
            args.out_dir / "scores.png"
            if str(args.plot) == "__default__"
            else args.plot
        )
        _plot_scores(result["candidates"], result["scores"], chosen, plot_path)
        print(f"Wrote score plot to {plot_path}")

    return 0


def _plot_scores(candidates, scores, chosen, out_path: Path) -> None:
    """Save a probability-vs-candidate-center plot. Lazy-imports matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend; no display needed
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "--plot requires matplotlib. Install with: pip install -e '.[plot]' "
            "(or: pip install matplotlib)"
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by candidate center so the line connects in monotonic order.
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
    # Dedupe legend entries (one axvline per picked center would clutter).
    handles, labels = ax.get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
