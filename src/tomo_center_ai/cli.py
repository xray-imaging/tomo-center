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

    chosen = inference_pipeline(
        infer_args,
        stack,
        centers,
        str(args.out_dir),
    )

    print("\nBest center(s):")
    for c in chosen:
        print(f"  {c:.1f}")
    print(f"\nAppended to {args.out_dir / 'center_of_rotation.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
