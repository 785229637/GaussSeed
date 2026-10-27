"""Command-line interface for standalone Gaussian prediction."""

import argparse

from .io import load_cameras, load_images, save_npz, save_preview_ply
from .predictor import GaussSeed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict prior 3D Gaussians from a calibrated image pair."
    )
    parser.add_argument("--checkpoint", required=True, help="Full encoder checkpoint")
    parser.add_argument("--images", nargs=2, required=True, metavar=("IMAGE0", "IMAGE1"))
    parser.add_argument("--camera-json", required=True, help="Two-view camera parameters")
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--ply", help="Optional RGB point-preview PLY")
    parser.add_argument("--experiment", default="re10k")
    parser.add_argument("--config-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", nargs=2, type=int, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--near", type=float)
    parser.add_argument("--far", type=float)
    parser.add_argument("--max-ply-points", type=int, default=50000)
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override")
    parser.add_argument("--no-strict", action="store_true", help="Allow checkpoint key mismatch")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    images, source_sizes = load_images(args.images, tuple(args.image_size))
    intrinsics, extrinsics, camera_near, camera_far = load_cameras(
        args.camera_json, source_sizes
    )
    near = camera_near if args.near is None else args.near
    far = camera_far if args.far is None else args.far

    initializer = GaussSeed(
        args.checkpoint,
        experiment=args.experiment,
        config_dir=args.config_dir,
        device=args.device,
        overrides=args.override,
        strict=not args.no_strict,
    )
    context = initializer.make_context(
        images, intrinsics, extrinsics, near=near, far=far
    )
    prediction = initializer.predict(context)
    npz_path = save_npz(prediction, args.output)
    print(f"Saved Gaussian parameters: {npz_path}")
    if args.ply:
        ply_path = save_preview_ply(
            prediction, args.ply, max_points=args.max_ply_points
        )
        print(f"Saved point preview: {ply_path}")
    print(f"Predicted Gaussians: {prediction.gaussians.means.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
