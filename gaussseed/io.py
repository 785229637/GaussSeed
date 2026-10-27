"""Input and output helpers for the standalone predictor."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_images(
    paths: Sequence[str | Path], image_size: tuple[int, int]
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    if len(paths) != 2:
        raise ValueError("The initializer currently expects exactly two images")

    images = []
    source_sizes = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_sizes.append((image.height, image.width))
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        images.append(tensor)
    return torch.stack(images), source_sizes


def load_cameras(
    path: str | Path, source_sizes: Sequence[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intrinsics = torch.as_tensor(data["intrinsics"], dtype=torch.float32)
    extrinsics = torch.as_tensor(data["extrinsics"], dtype=torch.float32)
    if intrinsics.shape != (2, 3, 3):
        raise ValueError("camera JSON intrinsics must have shape [2, 3, 3]")
    if extrinsics.shape != (2, 4, 4):
        raise ValueError("camera JSON extrinsics must have shape [2, 4, 4]")

    if not bool(data.get("intrinsics_normalized", True)):
        intrinsics = intrinsics.clone()
        for view, (height, width) in enumerate(source_sizes):
            intrinsics[view, 0] /= float(width)
            intrinsics[view, 1] /= float(height)

    return (
        intrinsics,
        extrinsics,
        float(data.get("near", 0.1)),
        float(data.get("far", 10.0)),
    )


def save_npz(prediction, path: str | Path) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "means": prediction.gaussians.means.detach().cpu().numpy(),
        "covariances": prediction.gaussians.covariances.detach().cpu().numpy(),
        "harmonics": prediction.gaussians.harmonics.detach().cpu().numpy(),
        "opacities": prediction.gaussians.opacities.detach().cpu().numpy(),
        "predicted_extrinsics": prediction.extrinsics.detach().cpu().numpy(),
        "pose_delta": prediction.pose_delta.detach().cpu().numpy(),
        "depth": prediction.depth.detach().cpu().numpy(),
    }
    np.savez_compressed(output, **arrays)
    return output.resolve()


def save_preview_ply(
    prediction, path: str | Path, max_points: int | None = None
) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    means = prediction.gaussians.means[0].detach().float().cpu()
    harmonics = prediction.gaussians.harmonics[0].detach().float().cpu()
    colors = (0.5 + 0.28209479177387814 * harmonics[:, :, 0]).clamp(0, 1)
    colors = (colors * 255).round().to(torch.uint8)

    if max_points is not None and max_points > 0 and means.shape[0] > max_points:
        indices = torch.linspace(0, means.shape[0] - 1, max_points).round().long()
        means = means[indices]
        colors = colors[indices]

    with output.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {means.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(means.tolist(), colors.tolist()):
            handle.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )
    return output.resolve()
