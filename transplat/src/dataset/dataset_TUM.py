from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as tf
import trimesh
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


TUM_INTRINSICS = {
    "rgbd_dataset_freiburg1_desk": (517.306408, 516.469215, 318.643040, 255.313989, 640, 480),
    "rgbd_dataset_freiburg1_desk2": (517.306408, 516.469215, 318.643040, 255.313989, 640, 480),
    "rgbd_dataset_freiburg1_room": (517.306408, 516.469215, 318.643040, 255.313989, 640, 480),
    "rgbd_dataset_freiburg1_xyz": (517.306408, 516.469215, 318.643040, 255.313989, 640, 480),
    "rgbd_dataset_freiburg2_xyz": (520.908620, 521.007327, 325.141442, 249.701764, 640, 480),
    "rgbd_dataset_freiburg3_long_office_household": (535.4, 539.2, 320.1, 247.6, 640, 480),
}


@dataclass
class DatasetTUMRGBDCfg(DatasetCfgCommon):
    name: Literal["tum_rgbd"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = 0.1
    far: float = 10.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    sequence: Optional[str] = "rgbd_dataset_freiburg1_desk"
    sequences: Optional[list[str]] = None
    max_time_diff: float = 0.02
    depth_scale: float = 5000.0
    context_stride: int = 5
    context_stride_jitter: int = 1
    target_radius: int = 10
    target_stride: int = 5


@dataclass
class TUMSceneData:
    name: str
    data: List[Tuple[Path, Path, np.ndarray]]
    intrinsics: Tensor


class DatasetTUMRGBD(IterableDataset):
    def __init__(
        self,
        cfg: DatasetTUMRGBDCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if not cfg.roots:
            raise ValueError("DatasetTUMRGBD requires non-empty 'roots' in config")
        self.root = Path(cfg.roots[0])
        self.near = float(cfg.near)
        self.far = float(cfg.far)

        sequence_names = cfg.sequences or [cfg.sequence or "rgbd_dataset_freiburg1_desk"]
        invalid_sequences = [name for name in sequence_names if name not in TUM_INTRINSICS]
        if invalid_sequences:
            raise ValueError(
                f"Unsupported TUM sequences: {invalid_sequences}; allowed: {sorted(TUM_INTRINSICS)}"
            )

        self.sequences = [str(name) for name in sequence_names]
        self.scenes = [self._load_scene(name) for name in self.sequences]
        if not self.scenes:
            raise ValueError("DatasetTUMRGBD requires at least one valid sequence")

        first_scene = self.scenes[0]
        self.cfg.sequence = first_scene.name
        self.data = first_scene.data
        self.sequence_intrinsics = {scene.name: scene.intrinsics for scene in self.scenes}

    @staticmethod
    def _load_timestamp_paths(file_path: Path, sequence_path: Path) -> list[tuple[float, Path]]:
        entries = []
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                entries.append((float(parts[0]), sequence_path / parts[1]))
        return entries

    @staticmethod
    def _load_poses(file_path: Path) -> list[tuple[float, np.ndarray]]:
        poses = []
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                timestamp = float(parts[0])
                translation = np.asarray(parts[1:4], dtype=np.float64)
                quaternion_xyzw = np.asarray(parts[4:8], dtype=np.float64)
                c2w = trimesh.transformations.quaternion_matrix(
                    np.roll(quaternion_xyzw, 1)
                ).astype(np.float32)
                c2w[:3, 3] = translation
                poses.append((timestamp, c2w))
        return poses

    @staticmethod
    def _normalized_intrinsics(sequence: str) -> Tensor:
        fx, fy, cx, cy, width, height = TUM_INTRINSICS[sequence]
        return torch.tensor(
            [
                [fx / width, 0.0, cx / width],
                [0.0, fy / height, cy / height],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def _load_scene(self, sequence: str) -> TUMSceneData:
        sequence_path = self.root / sequence
        rgb_file = sequence_path / "rgb.txt"
        depth_file = sequence_path / "depth.txt"
        pose_file = sequence_path / "groundtruth.txt"
        for path in (sequence_path, rgb_file, depth_file, pose_file):
            if not path.exists():
                raise FileNotFoundError(path)

        rgb_entries = self._load_timestamp_paths(rgb_file, sequence_path)
        depth_entries = self._load_timestamp_paths(depth_file, sequence_path)
        pose_entries = self._load_poses(pose_file)
        if not rgb_entries or not depth_entries or not pose_entries:
            raise ValueError(f"Missing RGB, depth, or pose data for TUM sequence: {sequence}")

        depth_times = np.asarray([timestamp for timestamp, _ in depth_entries])
        pose_times = np.asarray([timestamp for timestamp, _ in pose_entries])
        aligned_data: List[Tuple[Path, Path, np.ndarray]] = []
        for rgb_timestamp, rgb_path in rgb_entries:
            depth_index = int(np.argmin(np.abs(depth_times - rgb_timestamp)))
            pose_index = int(np.argmin(np.abs(pose_times - rgb_timestamp)))
            if abs(depth_times[depth_index] - rgb_timestamp) >= self.cfg.max_time_diff:
                continue
            if abs(pose_times[pose_index] - rgb_timestamp) >= self.cfg.max_time_diff:
                continue

            depth_path = depth_entries[depth_index][1]
            if not rgb_path.is_file() or not depth_path.is_file():
                continue
            aligned_data.append((rgb_path, depth_path, pose_entries[pose_index][1]))

        if not aligned_data:
            raise ValueError(f"No aligned RGB-D-pose samples found for TUM sequence: {sequence}")
        print(f"Aligned {len(aligned_data)} RGB-D-pose samples for sequence: {sequence}")
        return TUMSceneData(
            name=sequence,
            data=aligned_data,
            intrinsics=self._normalized_intrinsics(sequence),
        )

    def load_images(
        self,
        indices: list[int],
        image_paths: list[Path],
    ) -> Float[Tensor, "batch 3 h w"]:
        images = []
        height, width = (int(value) for value in self.cfg.image_shape)
        for index in indices:
            image = Image.open(image_paths[index]).convert("RGB")
            image = image.resize((width, height), Image.BILINEAR)
            images.append(self.to_tensor(image))
        return torch.stack(images)

    def load_images_target(
        self,
        indices: list[int],
        image_paths: list[Path],
    ) -> Float[Tensor, "batch 3 h w"]:
        return torch.stack(
            [self.to_tensor(Image.open(image_paths[index]).convert("RGB")) for index in indices]
        )

    def load_depths(
        self,
        indices: list[int],
        depth_paths: list[Path],
    ) -> Float[Tensor, "batch 1 h w"]:
        depths = []
        height, width = (int(value) for value in self.cfg.image_shape)
        for index in indices:
            depth = Image.open(depth_paths[index])
            depth = depth.resize((width, height), Image.NEAREST)
            depth_array = np.asarray(depth, dtype=np.float32) / float(self.cfg.depth_scale)
            depths.append(torch.from_numpy(depth_array).unsqueeze(0))
        return torch.stack(depths)

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, "view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def _iter_scene(self, scene_data: TUMSceneData, run_idx: int):
        image_paths = [item[0] for item in scene_data.data]
        depth_paths = [item[1] for item in scene_data.data]
        extrinsics = torch.tensor(
            np.asarray([item[2] for item in scene_data.data]),
            dtype=torch.float32,
        )
        intrinsics = scene_data.intrinsics.repeat(len(scene_data.data), 1, 1)

        base_context_stride = max(1, int(self.cfg.context_stride))
        context_stride_jitter = max(0, int(self.cfg.context_stride_jitter))
        max_context_offset = base_context_stride + context_stride_jitter
        target_radius = max(0, int(self.cfg.target_radius))
        target_stride = max(1, int(self.cfg.target_stride))
        max_index = len(scene_data.data) - max_context_offset
        if max_index <= 0:
            return

        if self.stage == "train":
            scene_indices = torch.randperm(max_index).tolist()
        else:
            scene_indices = range(run_idx % max_context_offset, max_index)

        for scene_index in scene_indices:
            min_offset = max(1, base_context_stride - context_stride_jitter)
            max_offset = min(max_context_offset, len(scene_data.data) - 1 - scene_index)
            if self.stage == "train" and max_offset > min_offset:
                context_offset = torch.randint(min_offset, max_offset + 1, size=()).item()
            else:
                context_offset = min(base_context_stride, max_offset)

            final_index = scene_index + context_offset
            context_indices = torch.tensor([scene_index, final_index])
            target_start = max(0, scene_index - target_radius)
            target_end = min(len(scene_data.data) - 1, final_index + target_radius)
            target_indices = torch.arange(target_start, target_end + 1, target_stride)
            target_indices = torch.unique(torch.cat((target_indices, context_indices))).sort().values

            context_images = self.load_images(context_indices.tolist(), image_paths)
            context_depths = self.load_depths(context_indices.tolist(), depth_paths)
            target_images = self.load_images_target(target_indices.tolist(), image_paths)
            nf_scale = 1.0
            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "depth": context_depths,
                    "near": self.get_bound("near", len(context_indices)) / nf_scale,
                    "far": self.get_bound("far", len(context_indices)) / nf_scale,
                    "index": context_indices,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / nf_scale,
                    "far": self.get_bound("far", len(target_indices)) / nf_scale,
                    "index": target_indices,
                },
                "scene": f"{scene_data.name}_{run_idx}",
            }
            yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def __iter__(self):
        runs = range(self.cfg.test_times_per_scene if self.stage == "test" else 1)
        for run_idx in runs:
            if self.stage == "train" and len(self.scenes) > 1:
                scene_iters = [self._iter_scene(scene, run_idx) for scene in self.scenes]
                while scene_iters:
                    for iterator_index in torch.randperm(len(scene_iters)).tolist():
                        if iterator_index >= len(scene_iters):
                            continue
                        try:
                            yield next(scene_iters[iterator_index])
                        except StopIteration:
                            scene_iters.pop(iterator_index)
                            break
            else:
                for scene_data in self.scenes:
                    yield from self._iter_scene(scene_data, run_idx)

    def __len__(self) -> int:
        max_context_offset = max(1, int(self.cfg.context_stride)) + max(
            0, int(self.cfg.context_stride_jitter)
        )
        return sum(max(0, len(scene.data) - max_context_offset) for scene in self.scenes)
