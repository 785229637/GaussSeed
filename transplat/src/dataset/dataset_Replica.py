import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Tuple
import torch.nn.functional as F
import random
import trimesh
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetReplicaCfg(DatasetCfgCommon):
    name: Literal["replica"]  # 数据集名称改为replica
    roots: list[Path]         # 保持与TUM一致的roots输入（多路径支持）
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = 1
    far: float = 100
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    sequence: Optional[str] = 'office0'  # 场景名称（如room0）
    sequences: Optional[list[str]] = None
    depth_scale: float = 6553.5
    context_stride: int = 5
    context_stride_jitter: int = 1
    target_radius: int = 10
    target_stride: int = 5


@dataclass
class ReplicaSceneData:
    name: str
    data: List[Tuple[Path, Path, np.ndarray]]
    fx: float
    fy: float
    cx: float
    cy: float
    original_width: int
    original_height: int


class DatasetReplica(IterableDataset):
    cfg: DatasetReplicaCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    near: float = 1.0       # 室内场景默认近平面
    far: float = 100.0      # 室内场景默认远平面
    data: List[Tuple[Path, Path, np.ndarray]]  # (图像路径, 深度路径, c2w位姿)
    scenes: list[ReplicaSceneData]
    
    # Replica特有内部变量（从camera_info.json解析）
    fx: float
    fy: float
    cx: float
    cy: float
    replica_original_size: Tuple[int, int]  # (原始高度, 原始宽度)


    def __init__(
        self,
        cfg: DatasetReplicaCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        

        # 处理root路径（从cfg.roots取第一个，保持输入兼容性）
        if not self.cfg.roots:
            raise ValueError("DatasetReplica requires non-empty 'roots' in config")
        self.cfg.root = self.cfg.roots[0]  # 统一为单root路径处理

        # 内部定义Replica默认参数（无外部输入新增）
        self.cfg.sequence = self.cfg.sequence if hasattr(self.cfg, "sequence") else "room0"  # 默认场景
        self.cfg.image_size = self.cfg.image_size if hasattr(self.cfg, "image_size") else (480, 640)  # 默认目标尺寸
        self.cfg.max_time_diff = 0.02  # 时间差阈值（Replica虽不用但保持接口一致）
        self.scene_idx_offset = 0      # 场景起始偏移
        self.scene_suffix = "_0"         # 场景后缀

        # 室内场景远近平面（允许配置覆盖）
        if self.cfg.near != -1:
            self.near = self.cfg.near
        if self.cfg.far != -1:
            self.far = self.cfg.far

        sequence_names = self.cfg.sequences or [self.cfg.sequence or "room0"]
        self.sequences = [str(sequence) for sequence in sequence_names]
        self.scenes = [self._load_scene(sequence) for sequence in self.sequences]
        if not self.scenes:
            raise ValueError("DatasetReplica requires at least one sequence")

        # Backward-compatible attributes for callers that inspect a single sequence.
        first_scene = self.scenes[0]
        self.cfg.sequence = first_scene.name
        self.data = first_scene.data
        self.fx = first_scene.fx
        self.fy = first_scene.fy
        self.cx = first_scene.cx
        self.cy = first_scene.cy
        self.replica_original_width = first_scene.original_width
        self.replica_original_height = first_scene.original_height
        self.replica_original_size = (first_scene.original_height, first_scene.original_width)

    def _set_scene_paths(self, sequence: str) -> None:
        # 调整路径以适应results文件夹结构
        self.sequence_path = self.cfg.root / sequence
        self.results_path = self.sequence_path / "results"  # RGB和深度图都在results文件夹中
        self.pose_file = self.sequence_path / "traj.txt"
        self.camera_info_file = self.sequence_path / "cam_params.json"
        if not self.camera_info_file.exists():
            self.camera_info_file = self.cfg.root / "cam_params.json"

    def _load_scene(self, sequence: str) -> ReplicaSceneData:
        self.cfg.sequence = sequence
        self._set_scene_paths(sequence)
        self._validate_paths()
        self._load_camera_intrinsics()
        data = self.load_and_align_data()
        if not data:
            raise ValueError(f"No aligned data found for sequence: {sequence}")
        return ReplicaSceneData(
            name=sequence,
            data=data,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            original_width=self.replica_original_width,
            original_height=self.replica_original_height,
        )


    def _validate_paths(self) -> None:
        """验证Replica必要文件路径存在"""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_path}")
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file (traj.txt) not found: {self.pose_file}")
        if not self.camera_info_file.exists():
            raise FileNotFoundError(f"Camera info file not found: {self.camera_info_file}")


    def _load_camera_intrinsics(self) -> None:
        """从camera_info.json加载内参"""
        with open(self.camera_info_file, "r") as f:
            camera_info = json.load(f)
        
        # 处理两种常见内参格式：直接字段或K矩阵
        if "K" in camera_info:
            K = np.array(camera_info["K"], dtype=np.float32).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
        else:
            camera = camera_info.get("camera", camera_info)
            self.fx = float(camera["fx"])
            self.fy = float(camera["fy"])
            self.cx = float(camera["cx"])
            self.cy = float(camera["cy"])
        
        # 记录原始图像尺寸
        camera = camera_info.get("camera", camera_info)
        self.replica_original_width = int(camera.get("w", camera_info.get("width", 640)))
        self.replica_original_height = int(camera.get("h", camera_info.get("height", 480)))
        self.replica_original_size = (self.replica_original_height, self.replica_original_width)


    def load_and_align_data(self) -> List[Tuple[Path, Path, np.ndarray]]:
        """加载并对齐数据（从results文件夹中）"""
        # 1. 加载并排序RGB图像（从results文件夹）
        rgb_extensions = (".jpg", ".png")
        # 筛选出RGB图像（假设名称中包含"rgb"标识）
        rgb_files = [
            f for f in self.results_path.iterdir() 
            if f.suffix.lower() in rgb_extensions and "jpg" in f.name.lower()
        ]
        
        if not rgb_files:
            raise FileNotFoundError(f"No RGB images found in results directory: {self.results_path}")
        
        # 按文件名序号排序
        def _get_image_index(file_path: Path) -> int:
            # 提取文件名中的数字部分（假设格式如"rgb_000000.png"）
            import re
            match = re.search(r'\d+', file_path.stem)
            return int(match.group()) if match else 0
        
        sorted_rgb_paths = sorted(rgb_files, key=_get_image_index)
        num_rgb = len(sorted_rgb_paths)
        print(f"Found {num_rgb} RGB images in results directory: {self.results_path}")

        depth_files = [
            f for f in self.results_path.iterdir()
            if f.suffix.lower() == ".png" and "depth" in f.name.lower()
        ]
        if not depth_files:
            raise FileNotFoundError(f"No depth images found in results directory: {self.results_path}")
        sorted_depth_paths = sorted(depth_files, key=_get_image_index)
        num_depth = len(sorted_depth_paths)
        print(f"Found {num_depth} depth images in results directory: {self.results_path}")

        # 2. 加载并排序位姿
        pose_data = []
        with open(self.pose_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    print(f"Skipping invalid pose line: {line}")
                    continue
                # 提取平移和四元数
                pose_matrix = np.array(parts, dtype=np.float32).reshape(4, 4)
                pose_data.append(pose_matrix)
        num_poses = len(pose_data)
        print(f"Found {num_poses} valid poses in traj.txt")

        # 3. 验证图像与位姿数量匹配
        if num_rgb != num_poses:
            raise ValueError(
                f"RGB count ({num_rgb}) != Pose count ({num_poses}) "
                f"for sequence: {self.cfg.sequence}"
            )
        if num_depth != num_rgb:
            raise ValueError(
                f"Depth count ({num_depth}) != RGB count ({num_rgb}) "
                f"for sequence: {self.cfg.sequence}"
            )

        # 4. 生成对齐数据
        aligned_data = []
        for img_path, depth_path, c2w in zip(sorted_rgb_paths, sorted_depth_paths, pose_data):
            T = c2w

            # # （可选）坐标系适配：若需与TUM右手系对齐，可翻转Y轴
            # T[:3, 1] *= -1

            aligned_data.append((img_path, depth_path, T.astype(np.float32)))

        print(f"Aligned {len(aligned_data)} images-poses for sequence: {self.cfg.sequence}")
        return aligned_data


    def c2w_to_w2c(self, extrinsics: Tensor) -> Tensor:
        """c2w转w2c"""
        assert extrinsics.shape[-2:] == (4, 4), "Extrinsics must be 4x4"
        B = extrinsics.shape[0]
        w2c = torch.zeros_like(extrinsics)

        R_w2c = extrinsics[:, :3, :3].transpose(1, 2)
        w2c[:, :3, :3] = R_w2c

        t_w2c = -torch.bmm(R_w2c, extrinsics[:, :3, 3:4])
        w2c[:, :3, 3:4] = t_w2c

        w2c[:, 3, 3] = 1.0
        return w2c


    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,   2*x*z + 2*y*w],
            [2*x*y + 2*z*w,   1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)


    def load_images(self, indices: list[int], image_paths: list[Path]) -> Float[Tensor, "batch 3 h w"]:
        """加载上下文图像并进行缩放"""
        images = []
        for idx in indices:
            img_path = image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            # 按配置尺寸缩放context图像
            img = img.resize((256,256), Image.BILINEAR)  # (w, h)输入
            images.append(self.to_tensor(img))
        return torch.stack(images)


    def load_images_target(self, indices: list[int], image_paths: list[Path]) -> Float[Tensor, "batch 3 h w"]:
        """加载目标图像，不进行缩放"""
        images = []
        for idx in indices:
            img_path = image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            # 不缩放target图像，保持原始尺寸
            images.append(self.to_tensor(img))
        return torch.stack(images)

    def load_depths(self, indices: list[int], depth_paths: list[Path]) -> Float[Tensor, "batch 1 h w"]:
        depths = []
        for idx in indices:
            depth = Image.open(depth_paths[idx])
            depth = depth.resize((256, 256), Image.NEAREST)
            depth_np = np.array(depth, dtype=np.float32) / float(self.cfg.depth_scale)
            depths.append(torch.from_numpy(depth_np).unsqueeze(0))
        return torch.stack(depths)


    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, "view"]:
        """获取远近平面"""
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)


    def _iter_scene(self, scene_data: ReplicaSceneData, run_idx: int):
        """迭代单个Replica场景。"""
        # 提取对齐后的数据
        image_paths = [item[0] for item in scene_data.data]
        depth_paths = [item[1] for item in scene_data.data]
        poses = [item[2] for item in scene_data.data]
        scene = f"{scene_data.name}_{run_idx}"

        # 位姿转张量
        extrinsics = torch.tensor(np.array(poses), dtype=torch.float32)

        # 归一化内参。Replica默认是1200x680、fx=fy=600，这里保留同等含义但允许cam_params覆盖。
        norm_intrinsics = torch.tensor([
            [scene_data.fx / scene_data.original_width, 0.0, scene_data.cx / scene_data.original_width],
            [0.0, scene_data.fy / scene_data.original_height, scene_data.cy / scene_data.original_height],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)

        intrinsics_256 = norm_intrinsics.repeat(len(image_paths), 1, 1)
        intrinsics = norm_intrinsics.repeat(len(image_paths), 1, 1)

        base_context_stride = max(1, int(self.cfg.context_stride))
        context_stride_jitter = max(0, int(self.cfg.context_stride_jitter))
        max_context_offset = base_context_stride + context_stride_jitter
        target_radius = max(0, int(self.cfg.target_radius))
        target_stride = max(1, int(self.cfg.target_stride))

        max_idx = extrinsics.size(0) - max_context_offset
        if max_idx <= 0:
            return
        if self.stage == "train":
            scene_indices = torch.randperm(max_idx).tolist()
        else:
            scene_indices = range(run_idx % 10, max_idx, 1)
        for scene_idx in scene_indices:
            min_offset = max(1, base_context_stride - context_stride_jitter)
            max_offset = min(max_context_offset, extrinsics.size(0) - 1 - scene_idx)
            if self.stage == "train" and max_offset > min_offset:
                context_offset = torch.randint(min_offset, max_offset + 1, size=()).item()
            else:
                context_offset = min(base_context_stride, max_offset)
            finally_n = scene_idx + context_offset

            context_indices = torch.tensor([scene_idx, finally_n])

            extended_start = max(0, scene_idx - target_radius)
            extended_end = min(extrinsics.size(0) - 1, finally_n + target_radius)
            target_indices = torch.arange(extended_start, extended_end + 1, target_stride)
            target_indices = torch.unique(torch.cat((target_indices, context_indices))).sort().values

            # 加载图像 - context缩放，target不缩放
            context_images = self.load_images(context_indices.tolist(), image_paths)
            context_depths = self.load_depths(context_indices.tolist(), depth_paths)
            target_images = self.load_images_target(target_indices.tolist(), image_paths)

            # 基线比例调整
            scale = 1.0
            # if context_indices.shape[0] == 2 and self.cfg.make_baseline_1:
            #     a, b = extrinsics[context_indices][:, :3, 3]
            #     scale = (a - b).norm()
            #     if scale < self.cfg.baseline_epsilon:
            #         print(f"Skipped {scene} (small baseline: {scale:.6f})")
            #         continue
            #     extrinsics[:, :3, 3] /= scale

            # 构建输出字典
            nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics_256[context_indices],
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
                "scene": scene,
            }

            # 数据增强
            # if self.stage == "train" and self.cfg.augment:
            #     example = apply_augmentation_shim(example)

            yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def __iter__(self):
        """迭代器"""
        runs = range(
            self.cfg.test_times_per_scene
            if self.stage == "test"
            else 1
        )
        for run_idx in runs:
            if self.stage == "train" and len(self.scenes) > 1:
                scene_iters = [self._iter_scene(scene, run_idx) for scene in self.scenes]
                while scene_iters:
                    for iter_idx in torch.randperm(len(scene_iters)).tolist():
                        if iter_idx >= len(scene_iters):
                            continue
                        try:
                            yield next(scene_iters[iter_idx])
                        except StopIteration:
                            scene_iters.pop(iter_idx)
                            break
            else:
                for scene_data in self.scenes:
                    yield from self._iter_scene(scene_data, run_idx)


    def __len__(self) -> int:
        """数据集长度"""
        max_context_offset = max(1, int(self.cfg.context_stride)) + max(0, int(self.cfg.context_stride_jitter))
        return sum(max(0, len(scene.data) - max_context_offset) for scene in self.scenes)
