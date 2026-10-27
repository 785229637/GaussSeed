"""Stable two-view proposal API for the GaussSeed module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from dacite import Config, from_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .checkpoint import load_encoder_checkpoint
from .pipeline import GaussSeedPipeline, PipelineConfig, PipelineResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GaussSeedPrediction:
    gaussians: object
    extrinsics: torch.Tensor
    pose_delta: torch.Tensor
    depth: torch.Tensor


class GaussSeed:
    """Predict canonical Gaussians and prepare paper-aligned backend tensors."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        experiment: str = "re10k",
        config_dir: str | Path | None = None,
        device: str | torch.device = "cuda",
        overrides: Iterable[str] = (),
        strict: bool = True,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.experiment = experiment
        self.config_dir = Path(
            config_dir or PROJECT_ROOT / "transplat" / "config"
        ).expanduser().resolve()
        self.device = torch.device(device)
        self.overrides = tuple(overrides)
        self.strict = strict
        self.pipeline = GaussSeedPipeline(pipeline_config)
        self.encoder = None
        self.config = None

    def _compose_config(self):
        overrides = [
            f"+experiment={self.experiment}",
            "mode=test",
            "dataset.view_sampler.num_context_views=2",
            f"checkpointing.load='{self.checkpoint}'",
            "model.encoder.dinov3_weights_path=null",
            "model.encoder.depth_head_weights_path=null",
            "model.encoder.depth_anything_weights_path=null",
            "model.encoder.unimatch_weights_path=null",
            *self.overrides,
        ]
        with initialize_config_dir(
            version_base=None, config_dir=str(self.config_dir)
        ):
            return compose(config_name="main", overrides=overrides)

    def load(self):
        if self.encoder is not None:
            return self.encoder
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {self.checkpoint}")
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Hydra config directory not found: {self.config_dir}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        try:
            from gaussseed_encoder.encoder_dinov3 import EncoderTrans, EncoderTransCfg
            from transplat.src.global_cfg import set_cfg
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("dinov3"):
                raise RuntimeError(
                    "DINOv3 is not installed. Follow the third_party/dinov3 "
                    "instructions in README.md."
                ) from exc
            raise

        cfg = self._compose_config()
        set_cfg(cfg)
        encoder_data = OmegaConf.to_container(
            cfg.model.encoder, resolve=True, throw_on_missing=True
        )
        encoder_cfg = from_dict(
            EncoderTransCfg,
            encoder_data,
            config=Config(check_types=True, strict=False),
        )
        encoder = EncoderTrans(encoder_cfg)
        load_encoder_checkpoint(encoder, self.checkpoint, strict=self.strict)
        self.encoder = encoder.to(self.device).eval()
        self.config = cfg
        return self.encoder

    @torch.inference_mode()
    def predict(self, context: dict) -> GaussSeedPrediction:
        encoder = self.load()
        context = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in context.items()
        }
        gaussians, extrinsics, pose_delta, depth = encoder(
            context=context,
            global_step=0,
            deterministic=True,
            scene_names=["standalone_inference"],
            pose_ture=False,
        )
        return GaussSeedPrediction(gaussians, extrinsics, pose_delta, depth)

    def prepare_backend(
        self,
        prediction: GaussSeedPrediction,
        *,
        target_camera_to_world: torch.Tensor,
        target_intrinsics: torch.Tensor,
        map_rendered_depth: torch.Tensor | None = None,
        proposal_to_global_rotation: torch.Tensor | None = None,
        proposal_to_global_translation: torch.Tensor | None = None,
        proposal_view: int = -1,
        initializing: bool = False,
        intrinsics_normalized: bool = True,
    ) -> PipelineResult:
        """Run alignment, consistency/novelty filtering, and tensor adaptation.

        The host supplies only its current camera state and, during expansion,
        depth rendered internally from the existing Gaussian map.  No measured
        or ground-truth depth is accepted by this interface.
        """

        depth = prediction.depth
        if depth.ndim != 5:
            raise ValueError("prediction.depth must have shape [B, V, S, H, W]")
        proposal_depth = depth[0, proposal_view, 0]
        return self.pipeline.process(
            prediction.gaussians,
            proposal_depth=proposal_depth,
            target_camera_to_world=target_camera_to_world,
            target_intrinsics=target_intrinsics,
            map_rendered_depth=map_rendered_depth,
            proposal_to_global_rotation=proposal_to_global_rotation,
            proposal_to_global_translation=proposal_to_global_translation,
            initializing=initializing,
            intrinsics_normalized=intrinsics_normalized,
        )

    @staticmethod
    def make_context(
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        *,
        near: float = 0.1,
        far: float = 10.0,
    ) -> dict:
        if images.ndim != 4 or images.shape[0] != 2:
            raise ValueError("images must have shape [2, 3, H, W]")
        if intrinsics.shape != (2, 3, 3):
            raise ValueError("intrinsics must have shape [2, 3, 3]")
        if extrinsics.shape != (2, 4, 4):
            raise ValueError("extrinsics must have shape [2, 4, 4] in camera-to-world form")
        return {
            "image": images.unsqueeze(0),
            "intrinsics": intrinsics.unsqueeze(0),
            "extrinsics": extrinsics.unsqueeze(0),
            "near": torch.full((1, 2), float(near), dtype=torch.float32),
            "far": torch.full((1, 2), float(far), dtype=torch.float32),
            "index": torch.tensor([[0, 1]], dtype=torch.int64),
        }
