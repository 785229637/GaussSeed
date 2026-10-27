from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        out_ex: Tensor, 
        delta_T_encoder: Tensor,
        depth: Tensor
    ) -> Float[Tensor, ""]:
        pred_depth = depth
        gt_depth = batch["context"]["depth"].to(pred_depth.device, pred_depth.dtype)
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(
                pred_depth.flatten(0, 1),
                size=gt_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(*gt_depth.shape)

        valid_mask = torch.isfinite(gt_depth) & (gt_depth > 1e-6)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, dtype=pred_depth.dtype, device=pred_depth.device)

        near = batch["context"]["near"][..., None, None, None].to(pred_depth.device, pred_depth.dtype)
        far = batch["context"]["far"][..., None, None, None].to(pred_depth.device, pred_depth.dtype)
        pred_depth = pred_depth.clamp(min=near, max=far)
        gt_depth = gt_depth.clamp(min=near, max=far)

        # Log-depth loss penalizes metric scale collapse while reducing overweighting of far pixels.
        depth_error = F.smooth_l1_loss(
            torch.log(pred_depth[valid_mask].clamp_min(1e-6)),
            torch.log(gt_depth[valid_mask].clamp_min(1e-6)),
            beta=0.1,
        )
        return self.cfg.weight * depth_error
