"""Checkpoint compatibility helpers shared by inference and SLAM adapters."""

from pathlib import Path
from typing import Mapping

import torch


_KEY_REPLACEMENTS = {
    "backbone.cam_param_reduce_conv": "backbone.cam_param_encoder.reduce_conv",
    "backbone.cam_param_context_conv": "backbone.cam_param_encoder.context_conv",
    "backbone.cam_param_bn": "backbone.cam_param_encoder.bn",
    "backbone.cam_param_layer": "backbone.cam_param_encoder.layer",
    "backbone.cam_param_context_mlp": "backbone.cam_param_encoder.context_mlp",
    "backbone.cam_param_context_se": "backbone.cam_param_encoder.context_se",
    "backbone_t.cam_param_reduce_conv": "backbone_t.cam_param_encoder.reduce_conv",
    "backbone_t.cam_param_context_conv": "backbone_t.cam_param_encoder.context_conv",
    "backbone_t.cam_param_bn": "backbone_t.cam_param_encoder.bn",
    "backbone_t.cam_param_layer": "backbone_t.cam_param_encoder.layer",
    "backbone_t.cam_param_context_mlp": "backbone_t.cam_param_encoder.context_mlp",
    "backbone_t.cam_param_context_se": "backbone_t.cam_param_encoder.context_se",
    "depth_predictor.cam_param_reduce_conv": "depth_predictor.cam_param_encoder.reduce_conv",
    "depth_predictor.cam_param_context_conv": "depth_predictor.cam_param_encoder.context_conv",
    "depth_predictor.cam_param_bn": "depth_predictor.cam_param_encoder.bn",
    "depth_predictor.cam_param_layer": "depth_predictor.cam_param_encoder.layer",
    "depth_predictor.cam_param_context_mlp": "depth_predictor.cam_param_encoder.context_mlp",
    "depth_predictor.cam_param_context_se": "depth_predictor.cam_param_encoder.context_se",
    "depth_predictor.coarse_transformer.layers": "depth_predictor.coarse_transformer.encoder.layers",
    "depth_predictor.fine_transformer.layers": "depth_predictor.fine_transformer.encoder.layers",
}


def load_checkpoint(path: str | Path) -> Mapping:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_encoder_state_dict(payload: Mapping) -> dict[str, torch.Tensor]:
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a state dictionary")

    has_encoder_prefix = any(str(key).startswith("encoder.") for key in state_dict)
    encoder_state = {}
    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        if has_encoder_prefix:
            if not key.startswith("encoder."):
                continue
            key = key.removeprefix("encoder.")
        for old, new in _KEY_REPLACEMENTS.items():
            key = key.replace(old, new)
        if key.startswith("backbone.backbone."):
            key = key.replace(
                "backbone.backbone.", "backbone.encoder.backbone.", 1
            )
        encoder_state[key] = value

    if not encoder_state:
        raise ValueError("No encoder weights were found in the checkpoint")
    return encoder_state


def load_encoder_checkpoint(
    encoder: torch.nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = True,
):
    state_dict = extract_encoder_state_dict(load_checkpoint(checkpoint))
    return encoder.load_state_dict(state_dict, strict=strict)
