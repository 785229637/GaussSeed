from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F
from collections import OrderedDict
# from gaussseed_encoder.PoseEstimation import PoseEstimationHead

from transplat.src.dataset.shims.bounds_shim import apply_bounds_shim
from transplat.src.dataset.shims.patch_shim import apply_patch_shim
from transplat.src.dataset.types import BatchedExample, DataShim
from transplat.src.geometry.projection import sample_image_grid
from transplat.src.model.types import Gaussians
from gaussseed_encoder.backbone.backbone_multiview import BackboneMultiview


from gaussseed_encoder.backbone.backbone_dinov3 import Dinov3Backbone
from gaussseed_encoder.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from gaussseed_encoder.matching.depth_predictor_trans import DepthPredictorTrans
from gaussseed_encoder.visualization.encoder_visualizer_costvolume_cfg import (
    EncoderVisualizerCostVolumeCfg,
)

from transplat.src.global_cfg import get_cfg

from transplat.src.depth_anything_v2.dpt import DepthAnythingV2


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return str(resolved.resolve())


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderTransCfg:
    name: Literal["trans"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_cost_volume_refine: bool
    dinov3_arch: str
    dinov3_weights_path: str | None
    depth_head_weights_path: str | None
    depth_anything_weights_path: str | None


class EncoderTrans(Encoder[EncoderTransCfg]):
    backbone_t: BackboneMultiview
    # depth_predictor:  DepthPredictorTrans
    backbone: Dinov3Backbone
    # backbone: depth_pre
    # position_predictor:  PosPredictor
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderTransCfg) -> None:
        super().__init__(cfg)

        # dino_v3骨干：返回dino图像特征和深度特征
        self.backbone = Dinov3Backbone(
            dinov3_arch=cfg.dinov3_arch,
            dinov3_ckpt=_resolve_repo_path(cfg.dinov3_weights_path),
            dpt_head_ckpt=_resolve_repo_path(cfg.depth_head_weights_path),
        )
        for param in self.backbone.parameters():
            param.requires_grad = True
        # self.backbone  = Dinov3Backbone(dinov3_arch="vit_7b")

        self.backbone_t = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
        )

        # 添加1x1卷积层，用于转换dino特征
        self.channel_conv = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)

        self.upsample_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=384,
                out_channels=128,
                kernel_size=4,      # 使用4x4卷积核
                stride=4,           # 步长为4，实现4倍上采样
                padding=0
            )
        )

        # # 定义转置卷积层，用于转化depth特征
        # self.transpose_conv = nn.ConvTranspose2d(
        #     in_channels=256,
        #     out_channels=64,
        #     kernel_size=19,
        #     stride=1,
        #     padding=1
        # ).to("cuda")

        # 定义普通卷积层（适合下采样），输入[2,256,256,256] → 输出[2,64,36,36]
        # 定义转置卷积层，用于转化depth特征
        # 使用卷积减少通道数，然后自适应池化调整空间尺寸
        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=80,
                out_channels=64,
                kernel_size=1,      # 使用1x1卷积只改变通道数
                stride=1,
                padding=0
            ),
            nn.AdaptiveAvgPool2d((144, 144))  # 自适应池化到目标尺寸
        )



        ckpt_path = _resolve_repo_path(cfg.unimatch_weights_path)
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone_t.state_dict()
                    }
                )
                self.backbone_t.load_state_dict(updated_state_dict, strict=False)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb' # or 'vits', 'vitb', 'vitl'
        DA_size = model_configs[encoder]['features'] // 2

        self.da_model = DepthAnythingV2(**model_configs[encoder])
        depth_anything_path = _resolve_repo_path(cfg.depth_anything_weights_path)
        if depth_anything_path is not None:
            self.da_model.load_state_dict(
                torch.load(depth_anything_path, map_location="cpu")
            )
        self.da_model = self.da_model.eval()
        
        for param in self.da_model.parameters():
            param.requires_grad = False

        self.depth_predictor = DepthPredictorTrans(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            # wo_depth_refine=cfg.wo_depth_refine,
            # wo_cost_volume=cfg.wo_cost_volume,
            # wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            DA_size=DA_size,
        )
        # 新增位姿估计头
        # self.pose_head = PoseEstimationHead(cfg.d_feature*64*64).to("cuda:0")
        # self.pose_head = PoseEstimationHead(int(cfg.d_feature*144*144/2)).to("cuda:0")


    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        pose_ture = False
    ) -> Gaussians:
        # Map-aware alignment and filtering are intentionally handled only by
        # gaussseed.pipeline after canonical proposal generation.
        torch.autograd.set_detect_anomaly(True)
        # 初始化设备和图像形状
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        # 处理相机内参
        intr_curr = context["intrinsics"][:, :, :3, :3].clone().detach()
        intr_curr[:, :, 0, :] *= float(w)
        intr_curr[:, :, 1, :] *= float(h)
        camk = (
            torch.eye(4)
            .view(1, 1, 4, 4)
            .repeat(intr_curr.shape[0], intr_curr.shape[1], 1, 1)
            .to(intr_curr.device)
            .float()
        )
        camk[:, :, :3, :3] = intr_curr
        c2w = context["extrinsics"].clone().detach()

        camk = torch.inverse(camk)
        img2world = torch.matmul(c2w, camk)
        # img2world=None


        # 主干网络
        # 调用模型的背骨网络（self.backbone），输入图像和变换矩阵。
        # 返回两个特征：
        #     trans_features：变换后的特征。
        #     cnn_features：CNN提取的特征。 
                # 图像归一化和深度估计
        # with torch.no_grad():
        da_images = self.normalize_images(context["image"])
            # da_images = da_images[:,:,[2, 0, 1]]
            # b, v, c, h, w = da_images.shape
            # da_images = da_images.view(b*v, c, h, w)
            # da_images = F.interpolate(da_images, (252, 252), mode="bilinear", align_corners=True)

        # with torch.no_grad():
        dinov3_feature, depth_feature,depth_dino = self.backbone.forward(da_images[0])
        dinov3_feature_tran = self.upsample_conv(dinov3_feature.clone())
        depth_feature_tran = self.depth_conv(depth_feature.clone()).unsqueeze(0)
        depth = F.interpolate(
            depth_dino.clone(), 
            size=(256, 256),  # 目标尺寸
            mode='bilinear',     # 插值方式
            align_corners=False  # 是否对齐角落像素
            ).unsqueeze(0)
        
        # 展平空间维度，方便计算全局最大/最小值
        depth = depth.flatten(2)
        # 计算每个样本-视图的最大值（保留维度）
        da_max = torch.max(depth, dim=-1, keepdim=True)[0]
        # 计算每个样本-视图的最小值（保留维度）
        da_min = torch.min(depth, dim=-1, keepdim=True)[0]
        # 归一化到0-1：(x - min)/(max - min)
        depth = (depth - da_min) / (da_max - da_min)
        # 恢复空间维度形状
        depth = depth.reshape(b, v, 1, h, w)



            # da_depth = F.interpolate(da_depth[None], (h, w), mode="bilinear", align_corners=True)
            # da_depth = da_depth.view(b, v, 1, h, w)
            # # normalize to 0 - 1
            # da_depth = da_depth.flatten(2)
            # da_max = torch.max(da_depth, dim=-1, keepdim=True)[0]
            # da_min = torch.min(da_depth, dim=-1, keepdim=True)[0]
            # da_depth = (da_depth - da_min) / (da_max - da_min)
            # da_depth = da_depth.reshape(b, v, 1, h, w)
        # 深度估计模型的输出特征重新调整形状，以匹配输入图像的批量和视图维度
        trans_features, cnn_features, updated_pose, delta_T = self.backbone_t(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            camk=camk,
            c2w=c2w,
            dino_feature = dinov3_feature_tran,
            depth_feature = depth_feature_tran[0],
            pose_ture = pose_ture,
            dinov3_feature = dinov3_feature
        )
        out_ex=update_extrinsics(context["extrinsics"], updated_pose)
        


        # show depthanything result here
        '''
        import cv2
        lowres_da_depth = F.interpolate(
            da_depth[0],
            scale_factor=0.25,
            mode="bilinear",
            align_corners=True,
        )

        depth_vis = lowres_da_depth[0,0].cpu().numpy()
        depth_vis = depth_vis * 255
        depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('vis_lor.png', depth_rgb)

        image_vis = context["image"][0,0].permute(1, 2, 0)
        image_vis = image_vis.cpu().numpy()
        image_vis = image_vis * 255
        cv2.imwrite('image.png', image_vis)
        import pdb
        pdb.set_trace()
        '''

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        # 深度预测
        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            out_ex,
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            da_depth=depth,
            dino_feature=depth_feature_tran,
        )
        metric_depth = rearrange(
            depths[..., 0],
            "b v (h w) srf -> b v srf h w",
            h=h,
            w=w,
        )

        # Convert proposal attributes to canonical Gaussians.  Scale alignment,
        # consistency filtering, novelty selection, and backend conversion live
        # exclusively in gaussseed.pipeline.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gaussians = self.gaussian_adapter.forward(
            rearrange(out_ex, "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )
        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1



        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        ),out_ex ,delta_T,metric_depth

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            # TTRANS Do not patch now
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None


def update_extrinsics(batch, updated_pose):
    # 创建输入extrinsics的副本
    extrinsics = batch.clone()
    
    
    
    
    # 创建新的extrinsics张量
    updated_extrinsics = extrinsics.clone()
    updated_extrinsics[:, 1:, :, :] = updated_pose.unsqueeze(1)
    
    return updated_extrinsics
