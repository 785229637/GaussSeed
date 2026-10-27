

import os
import torch.nn as nn
import torch
# from torchvision.ops import roi_align
from dinov3.eval.dense.depth.models.embed import CenterPadding
from dinov3.hub import backbones as dinov3_backbones
from dinov3.eval.dense.depth.models.dpt_head import DPTHead

# from model.pmfs3d import ConvBlock, DownSampleWithLocalPMFSBlock, GlobalPMFSBlock_AP_Separate
import time

from gaussseed_encoder.DepthHead_ import DepthHeadLite

# from gaussseed_encoder.DepthHead_ import DepthHeadMultiBranch


def _torch_load_weights(path):
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))

class FeaturesToDepth(torch.nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=80,
        bins_strategy="linear",
        norm_strategy="linear",
    ):
        """
        Module which converts a feature maps into a depth map

        Args:
        min_depth (float): minimum depth, used to calibrate the depth range
        max_depth (float): maximum depth, used to calibrate the depth range
        bins_strategy (str): Choices are 'linear' or 'log', for Uniform or Scale Invariant distributions for depth bins.
                             See AdaBins [1] for more details.
        norm_strategy (str): Choices are 'linear', 'softmax' or 'sigmoid', for the conversion of features to depth logits
        scale_up (bool): If true, and only if regression by classification is not used, the result is multiplied by max_depth


        Example:
        x = depth_model(input_image)  # N C H W
        - If pure regression (C == 1), depth is obtained by scaling and/or shifting x
        - If C > 1, bins are used:
            Depth is obtained as a weighted sum of depth bins, where weights are predicted logits. (see AdaBins [1] for more details)

        [1] AdaBins: https://github.com/shariqfarooq123/AdaBins
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert bins_strategy in ["linear", "log"], "Support bins_strategy: linear, log"
        assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x):
        n_bins = x.shape[1]  # N n_bins H W
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                    device=x.device,
                )
                bins = torch.exp(bins)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(x)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(x, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(x)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        else:
            # standard regression
            output = torch.relu(x) + self.min_depth
        return output

# class DinoVisionTransformerWrapper(nn.Module):
#     """Vision Transformer."""

#     def __init__(
#         self,
#         backbone_model: nn.Module,
#         backbone_out_layers: list =[5, 11, 17, 23] ,
#         use_backbone_norm: bool = False,
#         adapt_to_patch_size:str ="center_padding",
#     ):
#         super().__init__()

#         self.final_norm = use_backbone_norm
#         self.backbone = backbone_model
#         self.backbone_out_indices = backbone_out_layers
#         # If the backbone does not define embed_dims, use [embed_dim] * n_blocks
#         try:
#             embed_dims: list[int] = getattr(self.backbone, "embed_dims")
#         except AttributeError:
#             embed_dim: int = getattr(self.backbone, "embed_dim")
#             n_blocks: int = getattr(self.backbone, "n_blocks")
#             print(f"Backbone does not define embed_dims, using {[embed_dim] * n_blocks} instead")
#             embed_dims = [embed_dim] * n_blocks
#         self.embed_dims = [embed_dims[idx] for idx in self.backbone_out_indices]

#         # How to adapt input images to the patch size of the model?
#         try:
#             input_pad_size = getattr(self.backbone, "input_pad_size")
#         except AttributeError:
#             patch_size = getattr(self.backbone, "patch_size")
#             print(f"Backbone does not define input_pad_size, using {patch_size=} instead")
#             input_pad_size = patch_size
#         self.patch_size_adapter: nn.Module = nn.Identity()

#         self.patch_size_adapter = CenterPadding(input_pad_size)


#         # Freeze backbone
#         self.backbone.requires_grad_(False)

#     def forward(
#         self,
#         x: torch.Tensor,  # [B, rgb, H, W]
#     ) -> list[tuple[torch.Tensor, torch.Tensor]]:
#         x = self.patch_size_adapter(x)
#         outputs = self.backbone.get_intermediate_layers(  # type: ignore
#             x,
#             n=self.backbone_out_indices,
#             reshape=True,
#             return_class_token=True,
#             norm=self.final_norm,
#         )  # List of (patch feats [B, C, h, w], class token [B, C])
#         return outputs
    

class Dinov3Backbone(nn.Module):
    def __init__(
            self,
            num_classes=1,
            dinov3_arch="vit_s",
            dinov3_ckpt=None,
            depth_arch="dpt",
            dpt_head_ckpt=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dinov3_arch = dinov3_arch
        self.dinov3_type, self.dinov3_size = dinov3_arch.split("_")
        self.backbone_dinov3 = None
        if self.dinov3_type == 'convnext':
            if self.dinov3_size == 'tiny':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_tiny(pretrained=False)

            elif self.dinov3_size == 'small':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_small(pretrained=False)
                # 假设small版本的第一个特征图通道数不同，这里留空以便后续扩展
                # self.channel_conv = nn.Conv2d(input_channels, 128, kernel_size=1, stride=1, padding=0)
            elif self.dinov3_size == 'base':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_base(pretrained=False)
            elif self.dinov3_size == 'large':
                self.backbone_dinov3 = dinov3_backbones.dinov3_convnext_large(pretrained=False)

            self.backbone_dims = self.backbone_dinov3.embed_dims
        elif self.dinov3_type == 'vit':
            if self.dinov3_size == 's':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vits16(pretrained=False)
            elif self.dinov3_size == 's+':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vits16plus(pretrained=False)
            elif self.dinov3_size == 'b':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vitb16(pretrained=False)
            elif self.dinov3_size == 'l':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vitl16(pretrained=False)
            elif self.dinov3_size == 'h+':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vith16plus(pretrained=False)
            elif self.dinov3_size == '7b':
                self.backbone_dinov3 = dinov3_backbones.dinov3_vit7b16(pretrained=False)
            self.backbone_dims = [self.backbone_dinov3.embed_dim] * 4
        else:
            raise ValueError('Error dinov3_arch: %s' % dinov3_arch)

        if dinov3_ckpt is not None:
            weights = _torch_load_weights(dinov3_ckpt)
            self.backbone_dinov3.load_state_dict(weights, strict=True)

        # self.layers_to_use = [m * self.backbone_dinov3.n_blocks // 4 - 1 for m in range(1, 5)]
        # self.skip_spatial_attention = skip_spatial_attention
        # 启用骨干网络梯度计算
        # self.encoder = DinoVisionTransformerWrapper(
        #     backbone_model=self.backbone_dinov3,
        #     # backbone_out_layers=[5, 11, 17, 23],
        #     backbone_out_layers=[12],
        #     use_backbone_norm=True,
        #     adapt_to_patch_size="center_padding",
        # )

        
        # self.encoder.requires_grad_(False)

        self.depth = DepthHeadLite(in_ch=384, out_size=(480, 640))

        # make dpt depth decoder
        # use_batchnorm=True,
        # use_cls_token=False,
        # n_output_channels=256

        # self.decoder = self.make_head(embed_dims=self.encoder.embed_dims, n_output_channels=n_output_channels,
        #                          use_batchnorm=use_batchnorm, use_cls_token=use_cls_token, head_type=depth_arch,)
        
        # self.depth = DepthHeadMultiBranch(out_size=(480, 640))
        # self.depth.requires_grad = True
        
        if dpt_head_ckpt is not None:
            weights = _torch_load_weights(dpt_head_ckpt)
            state_dict = weights['model_state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                # 只保留depth相关的权重，剔除backbone_dinov3的权重
                if key.startswith("depth."):
                    # 去除"depth."前缀，匹配DepthHeadLite的键名
                    new_key = key.replace("depth.", "", 1)
                    new_state_dict[new_key] = value
            self.depth.load_state_dict(new_state_dict, strict=True)
        # self.decoder.requires_grad_(False)

    def make_head(
        self,
        embed_dims: int | list[int],
        n_output_channels: int,
        use_batchnorm: bool = False,
        use_cls_token: bool = False,
        # upsample: int = 4,
        head_type: str = "linear",
        **kwargs,
    ) -> torch.nn.Module:

        if isinstance(embed_dims, int):
            embed_dims = [embed_dims]

        if head_type == "linear":
            raise NotImplementedError
        elif head_type == "dpt":
            decoder = DPTHead(
                in_channels=embed_dims,
                n_output_channels=n_output_channels,
                readout_type="project" if use_cls_token else "ignore",
                use_batchnorm=use_batchnorm,** kwargs,  # TODO add here post_process_channels, n_hidden_channels
            )
        else:
            raise NotImplementedError("only linear and DPT head supported")
        return decoder

    def forward(self, x, max_depth = 100, min_depth = 0.01,training=False, gt=None):
        frame = x[:, :, :, :]  # (b, c, h, w)

        with torch.inference_mode():
            # f_raw_1_8, f_raw_1_16, f_raw_1_32 = self.backbone_dinov3.get_intermediate_layers(frame, n=[1, 2, 3],
            #                                                                                  reshape=True, norm=True)
            backbone_feature = self.backbone_dinov3.get_intermediate_layers(
                frame, n=12, reshape=True, return_class_token=True, norm=True
            )
            
            # 应用卷积层转换第一个特征图的通道数
            # 注意：这里假设backbone_feature[0]是特征图列表，backbone_feature[0][0]是我们要转换的张量
            # if isinstance(backbone_feature[0], list) and len(backbone_feature[0]) > 0:
                # 对特征图应用卷积
            # transformed_feature = self.channel_conv(backbone_feature[0][0])
                # 替换原始特征
                # backbone_feature[0] = [transformed_feature] + backbone_feature[0][1:]
                
            # depth_feature = self.decoder(backbone_feature)

            # features_to_depth = FeaturesToDepth(
            #     min_depth=0.001,
            #     max_depth=10,
            #     bins_strategy="linear",
            #     norm_strategy="linear",
            # )
        depth,depth_feature  = self.depth(backbone_feature[-1][0].clone())

        return backbone_feature[0][0], depth_feature, depth

# class Dinov3DepthTrain(Dinov3Backbone):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.features_to_depth = FeaturesToDepth(
#             min_depth=0.001,
#             max_depth=10,
#             bins_strategy="linear",
#             norm_strategy="linear",
#         )
#         # 新增上采样层：将模型原始输出（如128x160）缩放到480x640
#         # 替代Upsample的转置卷积方案（精度更高）
#         # 新增上采样层：将模型原始输出（如128x160）缩放到480x640
#         self.upsample = nn.Upsample(
#             size=(480, 640),  # 目标尺寸
#             mode='bilinear',  # 双线性插值（保持深度连续性）
#             align_corners=True  # 对齐角落像素，提升精度
#         )
#         # 新增：Dropout层（随机失活50%神经元，防止过拟合）
#         self.dropout = nn.Dropout(p=0.5)


#     def forward(self, x):
#         frame = x[:, :, :, :]
#         # with torch.inference_mode():
#         backbone_feature = self.backbone_dinov3.get_intermediate_layers(
#                 frame, n=[2, 5, 8, 11], reshape=True, return_class_token=True, norm=True
#             )
#         depth_feature = self.decoder(backbone_feature)
#         depth_feature = self.dropout(depth_feature)  # 解码后添加Dropout
#         pred_depth = self.features_to_depth(depth_feature)
#         # 关键：上采样到480x640
#         pred_depth = self.upsample(pred_depth)
#         return backbone_feature,depth_feature,pred_depth

# class depth_pre:
#     def __init__(self,resume_ckpt_path):
#         self.model = Dinov3DepthTrain(
#             dinov3_arch="vit_l",
#             dinov3_ckpt="checkpoints/dinov3_vitl16.pth",
#             depth_arch="dpt",
#             dpt_head_ckpt=None
#         ).to("cuda")

#         if not os.path.exists(resume_ckpt_path):
#             raise FileNotFoundError(f"续训失败：未找到checkpoint文件 {resume_ckpt_path}")
        
#         # 加载checkpoint
#         checkpoint = torch.load(resume_ckpt_path, map_location="cuda")
        
#         # 恢复模型参数
#         self.model.load_state_dict(checkpoint["model_state_dict"])
#         print(f"成功加载模型参数 from {resume_ckpt_path}")
    
#     def forward(self, image):
#         backbone_feature,depth_feature,pred_depth = self.model(image)
#         return backbone_feature,depth_feature,pred_depth

if __name__ == "__main__":

    x = torch.randn((2, 3, 672, 896)).float()
    net = Dinov3Backbone(dinov3_arch="convnext_large")
    features, depth = net(x, training=False)
    
    # 打印转换后的特征形状
    print("转换后的第一个特征图形状:", features[0][0].shape)  # 应该是 torch.Size([2, 128, 64, 64])
    print("depth_feature形状:", depth.shape)
