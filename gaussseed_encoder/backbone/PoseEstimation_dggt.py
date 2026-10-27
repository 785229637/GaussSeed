import torch
import torch.nn as nn
import torch.nn.functional as F

from dggt.dggt.layers import Mlp
from dggt.dggt.layers.block import Block
from dggt.dggt.heads.head_act import activate_pose


class PoseEstimationHead(nn.Module):
    def __init__(
        self,
        dino_in_dim: int = 384,
        dim_in: int = 2048,  # 原模型特征维度
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported encoding type: {pose_encoding_type}")

        # 修正特征适配层：确保输出形状为 [B, 1, dim_in]
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(dino_in_dim, dim_in, kernel_size=1, stride=1, padding=0),  # [B, 384, 16, 16] -> [B, 2048, 16, 16]
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化 -> [B, 2048, 1, 1]
            nn.Flatten(start_dim=1)  # 展平空间维度 -> [B, 2048]
        )

        # 新增维度调整：将 [B, 2048] 转换为 [B, 1, 2048]（匹配原token格式）
        self.adjust_dim = nn.Identity()  # 仅用于维度调整，无参数

        # 以下参数与原CameraHead保持一致
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = nn.LayerNorm(dim_in)  # 期望输入最后一维为 dim_in (2048)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )


    def quat_conj(self,q: torch.Tensor) -> torch.Tensor:
        """
        计算单位四元数的共轭（逆），输入为wxyz格式，支持任意批量维度
        单位四元数的逆 = 共轭（虚部取反，实部不变）
        
        Args:
            q: 输入四元数，torch.Tensor，shape=(..., 4)，wxyz顺序
        Returns:
            共轭四元数，torch.Tensor，shape=(..., 4)，wxyz顺序
        """
        conj_q = q.clone()
        # 虚部（x/y/z）取反，实部（w）保持不变
        conj_q[..., 1:] = -conj_q[..., 1:]
        return conj_q

    def quat_mult(self,q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        四元数乘法（q1 * q2），遵循wxyz格式的乘法规则，支持批量输入
        核心公式：
        w = w1w2 - x1x2 - y1y2 - z1z2
        x = w1x2 + x1w2 + y1z2 - z1y2
        y = w1y2 - x1z2 + y1w2 + z1x2
        z = w1z2 + x1y2 - y1x2 + z1w2
        
        Args:
            q1: 第一个四元数，torch.Tensor，shape=(..., 4)
            q2: 第二个四元数，torch.Tensor，shape=(..., 4)
        Returns:
            乘积四元数，torch.Tensor，shape=(..., 4)
        """
        # 拆分四元数分量（支持任意批量维度）
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # 计算乘法结果
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        # 拼接结果，保持原批量维度
        q_out = torch.stack([w, x, y, z], dim=-1)
        return q_out

    def get_relative_quat(self,q_prev: torch.Tensor, q_curr: torch.Tensor) -> torch.Tensor:
        """
        计算从初始姿态q_prev到当前姿态q_curr的相对位姿四元数
        数学关系：q_curr = q_rel * q_prev → q_rel = q_curr * q_prev⁻¹
        （若需 q_prev * q_rel = q_curr，可改为 quat_mult(quat_conj(q_prev), q_curr)）
        
        Args:
            q_prev: 初始姿态四元数，torch.Tensor，shape=(..., 4)，wxyz，单位四元数
            q_curr: 当前姿态四元数，torch.Tensor，shape=(..., 4)，wxyz，单位四元数
        Returns:
            相对位姿四元数，torch.Tensor，shape=(..., 4)，wxyz，单位四元数
        """
        # 计算初始四元数的逆（共轭）
        q_prev_inv = self.quat_conj(q_prev)
        # 计算相对四元数
        q_rel = self.quat_mult(q_curr, q_prev_inv)
        # 归一化（消除数值误差，保证单位四元数特性）
        q_rel = q_rel / torch.norm(q_rel, dim=-1, keepdim=True)
        return q_rel

    def forward(self, dino_feature: torch.Tensor, num_iterations: int = 4) -> list:
        # 1. 特征适配：将DINO特征转换为 [B, 1, dim_in]
        feat = self.feature_adapter(dino_feature)  # [B, 2048]（展平后）
        pose_tokens = feat.unsqueeze(1)  # 增加token维度 -> [B, 1, 2048]（关键修正）
        pose_tokens = self.token_norm(pose_tokens)  # 现在形状匹配：[B, 1, 2048]

        # 2. 调用原迭代优化逻辑
        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    # 以下方法与原CameraHead完全一致
    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        B, S, C = pose_tokens.shape  # S=1（单token）
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * self.modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)
            rotations = self.get_relative_quat(pred_pose_enc_list[-1][0][0][3:7],pred_pose_enc_list[-1][1][0][3:7])
            translations = pred_pose_enc_list[-1][1][0][0:3] - pred_pose_enc_list[-1][0][0][0:3]

        return rotations.unsqueeze(0) ,translations.unsqueeze(0)

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift




# 测试代码
if __name__ == "__main__":
    dino_feature_y = torch.randn(2, 384, 16, 16)  # 输入形状 [B=2, C=384, H=16, W=16]
    camera_head = PoseEstimationHead()
    pose_list = camera_head(dino_feature_y)
    final_pose = pose_list[-1]
    print(f"最终相机姿态形状: {final_pose.shape}")  # 应为 [2, 1, 9]
