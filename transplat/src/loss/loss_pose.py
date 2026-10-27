from dataclasses import dataclass
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


def _det3x3(matrix: Tensor) -> Tensor:
    return (
        matrix[..., 0, 0] * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1] * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2] * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


# ------------------------------
# 配置类（定义损失的超参数）
# ------------------------------
@dataclass
class LossPoseCfg:
    weight: float  # 损失权重
    alpha: float = 1.0  # 基础位姿损失权重（对应pose_loss中的alpha）
    beta: float = 0  # 几何一致性损失权重（对应pose_loss中的beta）


@dataclass
class LossPoseCfgWrapper:
    pose: LossPoseCfg  # 包装配置，与现有LossLpipsCfgWrapper等格式保持一致


# ------------------------------
# 损失计算类（实现forward方法）
# ------------------------------

class LossPose(Loss[LossPoseCfg, LossPoseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        out_ex: Tensor, 
        delta_T_encoder: Tensor,
        depth_ground:Tensor
    ) -> Float[Tensor, ""]:
        # 提取预测位姿和源/目标位姿（仅两张图，保持单对逻辑）
        updated_extrinsics = out_ex
        pred_poses = updated_extrinsics[:, 1:2, :, :]  # [B, 1, 4, 4]
        ne_pose = batch["context"]["extrinsics"][:, 1:2, :, :].detach()  # 目标位姿（单张）
        re_pose = batch["context"]["extrinsics"][:, 0:1, :, :].detach()  # 源位姿（单张）
        
        # 计算真实位姿增量 delta_T = 目标位姿 × 源位姿逆（单对图片）
        # src_pose_inv = torch.inverse(re_pose)
        # delta_T = torch.matmul(src_pose_inv, ne_pose)  # 真实增量
        delta_T = batch["context"]["extrinsics"][:, 1:2, :, :].detach()
        
        # 拆分旋转和平移目标（适配输入格式：R_target [B,1,3,3]，t_target [B,1,3]）
        t_target = delta_T[..., :3, 3]  # 对应输入t_target的维度
        R_target = delta_T[..., :3, :3]  # 对应输入R_target的维度
        
        # 单对图片的损失计算（移除多视图逻辑）
        loss_val = pose_loss_zeng(
            # delta_T=delta_T_encoder,
            delta_T=pred_poses,
            R_target=R_target,
            t_target=t_target,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            src_pose=re_pose,  # 仅用于单对验证
            tgt_pose=ne_pose   # 仅用于单对验证
        )

        return self.cfg.weight * loss_val


def pose_loss_zeng(delta_T, R_target, t_target, alpha=1.0, beta=0.2, src_pose=None, tgt_pose=None):
    # 提取预测旋转和平移（适配输入delta_T的4x4矩阵）
    pred_rot = delta_T[..., :3, :3]  # 预测旋转矩阵 [B,1,3,3]
    pred_trans = delta_T[..., :3, 3]  # 预测平移向量 [B,1,3]
    
    # 1. 基础旋转和平移损失（强化单对图片精度）
    # 旋转损失：增强正交性约束（对单矩阵关键）
    rot_loss = enhanced_rotation_loss(pred_rot, R_target)
    # 平移损失：优化小平移场景（输入t_target数值较小，增强敏感度）
    trans_loss = small_translation_loss(pred_trans, t_target)
    base_loss = rot_loss + trans_loss
    
    # 2. 单对图片的几何一致性（仅验证源×预测增量是否接近目标）
    consistency_loss = single_pair_consistency(
        pred_delta=delta_T,
        src_pose=src_pose,
        tgt_pose=tgt_pose
    )
    
    # 组合损失（侧重基础损失，降低冗余约束权重）
    return alpha * base_loss + beta * consistency_loss


def enhanced_rotation_loss(pred_R, target_R):
    """针对单旋转矩阵的增强损失：确保正交性+角度精度"""
    # 1. 角度误差（核心约束）
    R_diff = torch.matmul(pred_R.transpose(-1, -2), target_R)  # 相对旋转
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    # 限制输入范围，避免数值波动（单对图片更敏感）
    angle_rad = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
    angle_loss = F.smooth_l1_loss(angle_rad, torch.zeros_like(angle_rad))
    
    # 2. 正交性约束（单旋转矩阵必须满足R^T R = I）
    identity = torch.eye(3, device=pred_R.device).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
    ortho_loss = torch.norm(torch.matmul(pred_R.transpose(-1, -2), pred_R) - identity, p=2)
    
    # 3. 旋转矩阵行列式约束（确保右手系，单矩阵必要条件）
    det = _det3x3(pred_R)  # 旋转矩阵行列式应为1
    det_loss = F.smooth_l1_loss(det, torch.ones_like(det))
    
    return 0.5 * angle_loss + 0.3 * ortho_loss + 0.2 * det_loss


def small_translation_loss(pred_t, target_t):
    """针对小平移（如输入t_target≈0.03）的增强损失：提高敏感度"""
    # 1. 绝对误差（小数值下L1比L2更敏感）
    l1_loss = F.l1_loss(pred_t, target_t)  # 替换smooth_l1，增强小误差反馈
    
    # 2. 方向误差（小向量方向更关键）
    eps = 1e-8  # 避免小向量除零
    pred_norm = torch.norm(pred_t, p=2, dim=-1, keepdim=True) + eps
    target_norm = torch.norm(target_t, p=2, dim=-1, keepdim=True) + eps
    # 方向向量归一化后再计算余弦相似度（小向量方向误差放大）
    dir_pred = pred_t / pred_norm
    dir_target = target_t / target_norm
    direction_loss = 1 - (dir_pred * dir_target).sum(dim=-1).mean()  # 点积直接衡量方向
    
    # 3. 尺度误差（小向量尺度误差相对影响更大）
    scale_ratio = pred_norm / target_norm
    # 对小尺度（target_norm < 0.1）加大惩罚权重
    scale_weight = torch.where(target_norm < 0.1, 5.0, 1.0)  # 小平移尺度误差权重×5
    scale_loss = (torch.abs(scale_ratio - 1.0) * scale_weight).mean()
    
    return 0.4 * l1_loss + 0.4 * direction_loss + 0.2 * scale_loss


def single_pair_consistency(pred_delta, src_pose, tgt_pose):
    """单对图片的一致性验证：源位姿×预测增量 应接近目标位姿"""
    if src_pose is None or tgt_pose is None:
        return torch.tensor(0.0, device=pred_delta.device)
    
    # 应用预测增量到源位姿：src_pose × pred_delta 应 ≈ tgt_pose
    pred_tgt = torch.matmul(src_pose, pred_delta)  # [B,1,4,4]
    # 计算整体位姿误差（旋转+平移联合约束）
    # 旋转部分误差
    rot_error = torch.norm(pred_tgt[..., :3, :3] - tgt_pose[..., :3, :3], p=2)
    # 平移部分误差（适配小平移，用L1）
    trans_error = F.l1_loss(pred_tgt[..., :3, 3], tgt_pose[..., :3, 3])
    
    return 0.5 * rot_error + 0.5 * trans_error


# 保留必要的工具函数（适配单对图片场景）
def matrix_to_quaternion(rot_mat):
    """旋转矩阵转四元数（用于旋转表示辅助约束）"""
    batch_dim = rot_mat.shape[:-2]
    r00, r01, r02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    r10, r11, r12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    r20, r21, r22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]
    
    w = torch.sqrt(torch.clamp(1.0 + r00 + r11 + r22, min=1e-8)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=1e-8)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=1e-8)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=1e-8)) / 2.0
    
    x = x * torch.sign(r21 - r12)
    y = y * torch.sign(r02 - r20)
    z = z * torch.sign(r10 - r01)
    
    return torch.stack([w, x, y, z], dim=-1).view(*batch_dim, 4)

# 保留原始函数以确保引用兼容（但单对场景下主要使用上述新函数）
def rotation_loss(pred_R, target_R):
    R_diff = torch.matmul(pred_R.transpose(-1, -2), target_R)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle_rad = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    return F.smooth_l1_loss(angle_rad, torch.zeros_like(angle_rad))

def log_translation_loss(pred_t, target_t):
    pred_norm = torch.norm(pred_t, p=2, dim=-1, keepdim=True)
    target_norm = torch.norm(target_t, p=2, dim=-1, keepdim=True)
    direction_loss = 1 - F.cosine_similarity(pred_t, target_t, dim=-1)
    scale_loss = torch.abs(pred_norm - target_norm) / (target_norm + 1e-6)
    return 0.7 * direction_loss.mean() + 0.3 * scale_loss.mean()

def quaternion_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w,
        2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w,
        2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y
    ], dim=-1).view(*(q.shape[:-1]), 3, 3)
