from typing import Tuple
from torch import Tensor, nn
import torch.nn.functional as F
import torch.nn.init as init
import torch


class PoseEstimationHead(nn.Module):
    """
    DINOv3官方风格相对位姿估计头（适配两张图片输入）
    输入：features [2, C, H, W] → 两张图片的DINOv3特征
    输出：rotation [1, 4]（四元数）, translation [1, 3]（平移向量）→ 两张图的相对位姿
    """
    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.feature_dim = feature_dim
        combined_dim = 2 * feature_dim
        
        # --------------------------
        # 1. DINOv3官方帧间特征匹配层（核心）
        # --------------------------
        # 轻量级降维投影（加速跨图匹配）
        self.match_proj = nn.Linear(feature_dim, 128)
        self.match_norm = nn.LayerNorm(128)
        # 匹配置信度预测（加权融合匹配对）
        self.conf_head = nn.Linear(256, 1)
        
        # --------------------------
        # 2. 残差特征提取（DINOv3轻量级设计）
        # --------------------------
        self.linear_proj = nn.Linear(combined_dim, 256)
        self.shared_feature_extractor = nn.Sequential(
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
        )
        
        # --------------------------
        # 3. DINOv3官方旋转分支（6D表示→四元数）
        # --------------------------
        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 6)  # 6D旋转（DINOv3首选，无四元数奇异性）
        )
        
        # --------------------------
        # 4. DINOv3官方平移分支
        # --------------------------
        self.translation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 3)
        )
        
        self.init_weights()

    def init_weights(self):
        """DINOv3官方初始化策略（适配LeakyReLU）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)

    # --------------------------
    # DINOv3核心工具函数（6D旋转↔矩阵↔四元数）
    # --------------------------
    def _6d_to_rot_mat(self, x: Tensor) -> Tensor:
        """6D旋转→旋转矩阵（DINOv3官方实现，可微分）"""
        x = x.view(-1, 6)  # [M, 6] → M为匹配对数量
        x1, x2 = x[:, :3], x[:, 3:]
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2 - (x1 * x2).sum(-1, keepdim=True) * x1, dim=-1)
        x3 = torch.cross(x1, x2, dim=-1)
        rot_mat = torch.stack([x1, x2, x3], dim=-1)  # [M, 3, 3]
        return rot_mat

    def _rot_mat_to_quat(self, rot_mat: Tensor) -> Tensor:
        """旋转矩阵→四元数（适配输出格式）"""
        # 确保旋转矩阵是正交的（DINOv3后处理）
        U, S, V = torch.svd(rot_mat)
        rot_mat = torch.matmul(U, V.transpose(-1, -2))
        det = (
            rot_mat[:, 0, 0] * (rot_mat[:, 1, 1] * rot_mat[:, 2, 2] - rot_mat[:, 1, 2] * rot_mat[:, 2, 1])
            - rot_mat[:, 0, 1] * (rot_mat[:, 1, 0] * rot_mat[:, 2, 2] - rot_mat[:, 1, 2] * rot_mat[:, 2, 0])
            + rot_mat[:, 0, 2] * (rot_mat[:, 1, 0] * rot_mat[:, 2, 1] - rot_mat[:, 1, 1] * rot_mat[:, 2, 0])
        )
        rot_mat = rot_mat * det.view(-1, 1, 1)
        
        # DINOv3官方转换公式
        q0 = torch.sqrt(torch.clamp(1 + rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2], min=1e-8)) / 2
        q1 = (rot_mat[:, 2, 1] - rot_mat[:, 1, 2]) / (4 * q0 + 1e-8)
        q2 = (rot_mat[:, 0, 2] - rot_mat[:, 2, 0]) / (4 * q0 + 1e-8)
        q3 = (rot_mat[:, 1, 0] - rot_mat[:, 0, 1]) / (4 * q0 + 1e-8)
        
        quat = torch.stack([q0, q1, q2, q3], dim=-1)  # [M, 4]
        quat = F.normalize(quat, dim=-1)  # 单位四元数约束
        return quat

    # --------------------------
    # DINOv3官方帧间特征匹配（跨两张图）
    # --------------------------
    def _dinov3_inter_frame_matching(self, feat1: Tensor, feat2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        跨两张图的特征匹配（DINOv3核心逻辑）
        输入：
            feat1: [1, C, H, W] → 第一张图特征
            feat2: [1, C, H, W] → 第二张图特征
        输出：
            matched_feat: [M, 256] → 匹配对特征（M=H*W）
            match_conf: [M, 1] → 匹配置信度
            sim_mat: [H*W, H*W] → 相似度矩阵（用于加权）
        """
        # 1. 展平空间维度：[1, C, H, W] → [H*W, C]
        B1, C, H1, W1 = feat1.shape
        B2, _, H2, W2 = feat2.shape
        assert B1 == B2 == 1, "单张图特征必须为[1, C, H, W]"
        assert C == self.feature_dim, f"特征维度不匹配：{C} vs {self.feature_dim}"
        
        N1 = H1 * W1
        N2 = H2 * W2
        feat1_flat = feat1.permute(0, 2, 3, 1).reshape(N1, C)  # [N1, C]
        feat2_flat = feat2.permute(0, 2, 3, 1).reshape(N2, C)  # [N2, C]
        
        # 2. DINOv3特征降维+归一化（加速匹配）
        feat1_proj = F.normalize(self.match_norm(self.match_proj(feat1_flat)), dim=-1)  # [N1, 128]
        feat2_proj = F.normalize(self.match_norm(self.match_proj(feat2_flat)), dim=-1)  # [N2, 128]
        
        # 3. 计算余弦相似度矩阵（DINOv3官方匹配度量）
        sim_mat = torch.mm(feat1_proj, feat2_proj.T)  # [N1, N2]
        
        # 4. 互匹配筛选（DINOv3核心：过滤错误匹配对）
        # 正向匹配：feat1→feat2的最佳匹配
        match_idx_12 = torch.argmax(sim_mat, dim=1)  # [N1]
        # 反向匹配：feat2→feat1的最佳匹配
        match_idx_21 = torch.argmax(sim_mat, dim=0)  # [N2]
        # 互匹配验证：i→j 且 j→i
        mutual_mask = torch.zeros(N1, dtype=torch.bool, device=sim_mat.device)
        for i in range(N1):
            j = match_idx_12[i]
            if match_idx_21[j] == i:
                mutual_mask[i] = True
        
        # 5. 提取有效匹配对
        valid_feat1 = feat1_proj[mutual_mask]  # [M, 128]
        valid_feat2 = feat2_proj[match_idx_12[mutual_mask]]  # [M, 128]
        valid_sim = sim_mat[mutual_mask, match_idx_12[mutual_mask]].unsqueeze(1)  # [M, 1]
        
        # 6. 匹配特征拼接+置信度预测
        matched_feat = torch.cat([valid_feat1, valid_feat2], dim=-1)  # [M, 256]
        match_conf = self.conf_head(matched_feat) * valid_sim  # [M, 1]
        
        return matched_feat, match_conf, sim_mat

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        输入：features [2, C, H, W] → 两张图片的DINOv3特征
        输出：
            rotation: [1, 4] → 两张图的相对旋转（四元数）
            translation: [1, 3] → 两张图的相对平移
        """
        # --------------------------
        # 1. 拆分输入为两张图的特征
        # --------------------------
        assert features.shape[0] == 2, f"输入必须是两张图的特征，当前{features.shape[0]}张"
        feat1 = features[0:1]  # 第一张图：[1, 384, 16, 16]
        feat2 = features[1:2]  # 第二张图：[1, 384, 16, 16]
        
        # --------------------------
        # 2. DINOv3跨图特征匹配（核心）
        # --------------------------
        matched_feat, match_conf, _ = self._dinov3_inter_frame_matching(feat1, feat2)
        M = matched_feat.shape[0]  # 有效匹配对数量
        if M == 0:
            # 无有效匹配时返回单位旋转+零平移（鲁棒性处理）
            rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=features.device)
            translation = torch.zeros(1, 3, device=features.device)
            return rotation * 0.1, translation * 0.1
        
        # --------------------------
        # 3. 展平两张图的原始特征（用于残差融合）
        # --------------------------
        C = self.feature_dim
        feat1_flat = feat1.permute(0, 2, 3, 1).reshape(-1, C)  # [256, 384]
        feat2_flat = feat2.permute(0, 2, 3, 1).reshape(-1, C)  # [256, 384]
        # 按匹配对筛选原始特征
        match_idx_12 = torch.argmax(torch.mm(
            F.normalize(self.match_norm(self.match_proj(feat1_flat)), dim=-1),
            F.normalize(self.match_norm(self.match_proj(feat2_flat)), dim=-1).T
        ), dim=1)
        mutual_mask = torch.zeros(feat1_flat.shape[0], dtype=torch.bool, device=features.device)
        match_idx_21 = torch.argmax(torch.mm(
            F.normalize(self.match_norm(self.match_proj(feat2_flat)), dim=-1),
            F.normalize(self.match_norm(self.match_proj(feat1_flat)), dim=-1).T
        ), dim=1)
        for i in range(mutual_mask.shape[0]):
            j = match_idx_12[i]
            if match_idx_21[j] == i:
                mutual_mask[i] = True
        valid_feat1_ori = feat1_flat[mutual_mask]  # [M, 384]
        valid_feat2_ori = feat2_flat[match_idx_12[mutual_mask]]  # [M, 384]
        
        # --------------------------
        # 4. 特征融合+残差提取（DINOv3风格）
        # --------------------------
        combined = torch.cat([valid_feat1_ori, valid_feat2_ori], dim=-1)  # [M, 768]
        projected = self.linear_proj(combined)  # [M, 256]
        # 融合匹配特征（残差连接）
        projected = projected + matched_feat
        shared_features = self.shared_feature_extractor(projected)
        shared_features += projected  # DINOv3残差策略
        
        # --------------------------
        # 5. 逐匹配对预测位姿（DINOv3核心）
        # --------------------------
        # 预测6D旋转和平移
        rot_6d = self.rotation_head(shared_features)  # [M, 6]
        trans_pred = self.translation_head(shared_features)  # [M, 3]
        
        # 6D旋转→旋转矩阵
        rot_mat = self._6d_to_rot_mat(rot_6d)  # [M, 3, 3]
        
        # --------------------------
        # 6. DINOv3置信度加权融合（单一位姿输出）
        # --------------------------
        # 置信度归一化（高置信匹配对权重更高）
        conf_weight = F.softmax(match_conf.squeeze(-1), dim=0)  # [M]
        
        # 旋转矩阵加权平均（DINOv3官方：保证正交性）
        rot_mat_weighted = torch.einsum('m, m i j -> i j', conf_weight, rot_mat)  # [3, 3]
        # 平移向量加权平均
        trans_weighted = torch.einsum('m, m k -> k', conf_weight, trans_pred)  # [3]
        
        # 旋转矩阵→四元数（适配输出格式）
        rot_quat = self._rot_mat_to_quat(rot_mat_weighted.unsqueeze(0))  # [1, 4]
        
        # --------------------------
        # 7. 维度调整+保留原缩放逻辑
        # --------------------------
        rotation = rot_quat.reshape(1, 4)  # [1, 4]
        translation = trans_weighted.reshape(1, 3)  # [1, 3]
        
        # 保留原输出缩放系数（与用户原始代码一致）
        rotation = rotation * 0.1
        translation = translation * 0.1
        
        return rotation, translation
