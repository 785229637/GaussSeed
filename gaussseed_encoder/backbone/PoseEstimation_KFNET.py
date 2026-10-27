import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 李代数相关函数（可微）
def exp_map(theta):
    """从李代数到李群的指数映射"""
    theta_norm = torch.norm(theta, dim=-1, keepdim=True)
    theta_norm = torch.max(theta_norm, torch.tensor(1e-8, device=theta.device))
    
    # 计算旋转矩阵
    cos_theta = torch.cos(theta_norm)
    sin_theta = torch.sin(theta_norm)
    t = (1 - cos_theta) / theta_norm**2
    s = sin_theta / theta_norm
    
    rx, ry, rz = theta[..., 0], theta[..., 1], theta[..., 2]
    zeros = torch.zeros_like(rx)
    
    # 构建旋转矩阵
    R = torch.stack([
        torch.stack([cos_theta + rx**2 * t, rx*ry*t - rz*s, rx*rz*t + ry*s], dim=-1),
        torch.stack([ry*rx*t + rz*s, cos_theta + ry**2 * t, ry*rz*t - rx*s], dim=-1),
        torch.stack([rz*rx*t - ry*s, rz*ry*t + rx*s, cos_theta + rz**2 * t], dim=-1)
    ], dim=-2)
    
    return R

def project_points(points_3d, K, R, t):
    """可微的3D点投影到2D图像平面"""
    # 旋转和平移3D点
    points_3d_transformed = torch.matmul(R, points_3d.unsqueeze(-1)).squeeze(-1) + t
    
    # 透视投影
    x = points_3d_transformed[..., 0] / points_3d_transformed[..., 2]
    y = points_3d_transformed[..., 1] / points_3d_transformed[..., 2]
    
    # 应用内参
    u = K[..., 0, 0] * x + K[..., 0, 2]
    v = K[..., 1, 1] * y + K[..., 1, 2]
    
    return torch.stack([u, v], dim=-1)


class FeatureExtractionNetwork(nn.Module):
    """特征提取网络，整合了原SCoordNet和OFlowNet的特征提取部分"""
    def __init__(self):
        super(FeatureExtractionNetwork, self).__init__()
        # SCoordNet特征提取部分
        self.sconv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.sconv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.sconv2a = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.sconv2b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.sconv3a = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.sconv3b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.sconv4a = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.sconv4b = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        
        # OFlowNet特征提取部分
        self.oconv0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.oconv1a = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.oconv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.oconv2a = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.oconv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.oconv3a = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.oconv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # 上采样部分
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # 卷积层（修正输入通道数）
        self.conv4 = nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16 + 32, 16, kernel_size=3, stride=1, padding=1)

    def preprocess(self, x):
        """原代码中的预处理步骤"""
        return (x - 128.0) * 0.00625

    def forward(self, images, camk):
        # images shape: [1, 2, 3, 256, 256]
        # camk shape: [1, 2, 4, 4]
        
        batch_size, num_views = images.shape[0], images.shape[1]
        features = []
        
        # 提取每个视角的特征
        for v in range(num_views):
            img = images[:, v]  # [1, 3, 256, 256]
            img = self.preprocess(img)
            
            # SCoordNet特征提取
            sfeat1a = F.relu(self.sconv1a(img))
            sfeat1b = F.relu(self.sconv1b(sfeat1a))
            sfeat2a = F.relu(self.sconv2a(sfeat1b))
            sfeat2b = F.relu(self.sconv2b(sfeat2a))
            sfeat3a = F.relu(self.sconv3a(sfeat2b))
            sfeat3b = F.relu(self.sconv3b(sfeat3a))
            sfeat4a = F.relu(self.sconv4a(sfeat3b))
            sfeat4b = F.relu(self.sconv4b(sfeat4a))
            
            # OFlowNet特征提取
            ofeat0 = F.relu(self.oconv0(img))
            ofeat1a = F.relu(self.oconv1a(ofeat0))
            ofeat1b = F.relu(self.oconv1b(ofeat1a))
            ofeat2a = F.relu(self.oconv2a(ofeat1b))
            ofeat2b = F.relu(self.oconv2b(ofeat2a))
            ofeat3a = F.relu(self.oconv3a(ofeat2b))
            ofeat3b = F.relu(self.oconv3b(ofeat3a))
            
            # 上采样路径
            ufeat2 = F.relu(self.upconv2(ofeat3b))
            concat2 = torch.cat([ufeat2, ofeat2b], dim=1)
            cfeat4 = F.relu(self.conv4(concat2))
            
            ufeat1 = F.relu(self.upconv1(cfeat4))
            concat1 = torch.cat([ufeat1, ofeat1b], dim=1)
            cfeat5 = F.relu(self.conv5(concat1))
            
            ufeat0 = F.relu(self.upconv0(cfeat5))
            concat0 = torch.cat([ufeat0, ofeat0], dim=1)
            cfeat6 = F.relu(self.conv6(concat0))
            
            # 整合特征
            view_features = {
                'sfeat': [sfeat1a, sfeat1b, sfeat2a, sfeat2b, sfeat3a, sfeat3b, sfeat4a, sfeat4b],
                'ofeat': [ofeat0, ofeat1a, ofeat1b, ofeat2a, ofeat2b, ofeat3a, ofeat3b],
                'ufeat': [ufeat2, ufeat1, ufeat0],
                'cfeat': [cfeat4, cfeat5, cfeat6]
            }
            
            features.append(view_features)
        
        return features


class RelativePoseEstimationNetwork(nn.Module):
    """相对位姿估计网络，基于卡尔曼滤波框架，使用可微位姿估计"""
    def __init__(self):
        super(RelativePoseEstimationNetwork, self).__init__()
        # SCoordNet预测头
        self.sconv5 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.sconv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.sconv7 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.sprediction = nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0)
        
        # OFlowNet预测头
        self.oprediction = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.uncertainty = nn.Linear(32, 1)
        
        self.min_uncertainty = 1e-5
        # 位姿优化器超参数
        self.optimization_steps = 10
        self.learning_rate = 0.01

    def forward(self, features, camk, last_coord=None, last_uncertainty=None):
        # 从特征提取网络获取特征
        feat1, feat2 = features[0], features[1]
        
        # SCoordNet预测
        sfeat4b = feat1['sfeat'][-1]
        sfeat5 = F.relu(self.sconv5(sfeat4b))
        sfeat6 = F.relu(self.sconv6(sfeat5))
        sfeat7 = F.relu(self.sconv7(sfeat6))
        spred = self.sprediction(sfeat7)
        
        coord_map = spred[:, :3, :, :]  # 场景坐标
        uncertainty_map = torch.exp(spred[:, 3:, :, :])  # 不确定性
        
        # OFlowNet预测
        cfeat6 = feat1['cfeat'][-1]
        opred = self.oprediction(cfeat6)
        prob_map = F.softmax(opred, dim=1)
        
        # 不确定性计算
        ofeat3b = feat1['ofeat'][-1]
        batch_size, channels, height, width = ofeat3b.shape
        feat = ofeat3b.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        fc1_out = F.relu(self.fc1(feat))
        fc2_out = F.relu(self.fc2(fc1_out))
        flow_uncertainty = torch.exp(self.uncertainty(fc2_out)) * 1e-2
        flow_uncertainty = flow_uncertainty.view(batch_size, 1, height, width)
        
        # 卡尔曼滤波更新
        if last_coord is None or last_uncertainty is None:
            # 初始帧，直接使用测量值
            kf_coord = coord_map
            kf_uncertainty = uncertainty_map
        else:
            # 卡尔曼增益计算
            last_variance = torch.square(last_uncertainty)
            measure_variance = torch.square(uncertainty_map)
            weight_K = last_variance / (last_variance + measure_variance)
            weight_K_3 = weight_K.repeat(1, 3, 1, 1)
            
            # 更新坐标和不确定性
            kf_coord = (1.0 - weight_K_3) * last_coord + weight_K_3 * coord_map
            kf_variance = (1.0 - weight_K) * last_variance
            kf_uncertainty = torch.sqrt(kf_variance)
        
        # 从场景坐标计算相对位姿（可微版本）
        relative_pose = self.compute_pose_from_coords(kf_coord, camk)
        
        return {
            'coord_map': coord_map,
            'uncertainty_map': uncertainty_map,
            'kf_coord': kf_coord,
            'kf_uncertainty': kf_uncertainty,
            'relative_pose': relative_pose
        }
    
    def compute_pose_from_coords(self, coord_map, camk):
        """从场景坐标计算相对位姿（修复梯度传播问题的可微实现）"""
        batch_size = coord_map.shape[0]
        height, width = coord_map.shape[2], coord_map.shape[3]
        
        # 初始化相对位姿输出 [B, 1, 4, 4]
        relative_pose = torch.eye(4, device=coord_map.device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
        
        # 对每个批次处理
        for b in range(batch_size):
            # 获取当前批次的相机内参 [3, 3]
            K = camk[b, 0, :3, :3]  # 使用第一个视角的内参
            
            # 从coord_map提取3D点 [H, W, 3]，确保保留梯度
            points_3d = coord_map[b].permute(1, 2, 0).requires_grad_(True)  # [H, W, 3]
            
            # 生成对应的2D图像点坐标，确保在正确设备上
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, device=coord_map.device),
                torch.arange(width, device=coord_map.device),
                indexing='ij'
            )
            points_2d = torch.stack([x_coords.float(), y_coords.float()], dim=-1).requires_grad_(False)  # 2D坐标不需要梯度
            
            # 过滤无效点（例如z值为负或接近零的点）
            valid_mask = points_3d[..., 2] > 1e-3
            if valid_mask.sum() < 6:  # 至少需要6个点
                continue  # 保持单位矩阵
            
            # 只使用有效点
            points_3d_valid = points_3d[valid_mask]
            points_2d_valid = points_2d[valid_mask]
            
            # 初始化位姿参数（李代数），确保在正确设备上并需要梯度
            pose_params = torch.zeros(6, device=coord_map.device, requires_grad=True)
            
            # 使用LBFGS优化器，更适合小参数集的优化
            optimizer = torch.optim.LBFGS(
                [pose_params],
                lr=self.learning_rate,
                max_iter=5,
                line_search_fn="strong_wolfe"
            )
            
            # 优化位姿参数
            for _ in range(self.optimization_steps):
                def closure():
                    optimizer.zero_grad()
                    
                    # 分离旋转和平移参数
                    rot_params = pose_params[:3]
                    trans_params = pose_params[3:]
                    
                    # 从李代数到旋转矩阵
                    R = exp_map(rot_params)
                    
                    # 投影3D点到2D
                    projected_points = project_points(points_3d_valid, K, R, trans_params)
                    
                    # 计算重投影误差
                    reproj_error = torch.mean(torch.norm(projected_points - points_2d_valid, dim=-1))
                    
                    # 反向传播
                    reproj_error.backward(retain_graph=True)
                    return reproj_error
                
                optimizer.step(closure)
            
            # 获取最终位姿，保留计算图连接
            rot_params = pose_params[:3]
            trans_params = pose_params[3:]
            
            R = exp_map(rot_params)
            t = trans_params.unsqueeze(-1)
            
            # 构建变换矩阵 [4,4]
            T = torch.eye(4, device=coord_map.device)
            T[:3, :3] = R
            T[:3, 3:] = t
            
            relative_pose[b, 0] = T
        
        return relative_pose



class PoseEstimationHead(nn.Module):
    """完整的位姿估计网络头部，整合特征提取与位姿估计模块"""
    def __init__(self):
        super(PoseEstimationHead, self).__init__()
        # 初始化子网络
        self.feature_extractor = FeatureExtractionNetwork()
        self.pose_estimator = RelativePoseEstimationNetwork()
        
        # 可学习的初始不确定性参数
        self.initial_uncertainty = nn.Parameter(torch.tensor([0.1]))

    def forward(self, images, camk, last_coord=None, last_uncertainty=None):
        """
        完整的位姿估计前向传播
        
        参数:
            images: 输入图像序列，shape为[B, V, 3, H, W]，其中V为视角数量
            camk: 相机内参矩阵，shape为[B, V, 4, 4]
            last_coord: 上一时刻的坐标映射（用于卡尔曼滤波），默认None
            last_uncertainty: 上一时刻的不确定性（用于卡尔曼滤波），默认None
            
        返回:
            包含位姿估计结果及中间特征的字典
        """
        # 1. 特征提取
        features = self.feature_extractor(images, camk)
        
        # 2. 处理初始不确定性（若未提供）
        if last_uncertainty is None:
            batch_size = images.shape[0]
            h, w = images.shape[-2], images.shape[-1]
            # 生成与坐标图同形状的初始不确定性
            last_uncertainty = torch.exp(self.initial_uncertainty) * torch.ones(
                batch_size, 1, h//16, w//16,  # 与SCoordNet输出的坐标图保持相同尺寸
                device=images.device
            )
        
        # 3. 位姿估计
        pose_result = self.pose_estimator(
            features, 
            camk, 
            last_coord=last_coord, 
            last_uncertainty=last_uncertainty
        )
        

        return pose_result
    
    def get_initial_state(self, image_shape):
        """生成初始状态（用于序列估计的第一帧）"""
        batch_size, _, h, w = image_shape
        device = next(self.parameters()).device
        
        # 初始坐标图（零初始化）
        initial_coord = torch.zeros(
            batch_size, 3, h//16, w//16,  # SCoordNet输出尺寸为输入的1/16
            device=device
        )
        
        # 初始不确定性
        initial_uncertainty = torch.exp(self.initial_uncertainty) * torch.ones(
            batch_size, 1, h//16, w//16,
            device=device
        )
        
        return initial_coord, initial_uncertainty


