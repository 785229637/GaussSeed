import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

class PoseEstimationHead(nn.Module):
    """
    带LSTM的位姿估计网络头，增强序列信息处理能力
    
    输入:
        features: CNN提取的图像特征，形状为[2, 128, 64, 64]，包含2张顺序图片
        dino_feature: 深度估计特征，形状为[2, 64, 144, 144]，包含2张顺序图片
    
    输出:
        rotations: 旋转四元数，形状为[1, 1, 4]
        translations: 平移向量，形状为[1, 1, 3]
    """
    def __init__(self, hidden_dim=256, lstm_hidden=512, lstm_layers=2, dropout_rate=0.3):
        super(PoseEstimationHead, self).__init__()
        
        # 处理CNN特征的卷积层
        self.cnn_feature_processor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 处理DINO特征的卷积层
        self.dino_feature_processor = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 融合特征的卷积层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 计算LSTM输入特征尺寸
        # 经过处理后特征图尺寸变为 8x8 (64 -> 32 -> 16 -> 8)
        self.lstm_input_size = hidden_dim * 8 * 8
        
        # LSTM层用于处理序列信息
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # LSTM输出处理
        self.lstm_output_processor = nn.Sequential(
            nn.Linear(lstm_hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        
        # 丰富的旋转预测头（四元数输出）
        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)  # 四元数输出
        )
        
        # 丰富的平移预测头
        self.translation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # 平移向量输出
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # 初始化LSTM权重
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        kaiming_normal_(param.data)
                    elif 'bias' in name:
                        param.data.zero_()
                        # 初始化遗忘门偏置为1
                        n = param.size(0)
                        param.data[n//4 : n//2].fill_(1.)
    
    def forward(self, features, dino_feature):
        # 处理CNN特征 [2, 128, 64, 64] -> [2, 64, 64, 64]
        cnn_feat = self.cnn_feature_processor(features)
        
        # 处理DINO特征并调整尺寸 [2, 64, 144, 144] -> [2, 64, 64, 64]
        dino_feat = self.dino_feature_processor(dino_feature)
        dino_feat = F.interpolate(
            dino_feat, 
            size=(64, 64), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 融合两种特征 [2, 128, 64, 64]
        fused_feat = torch.cat([cnn_feat, dino_feat], dim=1)
        
        # 进一步处理融合特征 [2, hidden_dim, 8, 8]
        fused_feat = self.fusion_conv(fused_feat)
        
        # 展平特征图，准备输入LSTM [2, hidden_dim*8*8]
        seq_len = fused_feat.size(0)
        lstm_input = fused_feat.view(1, seq_len, -1)  # [1, 2, lstm_input_size]
        
        # LSTM处理序列信息
        lstm_out, _ = self.lstm(lstm_input)  # [1, 2, lstm_hidden]
        
        # 取最后一个时间步的输出，代表序列的整体信息
        last_step_out = lstm_out[:, -1, :]  # [1, lstm_hidden]
        
        # 处理LSTM输出
        processed_feat = self.lstm_output_processor(last_step_out)  # [1, 256]
        
        # 预测旋转和平移
        rotations = self.rotation_head(processed_feat)  # [1, 4]
        translations = self.translation_head(processed_feat)  # [1, 3]
        
        # 对旋转向量进行归一化（四元数需要单位长度）
        rotations = F.normalize(rotations, p=2, dim=1)
        
        # 调整输出形状以匹配要求 [1, 1, 4] 和 [1, 1, 3]
        rotations = rotations.unsqueeze(0)  # [1, 1, 4]
        translations = translations.unsqueeze(0)  # [1, 1, 3]*
        
        return rotations, translations