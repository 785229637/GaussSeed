import torch
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from transplat.src.geometry.epipolar_lines import get_depth
from gaussseed_encoder.matching.conversions import depth_to_relative_disparity

from transplat.src.model.utils.cam_param_encoder import cam_param_encoder
from gaussseed_encoder.backbone.PoseEstimation_dino import PoseEstimationHead


def _det3x3(matrix: torch.Tensor) -> torch.Tensor:
    return (
        matrix[..., 0, 0] * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1] * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2] * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list

class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""
    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
        downscale_factor=8,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )
        
        self.cam_param_encoder = cam_param_encoder(in_channels=128, mid_channels=128, embed_dims=128)

        self.pose_head = PoseEstimationHead()
        self.last_coord=None
        self.last_uncertainty=None
        
        

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
            归一化处理
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        return features

    

    def quaternion_to_rotation_matrix(self, quat):
        """将四元数转换为旋转矩阵，并确保其正交性
        Args:
            quat: [B, 4] 格式为 (w, x, y, z)
        Returns:
            rot_mat: [B, 3, 3] 正交旋转矩阵，行列式为1
        """
        # 更稳定的四元数归一化，防止除零
        # quat_norm = torch.norm(quat, dim=1, keepdim=True)
        # quat = quat / (quat_norm + 1e-12)  # 增加微小值避免除零
        
        w , x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        batch_size = quat.size(0)
        
        # 初始旋转矩阵计算
        rot_mat = torch.empty((batch_size, 3, 3), device=quat.device)
        rot_mat[:, 0, 0] = 1 - 2*y*y - 2*z*z
        rot_mat[:, 0, 1] = 2*x*y - 2*z*w
        rot_mat[:, 0, 2] = 2*x*z + 2*y*w
        
        rot_mat[:, 1, 0] = 2*x*y + 2*z*w
        rot_mat[:, 1, 1] = 1 - 2*x*x - 2*z*z
        rot_mat[:, 1, 2] = 2*y*z - 2*x*w
        
        rot_mat[:, 2, 0] = 2*x*z - 2*y*w
        rot_mat[:, 2, 1] = 2*y*z + 2*x*w
        rot_mat[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        # # 第一次正交化：使用SVD分解
        # eps = 1e-6  # 小常数
        # rot_mat += torch.eye(rot_mat.shape[-1]).to("cuda:0") * eps
        # u, s, vh = torch.svd(rot_mat)
        # rot_mat = u @ vh  # 正交化
        
        # # 确保行列式为1（特殊正交群）
        # det = torch.det(rot_mat)
        # # 创建修正矩阵，对行列式为负的情况进行修正
        # correction = torch.eye(3, device=rot_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # correction[:, 2, 2] = det.sign()  # 只改变最后一个元素的符号
        
        # # 应用修正
        # rot_mat = rot_mat @ correction
        
        # # 第二次正交化，确保修正后仍然正交        
        # eps = 1e-6  # 小常数
        # rot_mat += torch.eye(rot_mat.shape[-1]).to("cuda:0")  * eps
        # u, s, vh = torch.svd(rot_mat)
        # rot_mat = u @ vh
        
        # # 调试：检查行列式是否接近1
        # final_det = torch.det(rot_mat)
        # if not torch.allclose(final_det, torch.ones_like(final_det), atol=1e-5):
        #     print(f"警告：旋转矩阵行列式为{final_det}，与1偏差较大")
        
        return rot_mat
    
    def construct_transformation_matrix(self, rotation, translation):
        """构建齐次变换矩阵
        Args:
            rotation: [B, 4] 四元数 (w, x, y, z)
            translation: [B, 3] 平移向量
        Returns:
            T: [B, 4, 4] 齐次变换矩阵
        """
        batch_size = rotation.size(0)
        device = rotation.device
        
        # 转换为旋转矩阵
        rot_mat = self.quaternion_to_rotation_matrix(rotation)  # [B, 3, 3]
        
        # 构建变换矩阵
        T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 4, 4]
        T[:, :3, :3] = rot_mat
        T[:, :3, 3] = translation
        
        return T

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        camk=None,
        c2w=None,
        dino_feature =None,
        depth_feature =None,
        pose_ture = False,
        dinov3_feature = None

    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high 提取特征 (低分辨率到高分辨率)
        features = dino_feature
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features_next = features[::-1]
        # rotations, translations, features_next = self.pose_head(images,camk)

        features_list = [[] for _ in range(2)]
        for feature in features_next:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=1, v=2)
            for idx in range(2):
                features_list[idx].append(feature[:, idx])

        cur_features_list = [x[0] for x in features_list]

        img2world = torch.matmul(c2w, camk)

        # 添加位姿增量预测和更新
        if img2world is not None and len(cur_features_list) >= 2:
            if pose_ture  == True:
                delta_T = torch.eye(4, device=img2world.device).unsqueeze(0).repeat(c2w.shape[0], 1, 1)  # [B, 4, 4]
                updated_pose = c2w[:, -1]
            else:

                # 预测增量位姿 (相对变换)
                # rotations, translations = self.pose_head(features[0],depth_feature)
                rotations, translations = self.pose_head(dinov3_feature.clone())
                rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
                # rotations = [rotations[0]]
                # # translations = [translations[0].squeeze(dim=2)]
                # translations = [translations[0]]

                
                # # 使用最后一次迭代结果
                # delta_rotation = rotations[-1]
                # delta_translation = translations[-1]

                
                # 3. 构建增量变换矩阵
                delta_T = self.construct_transformation_matrix(
                    rotations,  # [B, 4]
                    translations  # [B, 3]
                )  # [B, 4, 4]

                # output = self.pose_head(images,camk,self.last_coord,self.last_uncertainty)
                # self.last_coord=output['kf_coord']
                # self.last_uncertainty=output['kf_uncertainty']
                # delta_T = output['relative_pose'][0]
                
                # 4. 更新最后一个位姿
                # 获取当前最后一个位姿 (世界坐标系到相机坐标系)
                
                # with torch.no_grad():
                # 获取参考帧位姿（倒数第二帧的c2w）
                ref_pose = c2w[:, -2]  # [B, 4, 4]
                
                # 应用增量变换: new_pose = ref_pose * delta_T
                updated_pose = torch.matmul(ref_pose, delta_T)

                

            # updated_pose = check_and_fix_SE3(updated_pose)

            # 更新外参矩阵中的最后一帧
            
                # context["extrinsics"][:, -1] = updated_pose  # 同步更新context中的外参

            # 4. 重新计算img2world（如果需要）
            camk_inv = torch.inverse(camk)
            img2world_1 = torch.matmul(updated_pose, camk_inv[:,1].squeeze(1))

        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        

        # add cam param to features
        feature_list_with_cam = [] 
        # for v_id, cur_features in enumerate(cur_features_list):
        feature_list_with_cam.append(self.cam_param_encoder(cur_features_list[0], img2world[:,0]))
        feature_list_with_cam.append(self.cam_param_encoder(cur_features_list[1], img2world_1))
        cur_features_list = feature_list_with_cam

        # add position to features
        cur_features_list = feature_add_position_list(
            cur_features_list, attn_splits, self.feature_channels)

        # Transformer
        cur_features_list = self.transformer(
            cur_features_list, attn_num_splits=attn_splits)

        features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if return_cnn_features:
            out_lists = [features, cnn_features, updated_pose,delta_T]
        else:
            out_lists = [features, None]

        return out_lists

def to_SE3(T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    把 [B,4,4] 的齐次矩阵就地修正为 SE(3) 矩阵。
    仅处理 batch 中第 0 个矩阵（B=1）。
    """
    # --- 1. 最后一行必须是 [0,0,0,1] ---
    T[0, 3, :] = torch.tensor([0, 0, 0, 1], dtype=T.dtype, device=T.device)

    # --- 2. 旋转部分正交化 ---
    R = T[0, :3, :3]                       # [3,3]
    eps = 1e-8
    R  += torch.eye(R.shape[-1], device=R.device) * eps
    U, _, Vh = torch.linalg.svd(R, full_matrices=False)
    R_new = U @ Vh                         # 正交化
    # 保证 det = +1
    if _det3x3(R_new) < 0:
        Vh[2, :] *= -1
        R_new = U @ Vh
    T[0, :3, :3] = R_new
    return T

def check_and_fix_SE3(updated_pose: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    updated_pose: 已加载到 GPU/CPU 的 tensor，shape=[1,4,4]
    返回：必定是 SE(3) 矩阵的 tensor（就地修改或原样返回）。
    """
    R = updated_pose[0, :3, :3]
    t = updated_pose[0, :3, 3]
    last_row = updated_pose[0, 3]

    # 检查旋转是否正交、行列式是否为 +1
    ortho = torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=eps)
    det_ok = torch.allclose(_det3x3(R), torch.tensor(1.0, device=R.device), atol=eps)
    last_ok = torch.allclose(last_row, torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device, dtype=R.dtype), atol=eps)

    if ortho and det_ok and last_ok:
        return updated_pose          # 已经是 SE(3)
    else:
        return to_SE3(updated_pose)  # 就地修正
