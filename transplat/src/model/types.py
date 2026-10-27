from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"] # 位置
    covariances: Float[Tensor, "batch gaussian dim dim"] # 协方差
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] #球谐函数
    opacities: Float[Tensor, "batch gaussian"] #混淆矩阵-不透明度
