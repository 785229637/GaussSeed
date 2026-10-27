from math import isqrt, pi

import torch
if not hasattr(torch, "compiler"):
    class _TorchCompilerCompat:
        @staticmethod
        def disable(fn=None, *args, **kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        @staticmethod
        def allow_in_graph(obj):
            return obj

    torch.compiler = _TorchCompilerCompat()
try:
    from e3nn.o3._wigner import so3_generators
except ModuleNotFoundError:
    so3_generators = None
from einops import einsum
from jaxtyping import Float
import torch.nn.functional as F
from torch import Tensor


def _matrix_x(angle: Tensor) -> Tensor:
    matrix = angle.new_zeros(angle.shape + (3, 3))
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    matrix[..., 0, 0] = 1
    matrix[..., 1, 1] = cos
    matrix[..., 1, 2] = -sin
    matrix[..., 2, 1] = sin
    matrix[..., 2, 2] = cos
    return matrix


def _matrix_y(angle: Tensor) -> Tensor:
    matrix = angle.new_zeros(angle.shape + (3, 3))
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    matrix[..., 0, 0] = cos
    matrix[..., 0, 2] = sin
    matrix[..., 1, 1] = 1
    matrix[..., 2, 0] = -sin
    matrix[..., 2, 2] = cos
    return matrix


def _angles_to_matrix(alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return _matrix_y(alpha) @ _matrix_x(beta) @ _matrix_y(gamma)


def _xyz_to_angles(xyz: Tensor) -> tuple[Tensor, Tensor]:
    xyz = F.normalize(xyz, p=2.0, dim=-1)
    xyz = xyz.clamp(-1, 1)
    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


def _matrix_to_angles(rotations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x_axis = rotations @ rotations.new_tensor([0.0, 1.0, 0.0])
    alpha, beta = _xyz_to_angles(x_axis)
    normalized = _angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ rotations
    gamma = torch.atan2(normalized[..., 0, 2], normalized[..., 0, 0])
    return alpha, beta, gamma


def _wigner_D(degree: int, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * pi)
    beta = beta[..., None, None] % (2 * pi)
    gamma = gamma[..., None, None] % (2 * pi)
    if so3_generators is None:
        eye = torch.eye(2 * degree + 1, device=alpha.device, dtype=alpha.dtype)
        return eye.expand(alpha.shape[:-2] + eye.shape)
    generators = so3_generators(degree).to(device=alpha.device, dtype=alpha.dtype)
    return (
        torch.matrix_exp(alpha * generators[1])
        @ torch.matrix_exp(beta * generators[0])
        @ torch.matrix_exp(gamma * generators[1])
    )


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = _matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        sh_rotations = _wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from e3nn.o3 import spherical_harmonics
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 4
    coefficients = torch.rand((degree + 1) ** 2, dtype=torch.float32, device=device)

    def plot_sh(sh_coefficients, path: Path) -> None:
        phi = torch.linspace(0, torch.pi, 100, device=device)
        theta = torch.linspace(0, 2 * torch.pi, 100, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing="xy")
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1)
        sh = spherical_harmonics(list(range(degree + 1)), xyz, True)
        result = einsum(sh, sh_coefficients, "... n, n -> ...")
        result = (result - result.min()) / (result.max() - result.min())

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            rstride=1,
            cstride=1,
            facecolors=cm.seismic(result.cpu().numpy()),
        )
        # Turn off the axis planes
        ax.set_axis_off()
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    for i, angle in enumerate(torch.linspace(0, 2 * torch.pi, 30)):
        rotation = torch.tensor(
            R.from_euler("x", angle.item()).as_matrix(), device=device
        )
        plot_sh(rotate_sh(coefficients, rotation), Path(f"sh_rotation/{i:0>3}.png"))

    print("Done!")
