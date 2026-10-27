"""Backend-agnostic alignment and filtering for the GaussSeed interface.

The functions in this module implement Equations (6)--(11) of the paper.  They
operate on canonical Gaussian tensors and deliberately contain no target-SLAM
dependencies, so every integration uses the same scale, consistency, novelty,
and budget semantics.
"""

from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the staged acceptance rule in Equation (11)."""

    min_opacity: float = 0.05
    max_scale_depth_fraction: float = 0.08
    max_observation_error: float = 0.45
    max_map_error: float = 0.45
    min_novelty: float = 0.05
    budget: int = 0
    epsilon: float = 1e-6


@dataclass
class FilterResult:
    """Selected indices and diagnostics for one filtering invocation."""

    indices: Tensor
    accepted: Tensor
    scales: Tensor
    observation_error: Tensor
    map_error: Tensor
    novelty: Tensor
    score: Tensor
    stage_masks: dict[str, Tensor]


@dataclass
class CanonicalGaussians:
    """Backend-neutral representation used by Equations (1)--(2)."""

    means: Tensor
    covariances: Tensor
    harmonics: Tensor
    opacities: Tensor

    @classmethod
    def from_prediction(cls, gaussians: object, batch_index: int = 0) -> "CanonicalGaussians":
        batched_dimensions = {
            "means": 3,
            "covariances": 4,
            "harmonics": 4,
            "opacities": 2,
        }
        tensors = {
            name: getattr(gaussians, name)
            for name in ("means", "covariances", "harmonics", "opacities")
        }
        selected = {}
        for name, value in tensors.items():
            if not torch.is_tensor(value):
                raise TypeError(f"gaussians.{name} must be a torch.Tensor")
            expected_batched = batched_dimensions[name]
            if value.ndim == expected_batched:
                selected[name] = value[batch_index]
            elif value.ndim == expected_batched - 1:
                selected[name] = value
            else:
                raise ValueError(f"gaussians.{name} has unexpected shape {tuple(value.shape)}")
        return cls(**selected)

    def validate(self) -> None:
        count = self.means.shape[0]
        if self.means.shape != (count, 3):
            raise ValueError("means must have shape [N, 3]")
        if self.covariances.shape != (count, 3, 3):
            raise ValueError("covariances must have shape [N, 3, 3]")
        if self.harmonics.ndim != 3 or self.harmonics.shape[:2] != (count, 3):
            raise ValueError("harmonics must have shape [N, 3, D_sh]")
        if self.opacities.numel() != count:
            raise ValueError("opacities must contain one value per Gaussian")

    def index_select(self, indices: Tensor) -> "CanonicalGaussians":
        return CanonicalGaussians(
            means=self.means[indices],
            covariances=self.covariances[indices],
            harmonics=self.harmonics[indices],
            opacities=self.opacities.reshape(-1)[indices],
        )


@dataclass
class BackendParameters:
    """Common native 3DGS tensors produced by the adaptation operator."""

    means: Tensor
    log_scales: Tensor
    rotations_wxyz: Tensor
    opacity_logits: Tensor
    features_dc: Tensor
    features_rest: Tensor
    optimizer_state: dict[str, dict[str, Tensor]]

    def trainable_tensors(self) -> dict[str, Tensor]:
        return {
            "means": self.means,
            "log_scales": self.log_scales,
            "rotations_wxyz": self.rotations_wxyz,
            "opacity_logits": self.opacity_logits,
            "features_dc": self.features_dc,
            "features_rest": self.features_rest,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the complete alignment-to-adaptation interface."""

    filtering: FilterConfig = field(default_factory=FilterConfig)
    init_budget: int = 32768
    keyframe_budget: int = 8192
    scale_min_samples: int = 32
    scale_trim_quantiles: tuple[float, float] = (0.1, 0.9)
    scale_bounds: tuple[float, float] = (1e-3, 100.0)


@dataclass
class Projection:
    pixels: Tensor
    depth: Tensor
    in_image: Tensor


@dataclass
class PipelineResult:
    """Output of the complete GaussSeed insertion-side module."""

    scale: Tensor
    aligned: CanonicalGaussians
    accepted: CanonicalGaussians
    backend: BackendParameters
    projection: Projection
    filtering: FilterResult

    @property
    def requires_native_fallback(self) -> bool:
        """The host should retain its native birth operator when nothing survives."""

        return self.filtering.indices.numel() == 0


def _vector(value: Tensor, count: int, name: str) -> Tensor:
    value = torch.as_tensor(value)
    if value.numel() != count:
        raise ValueError(f"{name} must contain {count} values, got {value.numel()}")
    return value.reshape(count)


def robust_depth_scale(
    proposal_depth: Tensor,
    map_depth: Tensor,
    *,
    valid: Tensor | None = None,
    trim_quantiles: tuple[float, float] = (0.1, 0.9),
    min_samples: int = 32,
    scale_bounds: tuple[float, float] = (1e-3, 100.0),
) -> Tensor:
    """Estimate the median map/proposal depth ratio from co-visible pixels.

    Invalid depths never contribute.  Extreme ratio quantiles are removed
    before taking the median, matching the robust calibration in Equation (6).
    """

    proposal = torch.as_tensor(proposal_depth).reshape(-1)
    mapped = torch.as_tensor(
        map_depth, device=proposal.device, dtype=proposal.dtype
    ).reshape(-1)
    if proposal.numel() != mapped.numel():
        raise ValueError("proposal_depth and map_depth must have the same number of values")

    mask = (
        torch.isfinite(proposal)
        & torch.isfinite(mapped)
        & (proposal > 0)
        & (mapped > 0)
    )
    if valid is not None:
        mask &= _vector(valid, proposal.numel(), "valid").to(
            device=proposal.device, dtype=torch.bool
        )

    ratios = mapped[mask] / proposal[mask]
    ratios = ratios[torch.isfinite(ratios) & (ratios > 0)]
    if ratios.numel() < min_samples:
        raise ValueError(
            f"Scale calibration needs at least {min_samples} valid ratios; got {ratios.numel()}"
        )

    low_q, high_q = trim_quantiles
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("trim_quantiles must satisfy 0 <= low < high <= 1")
    bounds = torch.quantile(
        ratios,
        torch.tensor([low_q, high_q], device=ratios.device, dtype=ratios.dtype),
    )
    trimmed = ratios[(ratios >= bounds[0]) & (ratios <= bounds[1])]
    if trimmed.numel() == 0:
        raise ValueError("Scale calibration rejected every ratio during quantile trimming")
    return torch.median(trimmed).clamp(*scale_bounds)


def apply_similarity(
    means: Tensor,
    covariances: Tensor,
    scale: float | Tensor,
    *,
    rotation: Tensor | None = None,
    translation: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply Equation (7) to Gaussian means and covariances."""

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N, 3]")
    if covariances.shape != (means.shape[0], 3, 3):
        raise ValueError("covariances must have shape [N, 3, 3]")

    device, dtype = means.device, means.dtype
    scale_tensor = torch.as_tensor(scale, device=device, dtype=dtype)
    if scale_tensor.numel() != 1 or not torch.isfinite(scale_tensor) or scale_tensor <= 0:
        raise ValueError("scale must be one positive finite scalar")
    if rotation is None:
        rotation = torch.eye(3, device=device, dtype=dtype)
    else:
        rotation = torch.as_tensor(rotation, device=device, dtype=dtype)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    if translation is None:
        translation = torch.zeros(3, device=device, dtype=dtype)
    else:
        translation = torch.as_tensor(translation, device=device, dtype=dtype)
    if translation.shape != (3,):
        raise ValueError("translation must have shape [3]")

    aligned_means = (means * scale_tensor) @ rotation.transpose(0, 1) + translation
    aligned_covariances = (
        scale_tensor.square()
        * rotation.unsqueeze(0)
        @ covariances
        @ rotation.transpose(0, 1).unsqueeze(0)
    )
    return aligned_means, aligned_covariances


def resize_depth_map(depth: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize a depth field without inventing values across invalid pixels."""

    depth = torch.as_tensor(depth)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W] or [1, H, W]")
    if tuple(depth.shape) == tuple(size):
        return depth
    return F.interpolate(depth[None, None], size=size, mode="nearest")[0, 0]


def project_to_view(
    means: Tensor,
    camera_to_world: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    intrinsics_normalized: bool = True,
) -> Projection:
    """Project global Gaussian centers into one host-SLAM camera view."""

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N, 3]")
    camera_to_world = torch.as_tensor(camera_to_world, device=means.device, dtype=means.dtype)
    intrinsics = torch.as_tensor(intrinsics, device=means.device, dtype=means.dtype).clone()
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4, 4]")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3, 3]")

    height, width = image_size
    if intrinsics_normalized:
        intrinsics[0] *= float(width)
        intrinsics[1] *= float(height)
    world_to_camera = torch.linalg.inv(camera_to_world)
    camera_points = (
        means @ world_to_camera[:3, :3].transpose(0, 1)
        + world_to_camera[:3, 3]
    )
    depth = camera_points[:, 2]
    homogeneous_pixels = camera_points @ intrinsics.transpose(0, 1)
    pixels = homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2:3].clamp_min(1e-8)
    in_image = (
        torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    return Projection(pixels=pixels, depth=depth, in_image=in_image)


def sample_depth_map(depth: Tensor, projection: Projection) -> Tensor:
    """Sample a depth field at projected centers; invalid references stay NaN."""

    depth = torch.as_tensor(
        depth, device=projection.depth.device, dtype=projection.depth.dtype
    )
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W] or [1, H, W]")
    sampled = torch.full_like(projection.depth, float("nan"))
    if bool(torch.any(projection.in_image)):
        pixels = projection.pixels[projection.in_image].round().long()
        sampled[projection.in_image] = depth[pixels[:, 1], pixels[:, 0]]
    return sampled


def relative_depth_error(
    candidate_depth: Tensor,
    reference_depth: Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return Equation (9)/(10) errors and their active-reference mask."""

    candidate = torch.as_tensor(candidate_depth).reshape(-1)
    reference = torch.as_tensor(
        reference_depth, device=candidate.device, dtype=candidate.dtype
    ).reshape(-1)
    if candidate.numel() != reference.numel():
        raise ValueError("candidate_depth and reference_depth must have equal length")
    active = (
        torch.isfinite(candidate)
        & torch.isfinite(reference)
        & (candidate > 0)
        & (reference > 0)
    )
    error = torch.zeros_like(candidate)
    error[active] = (
        (candidate[active] - reference[active]).abs()
        / reference[active].clamp_min(epsilon)
    )
    return error, active


def novelty_from_map_depth(
    candidate_depth: Tensor,
    map_depth: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Estimate map novelty from rendered coverage and relative depth residual.

    Pixels without valid rendered map depth are fully novel.  At covered pixels,
    the normalized proposal-to-map discrepancy provides a continuous novelty
    score in ``[0, 1]``.  This separates map consistency from redundancy: an
    unavailable reference does not fail consistency, but it does signal novelty.
    """

    candidate = torch.as_tensor(candidate_depth).reshape(-1)
    mapped = torch.as_tensor(map_depth, device=candidate.device, dtype=candidate.dtype).reshape(-1)
    if candidate.numel() != mapped.numel():
        raise ValueError("candidate_depth and map_depth must have equal length")
    valid_candidate = torch.isfinite(candidate) & (candidate > 0)
    covered = valid_candidate & torch.isfinite(mapped) & (mapped > 0)
    novelty = torch.ones_like(candidate)
    novelty[~valid_candidate] = 0
    novelty[covered] = (
        (candidate[covered] - mapped[covered]).abs()
        / mapped[covered].clamp_min(epsilon)
    ).clamp(0, 1)
    return novelty


def filter_candidates(
    means: Tensor,
    covariances: Tensor,
    opacities: Tensor,
    candidate_depth: Tensor,
    *,
    proposal_depth: Tensor | None = None,
    map_depth: Tensor | None = None,
    novelty: Tensor | None = None,
    config: FilterConfig | None = None,
    initializing: bool = False,
) -> FilterResult:
    """Apply the staged finite/scale/consistency/novelty/budget filter."""

    config = config or FilterConfig()
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N, 3]")
    count = means.shape[0]
    if covariances.shape != (count, 3, 3):
        raise ValueError("covariances must have shape [N, 3, 3]")
    opacities = _vector(opacities, count, "opacities").to(means)
    candidate_depth = _vector(candidate_depth, count, "candidate_depth").to(means)

    symmetric_covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    finite_covariance = torch.isfinite(symmetric_covariances).flatten(1).all(dim=-1)
    safe_covariances = symmetric_covariances.clone()
    if not bool(torch.all(finite_covariance)):
        safe_covariances[~finite_covariance] = torch.eye(
            3, device=means.device, dtype=means.dtype
        )
    eigenvalues = torch.linalg.eigvalsh(safe_covariances)
    positive_semidefinite = (eigenvalues >= -config.epsilon).all(dim=-1)
    scales = torch.sqrt(eigenvalues.clamp_min(config.epsilon))

    finite = (
        torch.isfinite(means).all(dim=-1)
        & finite_covariance
        & torch.isfinite(opacities)
        & torch.isfinite(candidate_depth)
        & (candidate_depth > 0)
        & positive_semidefinite
    )
    opacity_ok = opacities >= config.min_opacity
    if config.max_scale_depth_fraction > 0:
        scale_ok = scales.max(dim=-1).values <= (
            candidate_depth * config.max_scale_depth_fraction
        )
    else:
        scale_ok = torch.ones(count, device=means.device, dtype=torch.bool)

    observation_error = torch.zeros(count, device=means.device, dtype=means.dtype)
    observation_ok = torch.ones(count, device=means.device, dtype=torch.bool)
    observation_active = torch.zeros_like(observation_ok)
    if proposal_depth is not None:
        proposal_depth = _vector(proposal_depth, count, "proposal_depth").to(means)
        observation_error, observation_active = relative_depth_error(
            candidate_depth, proposal_depth, epsilon=config.epsilon
        )
        observation_ok[observation_active] = (
            observation_error[observation_active] <= config.max_observation_error
        )

    map_error = torch.zeros(count, device=means.device, dtype=means.dtype)
    map_ok = torch.ones(count, device=means.device, dtype=torch.bool)
    map_active = torch.zeros_like(map_ok)
    mapped = None
    if map_depth is not None and not initializing:
        mapped = _vector(map_depth, count, "map_depth").to(means)
        map_error, map_active = relative_depth_error(
            candidate_depth, mapped, epsilon=config.epsilon
        )
        map_ok[map_active] = map_error[map_active] <= config.max_map_error

    novelty_score = torch.ones(count, device=means.device, dtype=means.dtype)
    novelty_ok = torch.ones(count, device=means.device, dtype=torch.bool)
    novelty_active = torch.zeros_like(novelty_ok)
    if not initializing:
        if novelty is not None:
            novelty_score = _vector(novelty, count, "novelty").to(means)
            novelty_active = torch.isfinite(novelty_score)
        elif mapped is not None:
            novelty_score = novelty_from_map_depth(
                candidate_depth, mapped, epsilon=config.epsilon
            )
            novelty_active = torch.isfinite(candidate_depth) & (candidate_depth > 0)
        novelty_ok[novelty_active] = (
            novelty_score[novelty_active] >= config.min_novelty
        )

    stage_masks = {
        "finite": finite,
        "opacity": opacity_ok,
        "relative_scale": scale_ok,
        "observation": observation_ok,
        "map": map_ok,
        "novelty": novelty_ok,
    }
    accepted = finite & opacity_ok & scale_ok & observation_ok & map_ok & novelty_ok

    score = opacities.clamp_min(0) * (0.5 + 0.5 * novelty_score.clamp(0, 1))
    score = score / (1.0 + observation_error + map_error)
    score[~accepted] = -torch.inf
    indices = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
    if config.budget > 0 and indices.numel() > config.budget:
        order = torch.argsort(score[indices], descending=True, stable=True)
        indices = indices[order[: config.budget]]
        budget_mask = torch.zeros_like(accepted)
        budget_mask[indices] = True
        accepted = accepted & budget_mask
        stage_masks["budget"] = budget_mask
    else:
        stage_masks["budget"] = accepted.clone()

    return FilterResult(
        indices=indices,
        accepted=accepted,
        scales=scales,
        observation_error=observation_error,
        map_error=map_error,
        novelty=novelty_score,
        score=score,
        stage_masks=stage_masks,
    )


def rotation_matrix_to_quaternion_wxyz(rotation: Tensor) -> Tensor:
    """Convert proper rotation matrices to normalized ``[w, x, y, z]`` quaternions."""

    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape [N, 3, 3]")
    r00, r01, r02 = rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2]
    r10, r11, r12 = rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2]
    r20, r21, r22 = rotation[:, 2, 0], rotation[:, 2, 1], rotation[:, 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1 + r00 + r11 + r22, min=0))
    qx = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1 + r00 - r11 - r22, min=0)), r21 - r12
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1 - r00 + r11 - r22, min=0)), r02 - r20
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1 - r00 - r11 + r22, min=0)), r10 - r01
    )
    return F.normalize(torch.stack((qw, qx, qy, qz), dim=-1), dim=-1)


def initialize_optimizer_state(parameters: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """Create zero Adam moments without depending on a particular SLAM optimizer."""

    return {
        name: {
            "step": torch.zeros((), device=value.device, dtype=torch.float32),
            "exp_avg": torch.zeros_like(value),
            "exp_avg_sq": torch.zeros_like(value),
        }
        for name, value in parameters.items()
    }


def adapt_backend_parameters(
    gaussians: CanonicalGaussians,
    *,
    opacity_epsilon: float = 1e-4,
) -> BackendParameters:
    """Convert canonical tensors to the common native 3DGS parameterization.

    Covariances are eigendecomposed into anisotropic log-scales and rotations;
    SH coefficients are preserved rather than reduced to RGB point colors.
    """

    gaussians.validate()
    covariance = 0.5 * (gaussians.covariances + gaussians.covariances.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if bool(torch.any(eigenvalues < -1e-6)):
        raise ValueError("Cannot adapt covariance matrices with negative eigenvalues")
    eigenvectors = eigenvectors.clone()
    negative_determinant = torch.det(eigenvectors) < 0
    eigenvectors[negative_determinant, :, 2] *= -1
    scales = torch.sqrt(eigenvalues.clamp_min(1e-12))
    opacities = gaussians.opacities.reshape(-1).clamp(
        opacity_epsilon, 1 - opacity_epsilon
    )
    features_dc = gaussians.harmonics[:, :, :1].transpose(1, 2).contiguous()
    features_rest = gaussians.harmonics[:, :, 1:].transpose(1, 2).contiguous()
    tensors = {
        "means": gaussians.means,
        "log_scales": torch.log(scales),
        "rotations_wxyz": rotation_matrix_to_quaternion_wxyz(eigenvectors),
        "opacity_logits": torch.logit(opacities).unsqueeze(-1),
        "features_dc": features_dc,
        "features_rest": features_rest,
    }
    return BackendParameters(
        **tensors,
        optimizer_state=initialize_optimizer_state(tensors),
    )


class GaussSeedPipeline:
    """Paper-aligned, backend-independent insertion-side GaussSeed module."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def process(
        self,
        gaussians: CanonicalGaussians | object,
        *,
        proposal_depth: Tensor,
        target_camera_to_world: Tensor,
        target_intrinsics: Tensor,
        map_rendered_depth: Tensor | None = None,
        proposal_to_global_rotation: Tensor | None = None,
        proposal_to_global_translation: Tensor | None = None,
        initializing: bool = False,
        intrinsics_normalized: bool = True,
    ) -> PipelineResult:
        """Run Equations (6)--(11), adaptation, and optimizer-state creation.

        During map expansion ``map_rendered_depth`` is mandatory because it is
        both the scale reference and the source of map consistency/novelty.  At
        initialization there is no existing map, so scale and map-side gates are
        intentionally bypassed as described in the paper.
        """

        if not isinstance(gaussians, CanonicalGaussians):
            gaussians = CanonicalGaussians.from_prediction(gaussians)
        gaussians.validate()
        proposal_depth = torch.as_tensor(
            proposal_depth, device=gaussians.means.device, dtype=gaussians.means.dtype
        )
        if proposal_depth.ndim == 3 and proposal_depth.shape[0] == 1:
            proposal_depth = proposal_depth[0]
        if proposal_depth.ndim != 2:
            raise ValueError("proposal_depth must have shape [H, W] or [1, H, W]")
        if not initializing and map_rendered_depth is None:
            raise ValueError("map_rendered_depth is required during map expansion")

        if map_rendered_depth is None:
            image_size = tuple(proposal_depth.shape)
            resized_proposal = proposal_depth
            resized_map = None
            scale = torch.ones((), device=gaussians.means.device, dtype=gaussians.means.dtype)
        else:
            resized_map = torch.as_tensor(
                map_rendered_depth,
                device=gaussians.means.device,
                dtype=gaussians.means.dtype,
            )
            if resized_map.ndim == 3 and resized_map.shape[0] == 1:
                resized_map = resized_map[0]
            if resized_map.ndim != 2:
                raise ValueError("map_rendered_depth must have shape [H, W] or [1, H, W]")
            image_size = tuple(resized_map.shape)
            resized_proposal = resize_depth_map(proposal_depth, image_size)
            if initializing:
                scale = torch.ones(
                    (), device=gaussians.means.device, dtype=gaussians.means.dtype
                )
            else:
                scale = robust_depth_scale(
                    resized_proposal,
                    resized_map,
                    trim_quantiles=self.config.scale_trim_quantiles,
                    min_samples=self.config.scale_min_samples,
                    scale_bounds=self.config.scale_bounds,
                )

        aligned_means, aligned_covariances = apply_similarity(
            gaussians.means,
            gaussians.covariances,
            scale,
            rotation=proposal_to_global_rotation,
            translation=proposal_to_global_translation,
        )
        aligned = CanonicalGaussians(
            means=aligned_means,
            covariances=aligned_covariances,
            harmonics=gaussians.harmonics,
            opacities=gaussians.opacities.reshape(-1),
        )
        projection = project_to_view(
            aligned.means,
            target_camera_to_world,
            target_intrinsics,
            image_size,
            intrinsics_normalized=intrinsics_normalized,
        )
        candidate_depth = projection.depth.clone()
        candidate_depth[~projection.in_image] = float("nan")
        sampled_proposal = sample_depth_map(resized_proposal, projection) * scale
        sampled_map = None if resized_map is None else sample_depth_map(resized_map, projection)
        budget = self.config.init_budget if initializing else self.config.keyframe_budget
        filtering = filter_candidates(
            aligned.means,
            aligned.covariances,
            aligned.opacities,
            candidate_depth,
            proposal_depth=sampled_proposal,
            map_depth=sampled_map,
            config=replace(self.config.filtering, budget=budget),
            initializing=initializing,
        )
        accepted = aligned.index_select(filtering.indices)
        backend = adapt_backend_parameters(accepted)
        return PipelineResult(
            scale=scale,
            aligned=aligned,
            accepted=accepted,
            backend=backend,
            projection=projection,
            filtering=filtering,
        )
