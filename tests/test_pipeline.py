import unittest

import torch

from gaussseed.pipeline import (
    CanonicalGaussians,
    FilterConfig,
    GaussSeedPipeline,
    PipelineConfig,
    adapt_backend_parameters,
    apply_similarity,
    filter_candidates,
    robust_depth_scale,
)


def _covariances(count: int, scale: float = 0.02) -> torch.Tensor:
    return torch.eye(3).repeat(count, 1, 1) * scale**2


class PipelineTests(unittest.TestCase):
    def test_robust_depth_scale_trims_outlier(self) -> None:
        proposal = torch.ones(100)
        mapped = torch.full((100,), 2.0)
        mapped[-1] = 1000.0
        scale = robust_depth_scale(proposal, mapped, min_samples=32)
        self.assertTrue(torch.isclose(scale, torch.tensor(2.0)))

    def test_similarity_scales_covariance_quadratically(self) -> None:
        means = torch.tensor([[1.0, 0.0, 0.0]])
        covariance = torch.eye(3).unsqueeze(0)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        aligned_means, aligned_covariance = apply_similarity(
            means,
            covariance,
            2.0,
            rotation=rotation,
            translation=torch.tensor([1.0, 0.0, 0.0]),
        )
        self.assertTrue(torch.allclose(aligned_means, torch.tensor([[1.0, 2.0, 0.0]])))
        self.assertTrue(
            torch.allclose(aligned_covariance, 4.0 * torch.eye(3).unsqueeze(0))
        )

    def test_filter_bypasses_missing_map_depth_and_prunes_duplicate(self) -> None:
        means = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        result = filter_candidates(
            means,
            _covariances(2),
            torch.tensor([0.8, 0.8]),
            torch.tensor([1.0, 1.0]),
            proposal_depth=torch.tensor([1.0, 1.0]),
            map_depth=torch.tensor([float("nan"), 1.0]),
            config=FilterConfig(min_novelty=0.05),
        )
        self.assertEqual(result.indices.tolist(), [0])
        self.assertEqual(result.novelty.tolist(), [1.0, 0.0])

    def test_filter_enforces_observation_map_and_budget_stages(self) -> None:
        result = filter_candidates(
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
            ),
            _covariances(3),
            torch.tensor([0.6, 0.9, 0.8]),
            torch.tensor([1.0, 1.0, 1.0]),
            proposal_depth=torch.tensor([1.0, 2.0, 1.0]),
            map_depth=torch.tensor([float("nan"), float("nan"), float("nan")]),
            config=FilterConfig(max_observation_error=0.2, budget=1),
        )
        self.assertEqual(result.indices.tolist(), [2])
        self.assertFalse(bool(result.stage_masks["observation"][1]))

    def test_backend_adaptation_preserves_anisotropy_and_sh(self) -> None:
        canonical = CanonicalGaussians(
            means=torch.zeros(1, 3),
            covariances=torch.diag(torch.tensor([0.01, 0.04, 0.09])).unsqueeze(0),
            harmonics=torch.arange(15, dtype=torch.float32).reshape(1, 3, 5),
            opacities=torch.tensor([0.75]),
        )
        backend = adapt_backend_parameters(canonical)
        self.assertTrue(
            torch.allclose(backend.log_scales.exp(), torch.tensor([[0.1, 0.2, 0.3]]))
        )
        self.assertEqual(backend.features_dc.shape, (1, 1, 3))
        self.assertEqual(backend.features_rest.shape, (1, 4, 3))
        self.assertTrue(
            torch.allclose(backend.features_dc[:, 0], canonical.harmonics[:, :, 0])
        )
        self.assertEqual(set(backend.optimizer_state), set(backend.trainable_tensors()))

    def test_complete_pipeline_aligns_filters_and_adapts(self) -> None:
        canonical = CanonicalGaussians(
            means=torch.tensor([[0.0, 0.0, 1.0]]),
            covariances=_covariances(1),
            harmonics=torch.ones(1, 3, 4),
            opacities=torch.tensor([0.8]),
        )
        proposal_depth = torch.ones(4, 4)
        map_depth = torch.full((4, 4), 2.0)
        map_depth[2, 2] = float("nan")
        pipeline = GaussSeedPipeline(
            PipelineConfig(
                filtering=FilterConfig(min_novelty=0.05),
                init_budget=8,
                keyframe_budget=8,
                scale_min_samples=4,
            )
        )
        result = pipeline.process(
            canonical,
            proposal_depth=proposal_depth,
            map_rendered_depth=map_depth,
            target_camera_to_world=torch.eye(4),
            target_intrinsics=torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
            ),
        )
        self.assertTrue(torch.isclose(result.scale, torch.tensor(2.0)))
        self.assertTrue(
            torch.allclose(result.aligned.means, torch.tensor([[0.0, 0.0, 2.0]]))
        )
        self.assertEqual(result.filtering.indices.tolist(), [0])
        self.assertFalse(result.requires_native_fallback)


if __name__ == "__main__":
    unittest.main()
