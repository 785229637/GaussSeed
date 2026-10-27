# GaussSeed architecture

## 1. Proposal network

`GaussSeed.predict` consumes two RGB views, normalized intrinsics, camera-to-world
poses, and near/far bounds. The proposal network follows the paper:

1. `Dinov3Backbone` extracts structural and learned-depth features.
2. `BackboneMultiview` exchanges cross-view information and predicts a relative
   pose update.
3. `DepthPredictorTrans` builds a multi-view cost volume and predicts proposal
   depth, density, and raw Gaussian attributes.
4. `GaussianAdapter` unprojects centers and transports covariance and spherical
   harmonics into the canonical proposal frame.

The proposal output is `CanonicalGaussians(means, covariances, harmonics,
opacities)`. Inference receives no sensor or ground-truth depth.

## 2. Geometric alignment

All insertion-side geometry is centralized in `gaussseed.pipeline`.

- `robust_depth_scale` implements Equation (6): it takes the median of
  map/proposal depth ratios after invalid-value removal and quantile trimming.
- `apply_similarity` implements Equation (7): means scale linearly, while
  covariance transforms by `s² R Σ Rᵀ`.
- `project_to_view` implements Equation (8) under an explicit camera-to-world
  convention.

Scale alignment is bypassed at initialization because no existing map is
available. During expansion, map-rendered depth is mandatory.

## 3. Consistency and novelty

`filter_candidates` implements Equations (9)–(11) as named stages:

- finite and positive-semidefinite covariance check;
- minimum opacity;
- maximum principal scale relative to candidate depth;
- proposal-depth observation consistency;
- existing-map rendered-depth consistency;
- rendered-coverage novelty gate;
- deterministic initialization/keyframe budget.

A depth constraint is active only where its reference depth is positive and
finite. Missing map-rendered depth therefore does not fail map consistency; it
instead produces a high novelty score, representing an uncovered region.

## 4. Backend parameter adaptation

`adapt_backend_parameters` performs deterministic tensor-to-tensor conversion:

- covariance eigendecomposition → anisotropic log-scales and `wxyz` quaternion;
- opacity → inverse-sigmoid logit;
- SH → DC and remaining coefficients without discarding view-dependent terms;
- each trainable tensor → zero Adam first/second moments and step counter.

The result has no dependency on a particular SLAM repository. The host remains
responsible only for registering these tensors with its map container and
optimizer.

## 5. Failure semantics

Invalid scale calibration raises a descriptive error before mutation. A valid
pipeline run that rejects every candidate returns
`PipelineResult.requires_native_fallback == True`, matching the paper's fallback
to the host's native birth operator.
