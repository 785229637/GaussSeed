# Module API

GaussSeed contains no target-system patch or insertion code. The public API
accepts host state as tensors and returns backend-ready tensors.

## Proposal input

`GaussSeed.make_context` accepts:

- images: `[2, 3, H, W]`, RGB in `[0, 1]`;
- intrinsics: `[2, 3, 3]`, normalized by width and height;
- extrinsics: `[2, 4, 4]`, camera-to-world;
- near/far: positive bounds.

`GaussSeed.predict` returns canonical Gaussians plus refined poses and proposal
depth. It does not accept external depth.

## Transformation input

`GaussSeed.prepare_backend` accepts:

- `target_camera_to_world`: `[4, 4]`;
- `target_intrinsics`: `[3, 3]`;
- `map_rendered_depth`: `[H, W]`, required for expansion and omitted at init;
- `proposal_to_global_rotation`: optional `[3, 3]`;
- `proposal_to_global_translation`: optional `[3]`;
- `initializing`: whether the existing map is empty.

The map depth must be rendered internally by the active Gaussian map. Dataset,
sensor, or ground-truth depth must not be supplied.

## Transformation output

`PipelineResult` contains:

- `scale`: robust proposal-to-map scale;
- `aligned`: all canonical Gaussians after the similarity transform;
- `accepted`: the filtered and budgeted canonical subset;
- `projection`: target-view pixels, depth, and image mask;
- `filtering`: stage masks, errors, novelty, scores, and selected indices;
- `backend`: log-scales, quaternions, opacity logits, full SH tensors, and Adam
  state buffers.

No map is mutated by GaussSeed. If `requires_native_fallback` is true, the host
should leave its native initialization path unchanged.

## Direct functional API

The following functions are independently testable and exported from
`gaussseed`:

- `robust_depth_scale` — Equation (6);
- `apply_similarity` — Equation (7);
- `project_to_view` — Equation (8);
- `filter_candidates` — Equations (9)–(11);
- `novelty_from_map_depth` — rendered-coverage novelty;
- `adapt_backend_parameters` — covariance/opacity/SH conversion and state init.
