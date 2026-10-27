# Third-party components

This repository contains adapted TranSplat code under its MIT license and a
modified differentiable Gaussian rasterizer whose license is preserved in
`third_party/diff-gaussian-rasterization-modified/LICENSE.md`.

The named copyright notice in the repository-level `LICENSE` is preserved from
the upstream TranSplat distribution and is third-party attribution, not author
metadata for this anonymous submission.

The model also depends on projects that are installed separately:

- DINOv3: source and weights are governed by Meta's DINOv3 license.
- Depth Anything V2: model architecture is included through the adapted
  TranSplat tree; weights retain their upstream terms.
- UniMatch/GMDepth: training initialization weights retain their upstream terms.
- MMCV, PyTorch, xFormers, and downstream host systems retain their own licenses.

Before public release, add download URLs and checksums for every checkpoint you
publish. Do not upload third-party weights unless their licenses allow it.
