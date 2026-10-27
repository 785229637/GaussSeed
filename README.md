# GaussSeed

GaussSeed is the anonymous code package for **“GaussSeed: Backend-Aware Learned Gaussian Initialization and Map Expansion for Monocular 3DGS SLAM”** It predicts canonical
3D Gaussians from two calibrated RGB frames and exposes a backend-independent
module for scale calibration, geometric alignment, consistency-aware filtering,
novelty selection, tensor adaptation, and optimizer-state initialization.

This review package intentionally contains **module code only**. It does not
contain patches, insertion examples, or source files from any evaluated SLAM
system. A host system only needs to pass its camera state and depth rendered
from its existing Gaussian map to the public module API.



## Checkpoint availability

The exact trained GaussSeed checkpoint used for the paper is **not included in
the anonymous review package**. If the paper is accepted, the authors will
publicly release the specific checkpoint together with its download.

## Installation

Linux, Python 3.10, and a CUDA-capable GPU are recommended. Install a PyTorch
build matching the local CUDA toolkit, then install the remaining dependencies:

```bash
conda create -n gaussseed python=3.10 -y
conda activate gaussseed
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install DINOv3, the modified rasterizer, and this package:

```bash
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3-src
pip install -e third_party/dinov3-src
pip install -e third_party/diff-gaussian-rasterization-modified
pip install -e .
```

## Standalone proposal inference

The camera JSON contains two normalized intrinsic matrices and two 4×4
camera-to-world matrices. Pixel-space intrinsics are also supported when
`intrinsics_normalized` is set to `false`.

```bash
gaussseed-predict \
  --checkpoint checkpoints/encoder.ckpt \
  --images /path/to/frame_000.png /path/to/frame_005.png \
  --camera-json examples/cameras.json \
  --output outputs/canonical_gaussians.npz \
  --device cuda
```

The NPZ file preserves `means`, `covariances`, `harmonics`, `opacities`, refined
extrinsics, pose update, and proposal depth. It is not reduced to a colored
point cloud.

## Public module API

```python
from gaussseed import GaussSeed

model = GaussSeed("checkpoints/encoder.ckpt", device="cuda")
context = model.make_context(images, intrinsics, camera_to_world, near=0.1, far=10.0)
prediction = model.predict(context)

prepared = model.prepare_backend(
    prediction,
    target_camera_to_world=current_camera_to_world,
    target_intrinsics=current_intrinsics,
    map_rendered_depth=current_map_rendered_depth,
    proposal_to_global_rotation=proposal_to_global_rotation,
    proposal_to_global_translation=proposal_to_global_translation,
    initializing=False,
)
```

`map_rendered_depth` must come from the active Gaussian map. Sensor depth and
ground-truth depth are not inputs to this API. `prepared.backend` contains
anisotropic log-scales, quaternions, opacity logits, complete SH coefficients,
and zero-initialized Adam moments. If every candidate is rejected,
`prepared.requires_native_fallback` is `True` so the host can retain its native
birth operator.

The full tensor contract is documented in [`docs/MODULE_API.md`](docs/MODULE_API.md).

## Training

The proposal network used in the paper is trained on **RealEstate10K (RE10K)**.
Training is configured with Hydra, and the dataset root is supplied through an
environment variable so no local absolute path is stored in the package.

```bash
export RE10K_ROOT=/path/to/re10k
python train.py +experiment=re10k \
  model.encoder.dinov3_weights_path=checkpoints/dinov3_vits16_pretrain_lvd1689m.pth \
  model.encoder.depth_head_weights_path=checkpoints/depth_head.pth \
  model.encoder.depth_anything_weights_path=checkpoints/depth_anything_v2_vitb.pth
```

Replica and TUM RGB-D are **not** training datasets for the paper checkpoint.
Their retained configurations are used only for post-training SLAM evaluation,
after the GaussSeed module has been connected to a host SLAM system. This review
package does not include host-specific insertion code.

See [`docs/TRAINING.md`](docs/TRAINING.md) for dataset layouts, configuration,
and reproducibility fields.



## License and third-party attribution

Our training code is adapted from TranSplat and is provided under the original MIT License. This project also utilizes DINOv3, Depth Anything V2, UniMatch/GMDepth, PyTorch, MMCV, and differentiable Gaussian rasterization. Please ensure you comply with the respective licenses and citation requirements of these third-party dependencies.
