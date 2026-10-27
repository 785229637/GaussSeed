# Training Guide

## Dataset layout

The proposal network reported in the paper is trained on **RealEstate10K
(RE10K)**. Set the preprocessed dataset root through the environment instead of
editing YAML:

```bash
export RE10K_ROOT=/datasets/re10k
```

The loader expects the chunk layout consumed by
`transplat/src/dataset/dataset_re10k.py`:

```text
$RE10K_ROOT/
├── train/
│   ├── *.torch
│   └── index.json
└── test/
    ├── *.torch
    └── index.json
```

Replica and TUM RGB-D are used only for post-training SLAM testing: the trained
GaussSeed module is connected to a host SLAM system and evaluated on these
sequences. They are not used to train the checkpoint reported in the paper.
Only the dataset-side test configurations are retained here; host-specific
insertion code is intentionally excluded from the review package.

## Initialization weights

Training from scratch loads four components from `checkpoints/`:

- DINOv3 ViT-S/16 backbone
- learned DINO depth head
- Depth Anything V2 Base
- GMDepth/UniMatch multi-view backbone

All paths can be overridden under `model.encoder.*`. A full Lightning checkpoint
contains the encoder components and can be resumed with `checkpointing.load`.

## Main configurations

- `+experiment=re10k`: paper training configuration
- `RE10K_ROOT`: location of the preprocessed RealEstate10K chunks
- `+experiment=replica` and `+experiment=replica_all_scenes`: post-training
  SLAM evaluation configurations on Replica
- `+experiment=tum_rgbd` and `+experiment=tum_rgbd_selected_scenes`:
  post-training SLAM evaluation configurations on TUM RGB-D

Common overrides:

```bash
python train.py +experiment=re10k \
  data_loader.train.batch_size=1 \
  optimizer.lr=1e-5
```

Outputs are written to Hydra's dated output directory unless `output_dir` is set.
The training entry saves `checkpoints/last.ckpt` at completion.

## Reproducibility

Record the RealEstate10K preprocessing version, chunk manifest, `seed`, component
checkpoint checksums, image size, context stride and every Hydra override. The
accepted-paper checkpoint release will include the exact training configuration.
