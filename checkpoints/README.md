# Checkpoint availability

No trained GaussSeed checkpoint is distributed with the anonymous review code.

If the paper is accepted, the authors will publicly release the **specific
checkpoint used for the reported experiments** as:

```text
checkpoints/encoder.ckpt
```

The release will include:

- a stable public download URL;
- a SHA-256 checksum;
- the exact Hydra configuration and relevant overrides;
- preprocessing and inference instructions;
- versioned dependency information;
- license information for every distributed weight file.

Third-party initialization weights (DINOv3, Depth Anything V2, and
UniMatch/GMDepth) are not redistributed here and remain subject to their
upstream licenses.
