# DINOv3 dependency

DINOv3 source code and weights are not redistributed in this repository.
Review and accept Meta's DINOv3 license, then install the official project:

```bash
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3-src
pip install -e third_party/dinov3-src
```

The initializer imports the public `dinov3.hub.backbones` and dense-depth
modules from that installation.
