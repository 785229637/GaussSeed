from ..types import BatchedExample, BatchedViews


def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    # Image size must be even so that naive center-cropping does not cause misalignment.
    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2

    # Center-crop the image.
    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new  # fx
    intrinsics[:, :, 1, 1] *= h / h_new  # fy

    result = {
        **views,
        "image": image,
        "intrinsics": intrinsics,
    }
    if "depth" in views:
        result["depth"] = views["depth"][:, :, :, row : row + h_new, col : col + w_new]
    return result

# 对批量数据中的图像进行裁剪，使得裁剪后的图像尺寸能够被指定的块大小整除。
def apply_patch_shim(batch: BatchedExample, patch_size: int) -> BatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }
