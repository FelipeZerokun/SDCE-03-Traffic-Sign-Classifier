"""Convert images into model-ready tensors."""

from typing import cast

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Return an RGB float32 tensor of shape [3, 32, 32] in [0, 1]."""
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(
        (32, 32),
        resample=Image.Resampling.BILINEAR,
    )

    tensor = cast(torch.Tensor, pil_to_tensor(resized_image))
    return tensor.to(dtype=torch.float32).div(255.0)
