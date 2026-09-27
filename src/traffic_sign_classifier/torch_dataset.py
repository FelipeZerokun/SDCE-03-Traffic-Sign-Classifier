"""Load traffic-sign images for PyTorch."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomAffine

from traffic_sign_classifier.dataset import Annotation
from traffic_sign_classifier.preprocessing import preprocess_image


class TrafficSignDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load images with optional random position and scale augmentation."""

    def __init__(
        self,
        annotations: Sequence[Annotation],
        data_root: Path,
        *,
        augment: bool = False,
    ) -> None:
        self.annotations = tuple(annotations)
        self.data_root = data_root

        self.augmentation = (
            RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        annotation = self.annotations[index]
        image_path = self.data_root / annotation.image_path

        try:
            with Image.open(image_path) as image:
                tensor = preprocess_image(image)
        except OSError as error:
            raise ValueError(f"{image_path}: cannot load image") from error

        if self.augmentation is not None:
            tensor = cast(torch.Tensor, self.augmentation(tensor))

        return tensor, annotation.class_id
