"""Load traffic-sign images for PyTorch."""

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from traffic_sign_classifier.dataset import Annotation
from traffic_sign_classifier.preprocessing import preprocess_image


class TrafficSignDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load and preprocess one traffic-sign image at a time."""

    def __init__(
        self,
        annotations: Sequence[Annotation],
        data_root: Path,
    ) -> None:
        self.annotations = tuple(annotations)
        self.data_root = data_root

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

        return tensor, annotation.class_id
