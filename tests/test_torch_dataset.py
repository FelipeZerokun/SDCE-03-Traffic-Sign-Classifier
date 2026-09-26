from pathlib import Path

import pytest
import torch
from PIL import Image

from traffic_sign_classifier.dataset import Annotation
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


@pytest.fixture
def annotation() -> Annotation:
    return Annotation(
        image_path=Path("example.png"),
        class_id=20,
        width=40,
        height=24,
        roi_x1=5,
        roi_y1=5,
        roi_x2=34,
        roi_y2=18,
    )


def test_dataset_length_without_loading_images(
    tmp_path: Path,
    annotation: Annotation,
) -> None:
    dataset = TrafficSignDataset([annotation], tmp_path)

    assert len(dataset) == 1


def test_returns_preprocessed_image_and_label(
    tmp_path: Path,
    annotation: Annotation,
) -> None:
    Image.new("RGB", (40, 24), color=(255, 0, 0)).save(tmp_path / "example.png")
    dataset = TrafficSignDataset([annotation], tmp_path)

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 20
    assert torch.all(image[0] == 1)
    assert torch.all(image[1:] == 0)


def test_reports_missing_image(
    tmp_path: Path,
    annotation: Annotation,
) -> None:
    dataset = TrafficSignDataset([annotation], tmp_path)

    with pytest.raises(ValueError, match="cannot load image"):
        dataset[0]


def test_rejects_out_of_range_index(
    tmp_path: Path,
    annotation: Annotation,
) -> None:
    dataset = TrafficSignDataset([annotation], tmp_path)

    with pytest.raises(IndexError):
        dataset[1]
