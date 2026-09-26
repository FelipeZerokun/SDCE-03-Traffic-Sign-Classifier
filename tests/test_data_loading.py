from pathlib import Path

import pytest
import torch
from PIL import Image

from traffic_sign_classifier.data_loading import create_data_loader
from traffic_sign_classifier.dataset import Annotation
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


@pytest.fixture
def dataset(tmp_path: Path) -> TrafficSignDataset:
    annotations = []

    for index in range(5):
        image_path = Path(f"{index}.png")
        Image.new("RGB", (40, 24), color=(index * 40, 0, 0)).save(tmp_path / image_path)
        annotations.append(
            Annotation(
                image_path=image_path,
                class_id=index,
                width=40,
                height=24,
                roi_x1=5,
                roi_y1=5,
                roi_x2=34,
                roi_y2=18,
            )
        )

    return TrafficSignDataset(annotations, tmp_path)


def test_batches_preserve_order_and_final_examples(
    dataset: TrafficSignDataset,
) -> None:
    loader = create_data_loader(dataset, batch_size=2, shuffle=False, seed=42)
    batches = list(loader)

    assert [images.shape[0] for images, _ in batches] == [2, 2, 1]
    assert torch.cat([labels for _, labels in batches]).tolist() == [0, 1, 2, 3, 4]

    for images, labels in batches:
        assert images.shape[1:] == (3, 32, 32)
        assert images.dtype == torch.float32
        assert labels.shape == (images.shape[0],)
        assert labels.dtype == torch.int64
        assert images.min().item() >= 0
        assert images.max().item() <= 1


def test_fresh_loaders_reproduce_shuffled_order(
    dataset: TrafficSignDataset,
) -> None:
    first = create_data_loader(dataset, batch_size=2, shuffle=True, seed=42)
    second = create_data_loader(dataset, batch_size=2, shuffle=True, seed=42)

    first_labels = torch.cat([labels for _, labels in first])
    second_labels = torch.cat([labels for _, labels in second])

    assert torch.equal(first_labels, second_labels)
    assert sorted(first_labels.tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_invalid_batch_size(
    dataset: TrafficSignDataset,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="Batch size must be positive"):
        create_data_loader(dataset, batch_size=batch_size, shuffle=False, seed=42)
