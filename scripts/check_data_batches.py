"""Check model-ready batches from the real dataset."""

from pathlib import Path

import torch

from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.data_loading import create_data_loader
from traffic_sign_classifier.manifest import load_split_manifest
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


def main() -> None:
    config = load_dataset_config(Path("configs/dataset.toml"))
    split = load_split_manifest(
        Path("outputs/splits/baseline-v2.json"),
        config.train_annotations,
    )

    for name, annotations, shuffle in (
        ("Training", split.training, True),
        ("Validation", split.validation, False),
    ):
        dataset = TrafficSignDataset(annotations, config.root)
        loader = create_data_loader(
            dataset,
            batch_size=64,
            shuffle=shuffle,
            seed=config.split.seed,
        )

        images, labels = next(iter(loader))

        assert images.shape == (64, 3, 32, 32)
        assert images.dtype == torch.float32
        assert torch.isfinite(images).all().item()
        assert 0.0 <= images.min().item() <= images.max().item() <= 1.0

        assert labels.shape == (64,)
        assert labels.dtype == torch.int64
        assert 0 <= labels.min().item() <= labels.max().item() < 43

        print(f"{name}:")
        print(f"  Dataset images: {len(dataset)}")
        print(f"  Batches: {len(loader)}")
        print(f"  Image shape: {tuple(images.shape)}")
        print(f"  Label shape: {tuple(labels.shape)}")
        print(f"  Pixel range: {images.min().item():.3f}–{images.max().item():.3f}")

    print("Both batch checks passed.")


if __name__ == "__main__":
    main()
