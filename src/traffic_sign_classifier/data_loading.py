"""Create reproducible batches of traffic-sign images."""

import torch
from torch.utils.data import DataLoader

from traffic_sign_classifier.torch_dataset import TrafficSignDataset


def create_data_loader(
    dataset: TrafficSignDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Create a loader that retains every example, including a partial batch."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
