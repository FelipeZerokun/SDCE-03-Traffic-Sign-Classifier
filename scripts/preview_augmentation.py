"""Preview position and scale augmentation on training images."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.manifest import load_split_manifest
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_dataset_config(project_root / "configs/dataset.toml")

    split = load_split_manifest(
        project_root / "outputs/splits/baseline-v2.json",
        config.train_annotations,
    )

    original_dataset = TrafficSignDataset(split.training, config.root)
    augmented_dataset = TrafficSignDataset(
        split.training,
        config.root,
        augment=True,
    )

    selected_indices = [
        next(
            index
            for index, annotation in enumerate(split.training)
            if annotation.class_id == class_id
        )
        for class_id in (5, 14, 27, 33)
    ]

    # Seed once so rerunning this preview reproduces the same variants.
    torch.manual_seed(42)

    fig, axes = plt.subplots(
        len(selected_indices),
        5,
        figsize=(15, 3 * len(selected_indices)),
        squeeze=False,
    )

    for row, index in enumerate(selected_indices):
        original, label = original_dataset[index]
        variants = [original]

        for _ in range(4):
            augmented, augmented_label = augmented_dataset[index]

            assert augmented_label == label
            assert augmented.shape == (3, 32, 32)
            assert augmented.dtype == torch.float32
            assert torch.isfinite(augmented).all().item()
            assert 0.0 <= augmented.min().item()
            assert augmented.max().item() <= 1.0

            variants.append(augmented)

        for column, tensor in enumerate(variants):
            ax = axes[row, column]
            ax.imshow(
                tensor.permute(1, 2, 0).numpy(),
                interpolation="nearest",
            )
            title = "Original input" if column == 0 else f"Variant {column}"
            ax.set_title(f"Class {label}\n{title}")
            ax.axis("off")

    fig.suptitle("Training-only augmentation preview")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir = project_root / "outputs/inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "augmentation_preview.png"
    fig.savefig(output_path, dpi=150)

    print(f"Preview saved: {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
