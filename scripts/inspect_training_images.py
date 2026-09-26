"""Create an overview of images from the saved training split."""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.provenance import sha256_file


def main() -> None:
    config = load_dataset_config(Path("configs/dataset.toml"))
    manifest_path = Path("outputs/splits/baseline-v2.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["schema_version"] != 2:
        raise ValueError("Expected manifest schema version 2")

    if manifest["annotations_sha256"] != sha256_file(config.train_annotations):
        raise ValueError("The manifest does not match the annotation file")

    paths_by_class: dict[int, list[str]] = defaultdict(list)
    for record in manifest["training"]:
        paths_by_class[record["class_id"]].append(record["path"])

    if set(paths_by_class) != set(range(43)):
        raise ValueError("Expected all 43 classes in the training split")

    generator = random.Random(42)
    figure, axes = plt.subplots(6, 8, figsize=(16, 12))
    selected = []

    for axis in axes.flat:
        axis.axis("off")

    for class_id, axis in zip(sorted(paths_by_class), axes.flat, strict=False):
        relative_path = generator.choice(sorted(paths_by_class[class_id]))

        with Image.open(config.root / relative_path) as image:
            image.load()
            width, height = image.size
            axis.imshow(image, interpolation="nearest")

        axis.set_title(f"Class {class_id} | {width} × {height}", fontsize=9)
        selected.append({"class_id": class_id, "path": relative_path})

    figure.suptitle("Training split: one example per class")
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir = Path("outputs/inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "training_examples.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    selection_path = output_dir / "training_examples.json"
    selection_path.write_text(
        json.dumps(selected, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Image grid: {figure_path.resolve()}")
    print(f"Selected paths: {selection_path.resolve()}")


if __name__ == "__main__":
    main()
