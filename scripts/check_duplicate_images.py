"""Find exact RGB image duplicates in training and validation."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.manifest import load_split_manifest


def image_fingerprint(path: Path) -> str:
    """Hash original dimensions and decoded RGB pixels."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")

        digest = hashlib.sha256()
        digest.update(rgb.width.to_bytes(8, "big"))
        digest.update(rgb.height.to_bytes(8, "big"))
        digest.update(rgb.tobytes())

    return digest.hexdigest()


def main() -> None:
    config = load_dataset_config(Path("configs/dataset.toml"))
    split = load_split_manifest(
        Path("outputs/splits/baseline-v2.json"),
        config.train_annotations,
    )

    groups: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    checked = 0

    for split_name, annotations in (
        ("training", split.training),
        ("validation", split.validation),
    ):
        for annotation in annotations:
            fingerprint = image_fingerprint(config.root / annotation.image_path)
            groups[fingerprint].append(
                {
                    "split": split_name,
                    "path": annotation.image_path.as_posix(),
                    "class_id": annotation.class_id,
                }
            )

            checked += 1
            if checked % 5000 == 0:
                print(f"Checked {checked} images...")

    duplicates = [
        {"sha256": fingerprint, "records": records}
        for fingerprint, records in sorted(groups.items())
        if len(records) > 1
    ]

    cross_split = sum(
        len({record["split"] for record in group["records"]}) > 1
        for group in duplicates
    )
    conflicting_labels = sum(
        len({record["class_id"] for record in group["records"]}) > 1
        for group in duplicates
    )

    output_path = Path("outputs/inspection/duplicate_images.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "comparison": "original dimensions and decoded RGB pixels",
                "images_checked": checked,
                "duplicate_groups": duplicates,
                "cross_split_groups": cross_split,
                "conflicting_label_groups": conflicting_labels,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Images checked: {checked}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Groups crossing splits: {cross_split}")
    print(f"Groups with conflicting labels: {conflicting_labels}")
    print(f"Report saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
