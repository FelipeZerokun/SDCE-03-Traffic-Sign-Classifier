"""Reproduce the five-image legacy demonstration with the frozen model."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from traffic_sign_classifier.classes import CLASS_NAMES
from traffic_sign_classifier.inference import load_model, predict
from traffic_sign_classifier.provenance import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "outputs/runs/augmentation-v1/best.pt"
    output = root / "outputs/final/demo-reviewed"
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    model, _ = load_model(checkpoint)
    records = []
    figure, axes = plt.subplots(1, 5, figsize=(18, 5))
    for index, (actual, axis) in enumerate(
        zip((17, 25, 2, 12, 14), axes, strict=True), start=1
    ):
        path = (
            root / f"legacy/Traffic_Sign_Classifier_Project/test_images/test{index}.png"
        )
        scores = predict(model, path)
        records.append(
            {
                "image": path.relative_to(root).as_posix(),
                "image_sha256": sha256_file(path),
                "actual": actual,
                "top_k": scores,
            }
        )
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"Image {index}: {CLASS_NAMES[actual]}\n"
            f"Predicted: {scores[0]['class_name']}\n"
            f"Score: {scores[0]['score']:.1%}",
            fontsize=9,
        )
        axis.axis("off")
    correct = sum(row["actual"] == row["top_k"][0]["class_id"] for row in records)
    output.mkdir(parents=True)
    report = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "correct": correct,
        "images": 5,
        "accuracy": correct / 5,
        "source": "Historical local legacy images; original URLs/licenses unknown",
        "label_review": "Image 4 is Priority road (12), correcting legacy Yield label",
        "predictions": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    figure.tight_layout()
    figure.savefig(output / "predictions.png", dpi=140)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
