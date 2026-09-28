"""Save a version-controlled metrics snapshot from final local reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "outputs/final/test/report.json").read_text())
    demo = json.loads((root / "outputs/final/demo-reviewed/report.json").read_text())
    subset = report["nonoverlapping_subset"]
    lines = [
        "# Final results snapshot",
        "",
        f"Checkpoint SHA-256: `{report['checkpoint_sha256']}`.",
        "",
        "| Evaluation | Images | Accuracy | Macro F1 |",
        "| --- | ---: | ---: | ---: |",
        f"| Full supplied test set | {report['images']} | {report['accuracy']:.2%} | {report['macro_f1']:.4f} |",
        f"| Excluding exact training/validation overlaps | {subset['images']} | {subset['accuracy']:.2%} | {subset['macro_f1']:.4f} |",
        "",
        f"The audit found {report['audit']['overlapping_test_images']} overlapping test images "
        f"in {report['audit']['exact_cross_source_duplicate_groups']} exact RGB groups. "
        f"Within-test duplicate groups: {report['audit']['within_test_duplicate_groups']}. "
        "Near-duplicate independence was not assessed.",
        "",
        "## Full supplied test set: per-class results",
        "",
        "| Class | Name | Support | Precision | Recall | F1 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_class"]:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | {row['support']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Historical external images",
            "",
            "Full photos are used without additional cropping. Original source URLs/licenses "
            "are unknown; images remain local and are not redistributed.",
            "",
            "Image 4 was visually relabeled Priority road (12), correcting the old Yield label.",
            "",
            f"Top-1 accuracy: {demo['correct']}/{demo['images']} ({demo['accuracy']:.0%}).",
            "",
            "| Image | Actual class ID | Top five class IDs and softmax scores |",
            "| --- | ---: | --- |",
        ]
    )
    for row in demo["predictions"]:
        top = "; ".join(
            f"{score['class_id']}: {score['score']:.4g}" for score in row["top_k"]
        )
        lines.append(f"| {Path(row['image']).name} | {row['actual']} | {top} |")
    lines.extend(
        [
            "",
            "Scores are softmax outputs, not calibrated confidence. "
            "This small historical sample is not an independent benchmark.",
            "",
        ]
    )
    (root / "docs/final-results.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(12, 10))
    matrix = np.array(report["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    chart = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    figure.colorbar(chart, ax=axis, label="Fraction of actual class")
    axis.set(
        xticks=range(43),
        yticks=range(43),
        xlabel="Predicted class ID",
        ylabel="Actual class ID",
        title="Frozen model: full supplied test set",
    )
    axis.tick_params(labelsize=7)
    figure.tight_layout()
    figure.savefig(root / "outputs/final/test/confusion_matrix.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
