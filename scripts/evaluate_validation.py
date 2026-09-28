"""Evaluate the baseline checkpoint on training and validation splits."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.data_loading import create_data_loader
from traffic_sign_classifier.dataset import training_track_key
from traffic_sign_classifier.manifest import load_split_manifest
from traffic_sign_classifier.model import BaselineCNN
from traffic_sign_classifier.provenance import sha256_file
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


def plot_class_tracks(
    dataset: TrafficSignDataset,
    predicted_labels: list[int],
    class_id: int,
    output_dir: Path,
) -> None:
    """Show first, middle, and last frames from each track of a class."""
    track_indices: dict[int, list[int]] = {}

    for index, annotation in enumerate(dataset.annotations):
        if annotation.class_id == class_id:
            _, track_id = training_track_key(annotation)
            track_indices.setdefault(track_id, []).append(index)

    if not track_indices:
        print(f"No validation images found for class {class_id}.")
        return

    fig, axes = plt.subplots(
        len(track_indices),
        3,
        figsize=(12, 4 * len(track_indices)),
        squeeze=False,
    )

    positions = ("First", "Middle", "Last")

    for row, (track_id, indices) in enumerate(sorted(track_indices.items())):
        indices.sort(key=lambda index: dataset.annotations[index].image_path.name)

        selected_indices = [
            indices[0],
            indices[len(indices) // 2],
            indices[-1],
        ]

        for column, index in enumerate(selected_indices):
            annotation = dataset.annotations[index]
            tensor, actual = dataset[index]
            predicted = predicted_labels[index]

            ax = axes[row, column]
            ax.imshow(
                tensor.permute(1, 2, 0).numpy(),
                interpolation="nearest",
            )
            ax.set_title(
                f"Track {track_id:05d} — {positions[column]}\n"
                f"{annotation.image_path.name}\n"
                f"Actual: {actual} | Predicted: {predicted}",
                color="green" if actual == predicted else "red",
                fontsize=10,
            )
            ax.axis("off")

    fig.suptitle(
        f"Class {class_id}: validation inputs by track",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = output_dir / f"class_{class_id}_tracks.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nImage grid saved: {output_path}")

    plt.show()
    plt.close(fig)


def evaluate_training(
    model: BaselineCNN,
    dataset: TrafficSignDataset,
    checkpoint_epoch: int,
    output_dir: Path,
) -> None:
    """Measure a fixed checkpoint on training data without updating weights."""
    loader = create_data_loader(
        dataset,
        batch_size=128,
        shuffle=False,
        seed=42,
    )

    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    if len(true_labels) != len(dataset) or len(predicted_labels) != len(dataset):
        raise ValueError("Prediction count does not match the training dataset.")

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(43)),
    )

    accuracy = float(np.trace(matrix) / matrix.sum())
    class_27_total = int(matrix[27].sum())
    class_27_correct = int(matrix[27, 27])
    class_27_recall = class_27_correct / class_27_total

    class_report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(43)),
        digits=3,
        zero_division=0,
        output_dict=False,
    )

    if not isinstance(class_report, str):
        raise TypeError("Expected classification_report to return text.")

    report = "\n".join(
        [
            f"Checkpoint epoch: {checkpoint_epoch}",
            "Split: training (diagnostic evaluation, not unseen data)",
            f"Training images: {len(true_labels)}",
            f"Training accuracy: {accuracy:.2%}",
            (
                f"Class 27 training recall: "
                f"{class_27_correct}/{class_27_total} "
                f"({class_27_recall:.2%})"
            ),
            "\nPer-class training metrics:",
            class_report,
        ]
    )

    print("\n" + report)

    report_path = output_dir / "training_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    np.save(output_dir / "training_confusion_matrix.npy", matrix)

    print(f"\nTraining report saved: {report_path}")


def main() -> None:
    # Resolve paths from the script location, not the terminal directory.
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("outputs/runs/baseline-v1")
    )
    args = parser.parse_args()
    run_dir = (project_root / args.run_dir).resolve()
    checkpoint_path = run_dir / "best.pt"
    manifest_path = project_root / "outputs" / "splits" / "baseline-v2.json"
    config_path = project_root / "configs" / "dataset.toml"
    inspected_class = 27

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    dataset_config = load_dataset_config(config_path)

    # Ensure we evaluate against the source files used for this run.
    if (
        sha256_file(dataset_config.train_annotations)
        != checkpoint["annotations_sha256"]
    ):
        raise ValueError("Annotations do not match the training checkpoint.")

    if sha256_file(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("Split manifest does not match the training checkpoint.")

    split = load_split_manifest(
        manifest_path,
        dataset_config.train_annotations,
    )

    validation_dataset = TrafficSignDataset(
        split.validation,
        dataset_config.root,
    )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=128,
        shuffle=False,
        seed=42,
    )

    model = BaselineCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    true_labels: list[int] = []
    predicted_labels: list[int] = []

    # Preserve dataset order so predictions can be matched to image paths.
    with torch.inference_mode():
        for images, labels in validation_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    if len(true_labels) != len(validation_dataset) or len(predicted_labels) != len(
        validation_dataset
    ):
        raise ValueError("Prediction count does not match the validation dataset.")

    correct = sum(
        actual == predicted
        for actual, predicted in zip(
            true_labels,
            predicted_labels,
            strict=True,
        )
    )
    accuracy = correct / len(true_labels)

    class_report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(43)),
        digits=3,
        zero_division=0,
        output_dict=False,
    )

    if not isinstance(class_report, str):
        raise TypeError("Expected classification_report to return text.")

    report_lines = [
        f"Saved epoch: {checkpoint['epoch']}",
        (f"Saved validation accuracy: {checkpoint['validation']['accuracy']:.2%}"),
        f"Validation images: {len(true_labels)}",
        f"Recomputed accuracy: {accuracy:.2%}",
        "\nPer-class validation metrics:",
        class_report,
    ]

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(43)),
    )

    # Summarize predictions for the class under investigation.
    support = int(matrix[inspected_class].sum())
    report_lines.append(f"\nPredictions for actual class {inspected_class}:")

    for predicted_class in np.argsort(matrix[inspected_class])[::-1]:
        count = int(matrix[inspected_class, predicted_class])
        if count == 0:
            continue

        report_lines.append(
            f"Predicted {predicted_class:2d}: {count}/{support} ({count / support:.1%})"
        )

    # Group related frames to avoid interpreting them as independent signs.
    track_predictions: dict[int, Counter[int]] = defaultdict(Counter)

    for annotation, predicted in zip(
        validation_dataset.annotations,
        predicted_labels,
        strict=True,
    ):
        if annotation.class_id == inspected_class:
            _, track_id = training_track_key(annotation)
            track_predictions[track_id][predicted] += 1

    report_lines.append(f"\nClass {inspected_class} performance by track:")

    for track_id, counts in sorted(track_predictions.items()):
        total = sum(counts.values())
        track_correct = counts[inspected_class]

        report_lines.append(
            f"Track {track_id:05d}: "
            f"{track_correct}/{total} correct "
            f"({track_correct / total:.1%}); "
            f"predictions={dict(counts.most_common())}"
        )

    # Rank mistakes across all classes; exclude correct predictions.
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    ranked_indices = np.argsort(errors, axis=None)[::-1]

    report_lines.append("\nTop 10 confusion pairs:")

    for index in ranked_indices[:10]:
        actual, predicted = np.unravel_index(index, errors.shape)
        count = int(errors[actual, predicted])

        if count == 0:
            break

        class_support = int(matrix[actual].sum())
        report_lines.append(
            f"Actual {actual:2d} -> predicted {predicted:2d}: "
            f"{count} images "
            f"({count / class_support:.1%} of class {actual})"
        )

    report = "\n".join(report_lines)
    print(report)

    report_path = run_dir / "validation_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    # Row = actual class, column = predicted class; both ordered 0–42.
    np.save(run_dir / "validation_confusion_matrix.npy", matrix)
    print(f"\nReport saved: {report_path}")

    training_dataset = TrafficSignDataset(
        split.training,
        dataset_config.root,
    )

    evaluate_training(
        model,
        training_dataset,
        checkpoint["epoch"],
        run_dir,
    )

    plot_class_tracks(
        validation_dataset,
        predicted_labels,
        inspected_class,
        run_dir,
    )


if __name__ == "__main__":
    main()
