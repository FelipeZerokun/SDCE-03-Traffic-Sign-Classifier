"""Command-line interface for the traffic-sign classifier."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from traffic_sign_classifier.audit import audit_dataset
from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.dataset import read_annotations
from traffic_sign_classifier.provenance import sha256_file
from traffic_sign_classifier.split import save_split_manifest, split_annotations


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and inspect a traffic-sign classifier."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit_parser = commands.add_parser(
        "audit",
        help="Validate training annotations and inspect training images.",
    )
    audit_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the dataset TOML configuration.",
    )

    split_parser = commands.add_parser(
        "split",
        help="Create training and validation assignments by track.",
    )
    split_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the dataset TOML configuration.",
    )
    split_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New manifest path, relative to the current directory.",
    )

    train_parser = commands.add_parser("train", help="Train and save the baseline CNN.")
    train_parser.add_argument("--config", type=Path, required=True)

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a saved model.")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument(
        "--split", choices=("validation", "test"), required=True
    )
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    predict_parser = commands.add_parser(
        "predict", help="Predict a cropped sign image."
    )
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    args = parser.parse_args(argv)

    try:
        if args.command == "predict":
            from traffic_sign_classifier.inference import load_model, predict

            model, _ = load_model(args.checkpoint, args.device)
            print(json.dumps(predict(model, args.image, args.top_k), indent=2))
            return 0
        if args.command == "evaluate":
            from traffic_sign_classifier.evaluation import evaluate

            report = evaluate(
                args.checkpoint,
                args.config,
                args.manifest,
                args.split,
                args.output,
                args.device,
            )
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in ("split", "images", "accuracy", "macro_f1")
                    },
                    indent=2,
                )
            )
            print(f"Reports saved: {args.output.resolve()}")
            return 0
        if args.command == "train":
            from traffic_sign_classifier.training import load_training_config, train

            train(load_training_config(args.config))
            return 0

        config = load_dataset_config(args.config)

        if args.command == "split":
            annotations_sha256 = sha256_file(config.train_annotations)
            annotations = read_annotations(config.train_annotations)
            split = split_annotations(
                annotations,
                config.split.validation_fraction,
                config.split.seed,
            )
            save_split_manifest(
                split,
                args.output,
                config.split.validation_fraction,
                config.split.seed,
                annotations_sha256=annotations_sha256,
            )

            print(f"Training images: {len(split.training)}")
            print(f"Validation images: {len(split.validation)}")
            print(f"Manifest saved: {args.output.resolve()}")
            return 0

        summary = audit_dataset(config.train_annotations, config.root)

    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Images checked: {summary.image_count}")
    print(f"Classes observed: {len(summary.class_counts)}")

    print("Image modes:")
    for mode, count in summary.mode_counts.items():
        print(f"  {mode}: {count}")

    print("Images per class:")
    for class_id, count in summary.class_counts.items():
        print(f"  {class_id}: {count}")

    print(f"Training tracks: {len(summary.track_counts)}")

    size_counts = Counter(summary.track_counts.values())
    print("Track sizes:")
    for size, count in sorted(size_counts.items()):
        print(f"  {size} images: {count} tracks")

    print("Tracks with sizes other than 30:")
    for (class_id, track_id), count in summary.track_counts.items():
        if count != 30:
            print(f"  Class {class_id}, track {track_id:05d}: {count} images")

    return 0
