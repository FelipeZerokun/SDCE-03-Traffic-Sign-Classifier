"""Command-line interface for the traffic-sign classifier."""

import argparse
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

    args = parser.parse_args(argv)

    try:
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
