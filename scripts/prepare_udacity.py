"""Convert trusted local Udacity pickles losslessly for the existing pipeline.

Only load pickle files from a trusted source: pickle can execute Python code.
The provided partition and array order are preserved; track IDs are unavailable.
"""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from traffic_sign_classifier.provenance import sha256_file


def prepare(source: Path, output: Path) -> None:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    arrays = {}
    sources = {}
    for name, filename in (
        ("training", "train.p"),
        ("validation", "valid.p"),
        ("test", "test.p"),
    ):
        path = source / filename
        sources[filename] = sha256_file(path)
        with path.open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
        images, labels = data["features"], data["labels"]
        if images.dtype != np.uint8 or images.shape != (len(labels), 32, 32, 3):
            raise ValueError(f"Invalid RGB arrays in {path}")
        if (
            labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)
            or set(labels.tolist()) != set(range(43))
        ):
            raise ValueError(f"Invalid labels in {path}")
        arrays[name] = (images, labels)
    output.mkdir(parents=True)
    records = {}
    hashes = {}
    header = [
        "Width",
        "Height",
        "Roi.X1",
        "Roi.Y1",
        "Roi.X2",
        "Roi.Y2",
        "ClassId",
        "Path",
    ]
    rows = {}
    for name, (images, labels) in arrays.items():
        (output / name).mkdir()
        records[name], rows[name], hashes[name] = [], [], set()
        for index, (image, label) in enumerate(zip(images, labels, strict=True)):
            path = f"{name}/{index:05d}.png"
            Image.fromarray(image).save(output / path)
            records[name].append({"path": path, "class_id": int(label)})
            rows[name].append([32, 32, 0, 0, 31, 31, int(label), path])
            hashes[name].add(hashlib.sha256(image.tobytes()).hexdigest())
        print(f"Prepared {name}: {len(labels)} images", flush=True)
    for filename, names in (
        ("Train.csv", ("training", "validation")),
        ("Test.csv", ("test",)),
    ):
        with (output / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for name in names:
                writer.writerows(rows[name])
    manifest = {
        "schema_version": 3,
        "split_policy": "provided",
        "annotations_sha256": sha256_file(output / "Train.csv"),
        "training": records["training"],
        "validation": records["validation"],
        "source_sha256": sources,
        "track_independence_verified": False,
    }
    (output / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {
        "source_sha256": sources,
        "counts": {name: len(labels) for name, (_, labels) in arrays.items()},
        "exact_training_validation_duplicate_groups": len(
            hashes["training"] & hashes["validation"]
        ),
        "track_independence_verified": False,
        "near_duplicates_checked": False,
        "conversion": "Lossless RGB PNG; original array order and provided splits preserved",
    }
    (output / "provenance.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/traffic-signs-data"))
    parser.add_argument("--output", type=Path, default=Path("data/udacity"))
    args = parser.parse_args()
    prepare(args.source, args.output)
