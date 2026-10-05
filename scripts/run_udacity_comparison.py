"""Run both Udacity experiments, then evaluate frozen validation-selected models."""

import argparse
import json
from pathlib import Path

from traffic_sign_classifier.evaluation import evaluate
from traffic_sign_classifier.training import load_training_config, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args()
    if not args.evaluate_only:
        for variant in ("baseline", "augmentation"):
            train(load_training_config(Path(f"configs/udacity-{variant}.toml")))
    for run in (
        "baseline-v1",
        "augmentation-v1",
        "udacity-baseline-v1",
        "udacity-augmentation-v1",
    ):
        run_dir = Path("outputs/runs") / run
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        if [row["epoch"] for row in history] != list(
            range(1, metadata["settings"]["epochs"] + 1)
        ):
            raise ValueError(f"Training is incomplete: {run}")
    # All training and checkpoint selection finish before test evaluation.
    for dataset in ("current", "udacity"):
        config = Path(
            "configs/dataset.toml"
            if dataset == "current"
            else "configs/udacity-dataset.toml"
        )
        manifest = Path(
            "outputs/splits/baseline-v2.json"
            if dataset == "current"
            else "data/udacity/split.json"
        )
        for variant in ("baseline", "augmentation"):
            run = f"{variant}-v1" if dataset == "current" else f"udacity-{variant}-v1"
            for split in ("validation", "test"):
                output = args.output_root / run / split
                report = evaluate(
                    Path(f"outputs/runs/{run}/best.pt"),
                    config,
                    manifest,
                    split,
                    output,
                    "cuda",
                )
                print(
                    f"{run} {split}: accuracy={report['accuracy']:.4f}, macro F1={report['macro_f1']:.4f}",
                    flush=True,
                )
