"""Create comparison figures from saved histories and frozen-model reports."""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from traffic_sign_classifier.classes import CLASS_NAMES

RUNS = [
    "baseline-v1",
    "augmentation-v1",
    "udacity-baseline-v1",
    "udacity-augmentation-v1",
]
LABELS = [
    "Current / baseline",
    "Current / augmented",
    "Udacity / baseline",
    "Udacity / augmented",
]
COLORS = ["#3774a8", "#22a6a1", "#dc8245", "#a557ad"]
OUT = Path("docs/figures/udacity-comparison")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {run: read(f"outputs/runs/{run}/run.json") for run in RUNS}
    reference = metadata[RUNS[0]]
    for run, info in metadata.items():
        for key in ("epochs", "batch_size", "learning_rate", "seed"):
            if info["settings"][key] != reference["settings"][key]:
                raise ValueError(f"Unmatched training setting: {run} / {key}")
        for key in ("architecture", "preprocessing", "versions"):
            if info[key] != reference[key]:
                raise ValueError(f"Unmatched pipeline metadata: {run} / {key}")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafbfd",
            "axes.facecolor": "#fafbfd",
        }
    )
    reports = {
        run: {
            split: read(args.reports / run / split / "report.json")
            for split in ("validation", "test")
        }
        for run in RUNS
    }
    histories = {run: read(f"outputs/runs/{run}/history.json") for run in RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, metric in enumerate(("accuracy", "loss")):
        for col, split in enumerate(("training", "validation")):
            ax = axes[row, col]
            for run, label, color in zip(RUNS, LABELS, COLORS, strict=True):
                history = histories[run]
                ax.plot(
                    [h["epoch"] for h in history],
                    [
                        h[split][metric] * (100 if metric == "accuracy" else 1)
                        for h in history
                    ],
                    label=label,
                    color=color,
                    marker="o",
                    markersize=3,
                )
            ax.set(
                title=f"{split.title()} {metric}",
                xlabel="Epoch",
                ylabel="Accuracy (%)" if metric == "accuracy" else "Cross-entropy loss",
            )
            ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Same CNN and training settings • two dataset preparations", fontsize=18
    )
    fig.text(
        0.5,
        0.02,
        "10 epochs · Adam 0.001 · batch 128 · seed 42 | Training curves include augmentation where enabled.\nCurrent: 31,379 / 7,830 train / validation; Udacity: 34,799 / 4,410. Validation populations differ.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, "learning-curves")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(4)
    for ax, split in zip(axes, ("validation", "test"), strict=True):
        for offset, metric, color in (
            (-0.19, "accuracy", "#3774a8"),
            (0.19, "macro_f1", "#dc8245"),
        ):
            values = [reports[run][split][metric] * 100 for run in RUNS]
            bars = ax.bar(
                x + offset,
                values,
                0.36,
                label="Accuracy" if metric == "accuracy" else "Macro F1 × 100",
                color=color,
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        ax.set(
            ylim=(0, 108),
            xticks=x,
            xticklabels=[
                "Current\nbaseline",
                "Current\naugmented",
                "Udacity\nbaseline",
                "Udacity\naugmented",
            ],
            title=f"{split.title()} • best validation checkpoint",
            ylabel="Score (0–100)",
        )
        ax.legend(loc="lower right")
    fig.suptitle("Frozen checkpoint results", fontsize=18)
    fig.text(
        0.5,
        0.02,
        "Single seed; different validation splits and image preparation. Test bars use each distribution's supplied images.\nExact-overlap-excluded test results and audit details are recorded in the report.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    save(fig, "results-summary")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for ax, split in zip(axes, ("validation", "test"), strict=True):
        values = np.array(
            [
                [c["recall"] * 100 for c in reports[run][split]["per_class"]]
                for run in RUNS
            ]
        )
        im = ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        ax.set(
            yticks=range(4),
            yticklabels=LABELS,
            title=f"{split.title()} recall by class",
            xticks=range(43),
        )
        fig.colorbar(im, ax=ax, label="Recall (%)", fraction=0.025, pad=0.02)
    axes[1].set_xlabel("GTSRB class ID (0–42)")
    fig.suptitle("Where each model succeeds and struggles", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "class-recall")

    # A deterministic selection spanning distinct true classes; no confidence cherry-picking.
    run = "udacity-augmentation-v1"
    predictions = reports[run]["test"]["predictions"]
    selected = []
    for correct in (True, False):
        seen = set()
        for p in predictions:
            if (p["actual"] == p["predicted"]) == correct and p["actual"] not in seen:
                selected.append(p)
                seen.add(p["actual"])
                if len(seen) == 6:
                    break
    fig, axes = plt.subplots(2, 6, figsize=(16, 10))
    for ax, p in zip(axes.flat, selected, strict=True):
        with Image.open(Path("data/udacity") / p["path"]) as image:
            ax.imshow(image, interpolation="nearest")
        correct = p["actual"] == p["predicted"]
        ax.set_title(
            f"Actual {p['actual']} → predicted {p['predicted']}\n"
            + textwrap.fill(CLASS_NAMES[p["actual"]], 23)
            + "\nPrediction: "
            + textwrap.fill(CLASS_NAMES[p["predicted"]], 23),
            color="#26785c" if correct else "#b44242",
            fontsize=10,
        )
        ax.axis("off")
    fig.suptitle("Udacity augmented model • test successes and errors", fontsize=17)
    fig.text(
        0.5,
        0.02,
        "Top: correct predictions. Bottom: mistakes. First example per distinct true class in supplied order; IDs refer to GTSRB classes.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.025, right=0.985, bottom=0.1, top=0.78, hspace=0.85, wspace=0.12
    )
    save(fig, "prediction-examples")
    summary = []
    for run, label in zip(RUNS, LABELS, strict=True):
        val, test = reports[run]["validation"], reports[run]["test"]
        summary.append(
            {
                "run": run,
                "label": label,
                "epoch": val["checkpoint_epoch"],
                "validation_accuracy": val["accuracy"],
                "validation_macro_f1": val["macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_exact_overlap_images": test["audit"]["overlapping_test_images"],
                "test_nonoverlapping_accuracy": test["nonoverlapping_subset"][
                    "accuracy"
                ],
                "test_nonoverlapping_images": test["nonoverlapping_subset"]["images"],
                "test_nonoverlapping_macro_f1": test["nonoverlapping_subset"][
                    "macro_f1"
                ],
            }
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
