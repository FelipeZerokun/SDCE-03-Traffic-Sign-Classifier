import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    run_dir = Path("outputs/runs/baseline-v1")

    with (run_dir / "history.json").open(encoding="utf-8") as file:
        history = json.load(file)

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["training"]["loss"] for entry in history]
    val_loss = [entry["validation"]["loss"] for entry in history]

    train_accuracy = [entry["training"]["accuracy"] * 100 for entry in history]
    val_accuracy = [entry["validation"]["accuracy"] * 100 for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="Training")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Cross-entropy")

    axes[1].plot(epochs, train_accuracy, marker="o", label="Training")
    axes[1].plot(epochs, val_accuracy, marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
