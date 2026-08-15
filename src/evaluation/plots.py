from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    output_path="outputs/training_history.png",
):
    """
    Plot training and validation loss across epochs.

    Args:
        history: Training history returned by Keras.
        output_path: Path where the generated plot will be saved.
    """
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_loss,
        marker="o",
        label="Training loss",
    )

    plt.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Denoising Autoencoder Training History")
    plt.legend()
    plt.grid(True)

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    print(f"Training history plot saved to: {output_file}")