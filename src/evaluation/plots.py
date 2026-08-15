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

    print(
        f"Training history plot saved to: "
        f"{output_file}"
    )


def plot_denoising_results(
    noisy_images,
    reconstructed_images,
    clean_images,
    sample_count=5,
    output_path="outputs/denoising_results.png",
):
    """
    Compare noisy, reconstructed, and clean images.

    Args:
        noisy_images: Noisy input images.
        reconstructed_images: Images reconstructed by the autoencoder.
        clean_images: Original clean images.
        sample_count: Number of examples to display.
        output_path: Path where the generated plot will be saved.
    """
    plt.figure(
        figsize=(12, 3 * sample_count)
    )

    for index in range(sample_count):
        plt.subplot(
            sample_count,
            3,
            index * 3 + 1,
        )

        plt.imshow(
            noisy_images[index].squeeze(),
            cmap="gray",
        )

        plt.title("Noisy")
        plt.axis("off")

        plt.subplot(
            sample_count,
            3,
            index * 3 + 2,
        )

        plt.imshow(
            reconstructed_images[index].squeeze(),
            cmap="gray",
        )

        plt.title("Denoised")
        plt.axis("off")

        plt.subplot(
            sample_count,
            3,
            index * 3 + 3,
        )

        plt.imshow(
            clean_images[index].squeeze(),
            cmap="gray",
        )

        plt.title("Clean")
        plt.axis("off")

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    print(
        f"Denoising results plot saved to: "
        f"{output_file}"
    )