from src.data.loader import load_fashion_mnist
from src.data.preprocessing import (
    add_channel_dimension,
    add_gaussian_noise,
    normalize_images,
    split_training_validation,
)
from src.evaluation.plots import plot_training_history
from src.models.autoencoder import build_autoencoder
from src.training.trainer import (
    compile_autoencoder,
    train_autoencoder,
)


def main():
    x_train, _, x_test, _ = load_fashion_mnist()

    x_train, x_validation = split_training_validation(
        x_train,
        validation_size=6000,
    )

    x_train_clean = normalize_images(x_train)
    x_validation_clean = normalize_images(x_validation)
    x_test_clean = normalize_images(x_test)

    x_train_clean = add_channel_dimension(x_train_clean)
    x_validation_clean = add_channel_dimension(x_validation_clean)
    x_test_clean = add_channel_dimension(x_test_clean)

    x_train_noisy = add_gaussian_noise(
        x_train_clean,
        seed=42,
    )

    x_validation_noisy = add_gaussian_noise(
        x_validation_clean,
        seed=43,
    )

    x_test_noisy = add_gaussian_noise(
        x_test_clean,
        seed=44,
    )

    print("Data prepared successfully.")
    print()
    print(f"Training images: {x_train_clean.shape}")
    print(f"Validation images: {x_validation_clean.shape}")
    print(f"Test images: {x_test_clean.shape}")

    autoencoder = build_autoencoder()

    compile_autoencoder(autoencoder)

    print()
    print("Starting training...")
    print()

    history = train_autoencoder(
        model=autoencoder,
        x_train_noisy=x_train_noisy,
        x_train_clean=x_train_clean,
        x_validation_noisy=x_validation_noisy,
        x_validation_clean=x_validation_clean,
        epochs=10,
        batch_size=128,
    )

    print()
    print("Training completed.")
    print(
        f"Final training loss: "
        f"{history.history['loss'][-1]:.6f}"
    )
    print(
        f"Final validation loss: "
        f"{history.history['val_loss'][-1]:.6f}"
    )

    print()
    print(
        "Test set preserved for final evaluation: "
        f"{x_test_noisy.shape}"
    )

    plot_training_history(history)


if __name__ == "__main__":
    main()