from src.data.loader import load_fashion_mnist
from src.data.preprocessing import (
    add_channel_dimension,
    add_gaussian_noise,
    normalize_images,
)
from src.models.autoencoder import build_autoencoder
from src.training.trainer import (
    compile_autoencoder,
    train_autoencoder,
)


def main():
    x_train, _, x_test, _ = load_fashion_mnist()

    x_train_clean = normalize_images(x_train)
    x_test_clean = normalize_images(x_test)

    x_train_clean = add_channel_dimension(x_train_clean)
    x_test_clean = add_channel_dimension(x_test_clean)

    x_train_noisy = add_gaussian_noise(
        x_train_clean,
        seed=42,
    )

    x_test_noisy = add_gaussian_noise(
        x_test_clean,
        seed=43,
    )

    print("Training data prepared.")
    print()
    print(f"Clean training shape: {x_train_clean.shape}")
    print(f"Noisy training shape: {x_train_noisy.shape}")
    print(f"Clean test shape: {x_test_clean.shape}")
    print(f"Noisy test shape: {x_test_noisy.shape}")

    autoencoder = build_autoencoder()

    compile_autoencoder(autoencoder)

    print()
    print("Starting training...")
    print()

    history = train_autoencoder(
        model=autoencoder,
        x_train_noisy=x_train_noisy,
        x_train_clean=x_train_clean,
        x_test_noisy=x_test_noisy,
        x_test_clean=x_test_clean,
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


if __name__ == "__main__":
    main()