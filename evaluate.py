from tensorflow.keras.models import load_model

from src.data.loader import load_fashion_mnist
from src.data.preprocessing import (
    add_channel_dimension,
    add_gaussian_noise,
    normalize_images,
)
from src.evaluation.plots import plot_denoising_results


def main():
    _, _, x_test, _ = load_fashion_mnist()

    x_test_clean = normalize_images(x_test)

    x_test_clean = add_channel_dimension(
        x_test_clean
    )

    x_test_noisy = add_gaussian_noise(
        x_test_clean,
        seed=44,
    )

    autoencoder = load_model(
        "artifacts/denoising_autoencoder.keras"
    )

    print("Trained model loaded successfully.")
    print()
    print(f"Test images shape: {x_test_clean.shape}")
    print()
    print("Generating denoised images...")
    print()

    reconstructed_images = autoencoder.predict(
        x_test_noisy,
        verbose=1,
    )

    print()
    print(
        f"Reconstructed images shape: "
        f"{reconstructed_images.shape}"
    )

    plot_denoising_results(
        noisy_images=x_test_noisy,
        reconstructed_images=reconstructed_images,
        clean_images=x_test_clean,
        sample_count=5,
    )


if __name__ == "__main__":
    main()