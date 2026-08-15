from tensorflow.keras.models import load_model

from src.data.loader import load_fashion_mnist
from src.data.preprocessing import (
    add_channel_dimension,
    add_gaussian_noise,
    normalize_images,
)
from src.evaluation.metrics import (
    calculate_mse,
    calculate_psnr,
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

    noisy_mse = calculate_mse(
        x_test_clean,
        x_test_noisy,
    )

    denoised_mse = calculate_mse(
        x_test_clean,
        reconstructed_images,
    )

    noisy_psnr = calculate_psnr(
        x_test_clean,
        x_test_noisy,
    )

    denoised_psnr = calculate_psnr(
        x_test_clean,
        reconstructed_images,
    )

    print()
    print("=== TEST SET METRICS ===")
    print()
    print(
        f"Noisy vs Clean MSE: "
        f"{noisy_mse:.6f}"
    )
    print(
        f"Denoised vs Clean MSE: "
        f"{denoised_mse:.6f}"
    )
    print()
    print(
        f"Noisy vs Clean PSNR: "
        f"{noisy_psnr:.2f} dB"
    )
    print(
        f"Denoised vs Clean PSNR: "
        f"{denoised_psnr:.2f} dB"
    )

    plot_denoising_results(
        noisy_images=x_test_noisy,
        reconstructed_images=reconstructed_images,
        clean_images=x_test_clean,
        sample_count=5,
    )


if __name__ == "__main__":
    main()