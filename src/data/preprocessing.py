import numpy as np


def normalize_images(images):
    """
    Normalize image pixel values from [0, 255] to [0, 1].
    """
    return images.astype("float32") / 255.0


def add_channel_dimension(images):
    """
    Add the grayscale channel dimension required by Conv2D layers.

    Example:
        (60000, 28, 28) -> (60000, 28, 28, 1)
    """
    return np.expand_dims(images, axis=-1)


def add_gaussian_noise(images, noise_factor=0.3, seed=42):
    """
    Add Gaussian noise to normalized images.

    Args:
        images: Images with pixel values in [0, 1].
        noise_factor: Controls the amount of noise.
        seed: Random seed for reproducibility.

    Returns:
        Noisy images clipped to the valid [0, 1] range.
    """
    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=images.shape,
    )

    noisy_images = images + noise_factor * noise

    return np.clip(noisy_images, 0.0, 1.0)