import numpy as np
import tensorflow as tf


def calculate_mse(reference_images, compared_images):
    """
    Calculate the Mean Squared Error between two sets of images.

    Lower values indicate more similar images.

    Args:
        reference_images: Ground-truth clean images.
        compared_images: Images to compare against the reference.

    Returns:
        Mean Squared Error as a float.
    """
    error = reference_images - compared_images

    return float(
        np.mean(np.square(error))
    )


def calculate_psnr(reference_images, compared_images):
    """
    Calculate the average Peak Signal-to-Noise Ratio.

    Higher values indicate better image quality.

    Args:
        reference_images: Ground-truth clean images.
        compared_images: Images to compare against the reference.

    Returns:
        Average PSNR value in decibels.
    """
    psnr_values = tf.image.psnr(
        reference_images,
        compared_images,
        max_val=1.0,
    )

    return float(
        tf.reduce_mean(psnr_values).numpy()
    )