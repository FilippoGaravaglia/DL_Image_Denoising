from tensorflow.keras import Model


def compile_autoencoder(model: Model):
    """
    Compile the denoising autoencoder.

    Args:
        model: Autoencoder model to compile.
    """
    model.compile(
        optimizer="adam",
        loss="mse",
    )


def train_autoencoder(
    model: Model,
    x_train_noisy,
    x_train_clean,
    x_validation_noisy,
    x_validation_clean,
    epochs=10,
    batch_size=128,
):
    """
    Train the denoising autoencoder.

    Args:
        model: Compiled autoencoder.
        x_train_noisy: Noisy training images.
        x_train_clean: Clean training images.
        x_validation_noisy: Noisy validation images.
        x_validation_clean: Clean validation images.
        epochs: Number of training epochs.
        batch_size: Number of images processed before a weight update.

    Returns:
        Training history produced by Keras.
    """
    history = model.fit(
        x=x_train_noisy,
        y=x_train_clean,
        validation_data=(
            x_validation_noisy,
            x_validation_clean,
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
    )

    return history