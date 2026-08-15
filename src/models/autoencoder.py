from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D


def build_encoder(input_shape=(28, 28, 1)):
    """
    Build the convolutional encoder used by the denoising autoencoder.

    Args:
        input_shape: Shape of a single input image.

    Returns:
        Keras Model representing the encoder.
    """
    inputs = Input(shape=input_shape, name="encoder_input")

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="encoder_conv_1",
    )(inputs)

    x = MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name="encoder_pool_1",
    )(x)

    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="encoder_conv_2",
    )(x)

    x = MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name="encoder_pool_2",
    )(x)

    latent = Conv2D(
        filters=8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="latent_representation",
    )(x)

    return Model(
        inputs=inputs,
        outputs=latent,
        name="encoder",
    )


def build_decoder(latent_shape=(7, 7, 8)):
    """
    Build the convolutional decoder used by the denoising autoencoder.

    Args:
        latent_shape: Shape of the latent representation produced by the encoder.

    Returns:
        Keras Model representing the decoder.
    """
    inputs = Input(shape=latent_shape, name="decoder_input")

    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="decoder_conv_1",
    )(inputs)

    x = UpSampling2D(
        size=(2, 2),
        name="decoder_upsampling_1",
    )(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="decoder_conv_2",
    )(x)

    x = UpSampling2D(
        size=(2, 2),
        name="decoder_upsampling_2",
    )(x)

    outputs = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same",
        name="decoder_output",
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="decoder",
    )


def build_autoencoder(input_shape=(28, 28, 1)):
    """
    Build the complete denoising autoencoder.

    Args:
        input_shape: Shape of a single input image.

    Returns:
        Keras Model representing the complete denoising autoencoder.
    """
    encoder = build_encoder(input_shape)

    decoder = build_decoder(
        latent_shape=encoder.output_shape[1:]
    )

    inputs = Input(
        shape=input_shape,
        name="autoencoder_input",
    )

    latent = encoder(inputs)
    outputs = decoder(latent)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="denoising_autoencoder",
    )