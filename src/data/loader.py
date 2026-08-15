from tensorflow.keras.datasets import fashion_mnist


def load_fashion_mnist():
    """
    Load the Fashion-MNIST dataset.

    Returns:
        tuple: Training and test images with their corresponding labels.
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    return x_train, y_train, x_test, y_test