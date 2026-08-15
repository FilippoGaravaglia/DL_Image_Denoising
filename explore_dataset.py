from src.data.loader import load_fashion_mnist


def main():
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    print("Fashion-MNIST loaded successfully.")
    print()
    print(f"Training images shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test images shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print()
    print(f"First training label: {y_train[0]}")
    print(f"Pixel value range: {x_train.min()} - {x_train.max()}")


if __name__ == "__main__":
    main()