import matplotlib.pyplot as plt

from src.data.loader import CLASS_NAMES, load_fashion_mnist


def plot_sample_images(images, labels, sample_count=10):
    """
    Display a selection of Fashion-MNIST images with their class names.
    """
    plt.figure(figsize=(12, 5))

    for index in range(sample_count):
        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index], cmap="gray")
        plt.title(CLASS_NAMES[labels[index]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def print_class_distribution(labels):
    """
    Print the number of samples for each Fashion-MNIST class.
    """
    print()
    print("Training class distribution:")

    for class_index, class_name in enumerate(CLASS_NAMES):
        count = (labels == class_index).sum()
        print(f"{class_index} - {class_name}: {count}")

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
    print(f"First training class: {CLASS_NAMES[y_train[0]]}")
    print(f"Pixel value range: {x_train.min()} - {x_train.max()}")

    print_class_distribution(y_train)
    plot_sample_images(x_train, y_train)


if __name__ == "__main__":
    main()