from src.models.autoencoder import build_encoder


def main():
    encoder = build_encoder()

    encoder.summary()


if __name__ == "__main__":
    main()