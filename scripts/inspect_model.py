from src.models.autoencoder import (
    build_autoencoder,
    build_decoder,
    build_encoder,
)


def main():
    encoder = build_encoder()
    decoder = build_decoder()
    autoencoder = build_autoencoder()

    print()
    print("=== ENCODER ===")
    encoder.summary()

    print()
    print("=== DECODER ===")
    decoder.summary()

    print()
    print("=== AUTOENCODER ===")
    autoencoder.summary()


if __name__ == "__main__":
    main()