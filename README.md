# DL Image Denoising

Deep Learning project for image denoising using a **Convolutional Denoising Autoencoder** trained on the **Fashion-MNIST** dataset.

The model learns to reconstruct clean grayscale images from artificially corrupted noisy inputs.

---

## Project Goal

The goal of this project is to implement and evaluate a Denoising Autoencoder capable of removing artificial noise from Fashion-MNIST images.

The learning task can be summarized as:

```text
Noisy Image
    ↓
Encoder
    ↓
Latent Representation
    ↓
Decoder
    ↓
Reconstructed Clean Image
```

During training:

- the **input** is an artificially corrupted image;
- the **target** is the corresponding original clean image.

The model therefore learns a mapping:

```text
Noisy Image → Clean Image
```

---

## Dataset

The project uses the **Fashion-MNIST** dataset provided through TensorFlow/Keras.

Fashion-MNIST contains **70,000 grayscale images** of clothing items and accessories.

Each image has shape:

```text
28 × 28
```

and belongs to one of 10 classes:

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The class labels are used only during dataset exploration.  
The denoising task itself does not require classification labels.

---

## Dataset Split

The original Fashion-MNIST dataset provides:

```text
60,000 training images
10,000 test images
```

For this project, the original training set is further divided into:

```text
54,000 → Training
 6,000 → Validation
10,000 → Final Test
```

The three subsets have different purposes:

- **Training set**: used to update the neural network weights.
- **Validation set**: used to monitor performance during training without updating the weights.
- **Test set**: kept separate until the final evaluation.

This prevents the final test data from influencing model training or validation.

---

## Data Preprocessing

### Normalization

Fashion-MNIST pixels originally have values in the range:

```text
0 - 255
```

They are normalized to:

```text
0.0 - 1.0
```

before being passed to the neural network.

### Channel Dimension

Because the model uses `Conv2D` layers, grayscale images are converted from:

```text
28 × 28
```

to:

```text
28 × 28 × 1
```

where the final dimension represents the single grayscale channel.

### Artificial Noise

Gaussian noise is added to the normalized clean images.

The generated noisy images become the inputs of the autoencoder, while the original normalized images remain the reconstruction targets.

The values are clipped to the valid range:

```text
[0, 1]
```

after noise generation.

Different deterministic random seeds are used for training, validation, and test data to make experiments reproducible while generating independent noise patterns.

---

## Model Architecture

The project implements a **Convolutional Denoising Autoencoder** composed of two main components:

- Encoder
- Decoder

### Encoder

The encoder progressively extracts image features and reduces the spatial representation.

```text
Input
28 × 28 × 1
      ↓
Conv2D - 32 filters - ReLU
      ↓
28 × 28 × 32
      ↓
MaxPooling2D
      ↓
14 × 14 × 32
      ↓
Conv2D - 16 filters - ReLU
      ↓
14 × 14 × 16
      ↓
MaxPooling2D
      ↓
7 × 7 × 16
      ↓
Conv2D - 8 filters - ReLU
      ↓
7 × 7 × 8
```

The final:

```text
7 × 7 × 8
```

tensor represents the **latent representation** of the input image.

The original image contains:

```text
28 × 28 × 1 = 784 values
```

while the latent representation contains:

```text
7 × 7 × 8 = 392 values
```

providing a compact internal representation.

---

### Decoder

The decoder performs the opposite transformation and reconstructs an image from the latent representation.

```text
7 × 7 × 8
      ↓
Conv2D - 16 filters - ReLU
      ↓
7 × 7 × 16
      ↓
UpSampling2D
      ↓
14 × 14 × 16
      ↓
Conv2D - 32 filters - ReLU
      ↓
14 × 14 × 32
      ↓
UpSampling2D
      ↓
28 × 28 × 32
      ↓
Conv2D - 1 filter - Sigmoid
      ↓
28 × 28 × 1
```

The final layer uses a **Sigmoid** activation so that reconstructed pixel values remain in the normalized interval:

```text
[0, 1]
```

---

## Complete Autoencoder

The full network is:

```text
Noisy Image
28 × 28 × 1
      ↓
Encoder
      ↓
Latent Representation
7 × 7 × 8
      ↓
Decoder
      ↓
Reconstructed Image
28 × 28 × 1
```

The complete model contains:

```text
12,201 trainable parameters
```

---

## Training

The autoencoder is trained using:

```text
Optimizer: Adam
Loss: Mean Squared Error (MSE)
Epochs: 10
Batch size: 128
```

For every training example:

```text
Input  = Noisy image
Target = Clean image
```

During training, the model repeatedly performs:

```text
Forward propagation
        ↓
Reconstructed image
        ↓
MSE loss
        ↓
Backpropagation
        ↓
Adam optimizer
        ↓
Weight update
```

The learned information of the neural network is encoded in its trained parameters.

---

## Training and Validation Results

Training and validation losses decrease consistently throughout the 10 training epochs.

A representative final training run produced:

```text
Final training loss:   0.012120
Final validation loss: 0.011988
```

Both curves remain close throughout training and decrease progressively.

This indicates:

- stable learning;
- improvement on both training and validation data;
- no evident overfitting during the selected number of epochs;
- progressive convergence of the optimization process.

The training history plot is generated automatically during training.

---

## Model Persistence

After training, the complete trained Keras model is saved as:

```text
artifacts/denoising_autoencoder.keras
```

This preserves the learned weights and allows the network to be reused without repeating the complete training process.

The model can then be loaded for inference and evaluation.

Generated model artifacts are excluded from Git version control.

---

## Inference

During inference:

```text
Noisy test image
      ↓
Trained Autoencoder
      ↓
Denoised image
```

Unlike training, inference does **not** perform:

- backpropagation;
- optimizer updates;
- weight modifications.

The already learned model parameters are only used to generate predictions.

---

## Qualitative Evaluation

The model is evaluated visually using previously unseen test images.

For each example, the project compares:

```text
Noisy | Denoised | Clean
```

The reconstructed images show:

- strong removal of artificial noise;
- preservation of the main shape of the objects;
- substantially improved visual quality compared with the noisy input.

The reconstructed images can appear slightly smoother or more blurred than the original clean images because the autoencoder compresses the image through its latent representation and optimizes pixel-level reconstruction error.

---

## Quantitative Evaluation

The final model is evaluated on the complete **10,000-image test set**.

Two metrics are used:

### Mean Squared Error — MSE

MSE measures the average squared pixel difference between two images.

```text
Lower MSE = better reconstruction
```

Final results:

```text
Noisy vs Clean MSE:    0.052630
Denoised vs Clean MSE: 0.012121
```

The denoising autoencoder substantially reduces the reconstruction error.

---

### Peak Signal-to-Noise Ratio — PSNR

PSNR evaluates reconstruction quality relative to the clean reference image.

```text
Higher PSNR = better reconstruction
```

Final results:

```text
Noisy vs Clean PSNR:    12.81 dB
Denoised vs Clean PSNR: 19.59 dB
```

The increase in PSNR confirms that the reconstructed images are significantly closer to the clean originals than the noisy inputs.

---

## Final Results

The quantitative results can be summarized as:

| Comparison | MSE ↓ | PSNR ↑ |
|---|---:|---:|
| Noisy vs Clean | 0.052630 | 12.81 dB |
| Denoised vs Clean | **0.012121** | **19.59 dB** |

Both metrics confirm the same result:

> The trained autoencoder successfully removes a significant amount of artificial Gaussian noise and reconstructs images substantially closer to the original clean Fashion-MNIST samples.

---

## Project Structure

```text
DL_Image_Denoising/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── autoencoder.py
│   │
│   └── training/
│       ├── __init__.py
│       └── trainer.py
│
├── scripts/
│   ├── __init__.py
│   ├── explore_dataset.py
│   ├── inspect_model.py
│   ├── train.py
│   └── evaluate.py
│
├── artifacts/
├── outputs/
├── .gitignore
├── README.md
└── requirements.txt
```

### Main Modules

`src/data/`

- Fashion-MNIST loading
- normalization
- train/validation split
- Gaussian noise generation
- grayscale channel preparation

`src/models/`

- encoder architecture
- decoder architecture
- complete denoising autoencoder

`src/training/`

- model compilation
- training pipeline

`src/evaluation/`

- MSE evaluation
- PSNR evaluation
- training history visualization
- denoising result visualization

`scripts/`

- dataset exploration
- model architecture inspection
- model training
- final model evaluation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/FilippoGaravaglia/DL_Image_Denoising.git
cd DL_Image_Denoising
```

Create a Python virtual environment:

```bash
python3.11 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

All commands should be executed from the project root.

### Explore Fashion-MNIST

```bash
python -m scripts.explore_dataset
```

This verifies:

- dataset dimensions;
- pixel range;
- class distribution;
- clean and noisy image samples.

### Inspect Model Architecture

```bash
python -m scripts.inspect_model
```

This prints the architecture and trainable parameters of:

- encoder;
- decoder;
- complete autoencoder.

### Train the Autoencoder

```bash
python -m scripts.train
```

This:

1. loads Fashion-MNIST;
2. creates train, validation, and test subsets;
3. normalizes the images;
4. generates noisy inputs;
5. trains the autoencoder;
6. plots the training history;
7. saves the trained model.

### Evaluate the Trained Model

```bash
python -m scripts.evaluate
```

This:

1. loads the saved trained model;
2. generates predictions for the test set;
3. calculates MSE and PSNR;
4. displays noisy, denoised, and clean images.

---

## Technologies

- Python 3.11
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Conclusion

This project demonstrates the complete Deep Learning workflow for image denoising using a Convolutional Autoencoder:

```text
Dataset exploration
        ↓
Preprocessing
        ↓
Artificial noise generation
        ↓
Encoder design
        ↓
Decoder design
        ↓
Model training
        ↓
Training validation
        ↓
Model persistence
        ↓
Inference
        ↓
Qualitative evaluation
        ↓
Quantitative evaluation
```

The final model substantially improves noisy Fashion-MNIST images, reducing the test MSE from **0.052630 to 0.012121** and increasing the PSNR from **12.81 dB to 19.59 dB**.

The experiment therefore confirms that the Convolutional Denoising Autoencoder successfully learns a representation capable of reconstructing cleaner versions of artificially corrupted images.