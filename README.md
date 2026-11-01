# Brain Tumor Segmentation using U-Net

An educational deep-learning project that trains a U-Net model to segment tumor regions in brain MRI images stored in HDF5 files.

> **Important:** This is a learning/research project, not a medical device. Predictions must not be used for diagnosis or treatment decisions.

## Features

- HDF5 dataset loader (`x` = MRI image, `y` = segmentation mask)
- U-Net encoder-decoder architecture with skip connections
- Multi-class segmentation (background + three foreground labels)
- Combined Cross-Entropy and Dice loss
- Validation and per-class Dice/IoU metrics
- Prediction visualization and Streamlit demo

## Dataset format

Place your dataset under `data/`. The loader searches recursively for `.h5` and `.hdf5` files. Every file must contain:

- `x`: MRI image array
- `y`: segmentation mask array

The default label mapping is `0 -> background`, `50 -> class 1`, `100 -> class 2`, `150 -> class 3`. If your dataset uses different values, edit `RAW_TO_CLASS` in `src/dataset.py`.

Example:
```text
data/
└── training/
    ├── sample_001.h5
    ├── sample_002.h5
    └── ...
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

For a GPU-enabled PyTorch installation, install the build appropriate for your CUDA version from the official PyTorch installation selector.

## Train

From the project root:

```bash
python src/train.py --data_dir data --epochs 30 --batch_size 8
```

The best checkpoint is saved to `models/best_model.pth`.

If your computer has limited memory, try:
```bash
python src/train.py --data_dir data --epochs 10 --batch_size 2 --base_channels 16
```

## Evaluate

```bash
python src/evaluate.py --data_dir data --checkpoint models/best_model.pth
```

For a scientifically valid final evaluation, use a separate held-out test folder rather than evaluating on training data.

## Predict on one MRI image

```bash
python src/predict.py --image path/to/mri.png
```

The visualization is saved to `outputs/prediction.png`.

## Run the demo

```bash
streamlit run app.py
```

Upload a grayscale MRI image and select the trained checkpoint.

## Google Colab

1. Upload this project folder or clone your GitHub repository.
2. Upload/mount your HDF5 dataset.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run the training command from a notebook cell, for example:
   `!python src/train.py --data_dir /content/drive/MyDrive/brain_data --epochs 30 --batch_size 8`
5. Download `models/best_model.pth` after training.

## Model overview

The encoder extracts image features using convolution blocks and max pooling. The bottleneck learns deeper representations. The decoder upsamples feature maps and concatenates encoder features through skip connections. A final 1x1 convolution produces one score map per class.

## Limitations and checks before training

- Confirm the HDF5 keys and tensor shapes match `x` and `y`.
- Inspect a few image/mask pairs before a long training run.
- Confirm the label values in your actual dataset. The default mapping is based on the expected values `0, 50, 100, 150`.
- The included loader uses a single image channel. Multi-modal MRI data requires adapting the loader and model input channels.
- Random file-level splitting may not be appropriate if multiple slices from the same patient exist. Prefer patient-level splitting to prevent data leakage.
- This starter project resizes images to 240x240; change `--image_size` if needed.

## Suggested GitHub description

A PyTorch-based U-Net project for multi-class brain MRI tumor segmentation, including HDF5 data loading, model training, Dice/IoU evaluation, prediction visualizations, and a Streamlit demo.
