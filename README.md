# Neural Style Transfer (AdaIN)

A from-scratch PyTorch implementation of *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (Huang & Belongie, ICCV 2017), deployed as a Flask web app that blends any content image with any style image in a single forward pass.

## Overview

Classic style transfer (Gatys et al.) optimizes pixel values for every new image pair, which is slow. AdaIN instead aligns the statistics (mean and standard deviation) of the content image's features with those of the style image, enabling arbitrary style transfer in real time with a single trained decoder network.

## How it works

- A pretrained, frozen **VGG19 encoder** extracts multi-level features from both the content and style images
- **Adaptive Instance Normalization** aligns the content feature statistics to the style feature statistics
- A trainable **decoder** (a mirrored VGG architecture) reconstructs an image from these stylized features
- An **alpha parameter** blends between the original content and the fully stylized output, giving control over style strength

## Training

The decoder was trained from scratch on the **COCO2017** dataset for 200 epochs, using a composite loss:
- **Content loss** — MSE between the generated image's features and the AdaIN target
- **Style loss** — MSE between the mean/std of generated and style features, computed across four VGG layers (`relu1-1` to `relu4-1`)

## Tech stack

- Python, PyTorch, Torchvision
- Flask, Flask-WTF, Flask-Bootstrap
- Gunicorn (production serving)

## Project structure

```
├── NST_Code/
│   ├── app.py               # Flask app
│   ├── train.py             # Training script
│   ├── utils/
│   │   ├── models.py         # VGGEncoder and Decoder architectures
│   │   └── utils.py          # AdaIN, dataset loading
│   ├── templates/index.html  # Web UI
│   ├── content_data/         # Sample content images
│   ├── style_data/           # Sample style images
│   └── experiment/           # Training logs and checkpoints
├── Demo_IO_Images/            # Example input/output pairs
└── requirements.txt
```

## Running locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get the model weights**

This repo excludes large model weight files (`.pth`, tracked via `.gitignore`) to keep it lightweight. You'll need:
- `vgg_normalised.pth` — pretrained VGG19 encoder weights
- A trained decoder checkpoint (or train your own with `train.py`)

**3. Run**
```bash
cd NST_Code
python app.py
```

## Training your own decoder

```bash
cd NST_Code
python train.py --content_dir <path_to_content_images> --style_dir <path_to_style_images> --vgg <path_to_vgg_weights> --epochs 200
```
