# ArtGAN WikiArt Classification Project

## Overview
Multi-task art classification using CNN-RNN architecture on the WikiArt dataset.
Classifies paintings by: Style, Artist, Genre.

## Dataset
- Source: WikiArt dataset from ArtGAN (25.4GB)
- Structure: 3 classification tasks with train/val splits
- Format: CSV files with (image_path, class_index) pairs

## Architecture
- Backbone: Pretrained VGG16/ResNet (frozen initially)
- Sequence modeling: LSTM/GRU on spatial features
- Heads: Separate classification heads per task

## Project Structure
```
├── data/           # Dataset and preprocessing
├── models/         # Model definitions
├── train.py        # Training loop
├── evaluate.py     # Evaluation and outlier detection
├── config.py       # Hyperparameters
└── utils/          # Utilities
```

## Commands
```bash
# Install dependencies
uv sync

# Download dataset
python scripts/download_data.py

# Train model
python train.py --task style

# Evaluate
python evaluate.py --checkpoint <path>
```

## Key Decisions
- Use pretrained CNN to leverage ImageNet features
- LSTM processes spatial feature maps as sequences
- Multi-task learning with shared backbone
- Outlier detection via reconstruction error / embedding distance
