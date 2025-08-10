# Enhanced Breast Tumor Segmentation via Preprocessing-Optimized TransUNet

A deep learning project for automated breast tumor segmentation in ultrasound images using a novel preprocessing-optimized TransUNet architecture with reconstruction regularization.

## Overview

This project implements an enhanced version of TransUNet specifically designed for breast tumor segmentation in ultrasound images. The key innovation lies in the integration of image reconstruction as a regularization technique, which helps the model learn better feature representations and improve segmentation accuracy.

### Key Features

- **Enhanced TransUNet Architecture**: Vision Transformer (ViT) backbone with ResNet-50 hybrid encoder
- **Dual-Task Learning**: Simultaneous segmentation and image reconstruction
- **Advanced Data Augmentation**: Comprehensive augmentation pipeline optimized for medical imaging
- **Multi-Scale Feature Fusion**: Skip connections for better fine-grained detail preservation
- **Comprehensive Evaluation**: Dice score metrics with detailed performance analysis

## Architecture

The model combines:
- **Encoder**: ResNet-50 + Vision Transformer (R50-ViT-B_16)
- **Decoder**: Multi-scale decoder with skip connections
- **Segmentation Head**: Binary segmentation output
- **Reconstruction Head**: Image reconstruction for regularization

## Dataset

The project uses the **Breast Ultrasound Images Dataset (BUSI)** from Kaggle:
- **Classes**: Benign and Malignant tumors (Normal cases excluded as they lack masks)
- **Format**: PNG images with corresponding binary masks
- **Split**: 64% Train / 16% Validation / 20% Test

## Quick Start

### Prerequisites

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/m-mutti/breast-tumor-segmentation
cd breast-segmentation
```

2. Install dependencies:
```bash
uv sync
```

### Data Preparation

```bash
# Download and prepare the BUSI dataset
python prepare_data.py
```

This will:
- Download the dataset from Kaggle
- Split into train/validation/test sets
- Organize images and masks into proper directory structure

### Training

```bash
# Basic training
python training.py

# With custom parameters (modify the script)
# - EPOCHS: Number of training epochs (default: 200)
# - ADD_AUGMENTATIONS: Enable/disable augmentations (default: True)
# - EXPERIMENT_NAME: Name for the experiment (default: "reconstruction")
```

### Evaluation

Use the Jupyter notebook for comprehensive evaluation:
```bash
jupyter notebook result-visualization.ipynb
```

## Project Structure

```
breast-segmentation/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── pyproject.toml                # UV project configuration
├── prepare_data.py               # Dataset download and preprocessing
├── training.py                   # Main training script
├── result-visualization.ipynb    # Evaluation and visualization
├── dataset_split/               # Processed dataset
│   ├── train/
│   ├── val/
│   └── test/
└── trained_models/              # Model checkpoints
    └── reconstruction/
        └── [timestamp]/
            └── best_model.pth
```

## Configuration

### Model Configurations Available:
- `ViT-B_16`: Vision Transformer Base with 16x16 patches
- `ViT-B_32`: Vision Transformer Base with 32x32 patches  
- `ViT-L_16`: Vision Transformer Large with 16x16 patches
- `R50-ViT-B_16`: ResNet-50 + ViT-Base hybrid (default)
- `R50-ViT-L_16`: ResNet-50 + ViT-Large hybrid

### Key Hyperparameters:
- **Input Size**: 224x224 pixels
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Scheduler**: Cosine Annealing
- **Loss Function**: BCE + Dice Loss + Reconstruction Loss
- **Loss Weights**: alpha=1.0 (segmentation), beta=0.1 (reconstruction)

## Training Features

- **Mixed Loss Function**: Combines Binary Cross Entropy, Dice Loss, and MSE Reconstruction Loss
- **Advanced Augmentations**: Horizontal/vertical flips, 90° rotations, normalization
- **Cosine Annealing Scheduler**: Smooth learning rate decay
- **Early Stopping**: Saves best model based on validation loss
- **WandB Integration**: Real-time training monitoring and visualization

## Evaluation Metrics

- **Dice Score**: Primary segmentation metric
- **Binary Cross Entropy**: Classification loss
- **Reconstruction MSE**: Image reconstruction quality
- **Per-image Analysis**: Individual Dice scores for detailed performance assessment

## Experimental Results

The model architecture supports various configurations for ablation studies:
- Base TransUNet vs. Reconstruction-enhanced TransUNet
- Different augmentation strategies
- Various loss weighting schemes
- Multiple backbone architectures

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (training tested on CUDA-enabled systems)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for dataset and model checkpoints

## Dependencies

Key libraries:
- PyTorch & TorchVision
- Albumentations (data augmentation)
- OpenCV (image processing)
- Weights & Biases (experiment tracking)
- Matplotlib (visualization)
- Scikit-learn (data splitting)
- KaggleHub (dataset download)

See `requirements.txt` for complete list.

## Usage Notes

- **CUDA Support**: The project automatically detects GPU availability
- **Reproducibility**: Fixed random seeds for consistent results
- **Monitoring**: WandB integration for experiment tracking
- **Checkpointing**: Automatic model saving based on validation performance

## Medical AI Compliance

This research tool is designed for:
- Academic research and educational purposes
- Algorithm development and validation
- Performance benchmarking

**Note**: This is a research implementation and should not be used for clinical diagnosis without proper validation and regulatory approval.

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{enhanced-breast-segmentation,
  title={Enhanced Breast Tumor Segmentation via Preprocessing-Optimized TransUNet},
  author={[Your Name]},
  year={2024},
  url={https://github.com/m-mutti/breast-tumor-segmentation}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch  
5. Create a Pull Request

## Contact

[Add your contact information here]

---

**Keywords**: Medical Image Segmentation, TransUNet, Vision Transformer, Breast Ultrasound, Deep Learning, PyTorch