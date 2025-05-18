# Leaf Unfolder ML

A deep learning project that uses diffusion models to learn the mapping between folded and straightened leaves. This project aims to automatically straighten folded leaves in images using a U-Net based diffusion model.

## Project Structure

```
leaf-unfolder-ml/
├── data/                      # Dataset directory
│   └── fotos hojas bromelias/ # Leaf images
├── example_predictions/       # Generated predictions during training
├── lightning_logs/           # Training logs
├── checkpoints/              # Model checkpoints
├── run_experiment.py         # Main training script
├── visualization.py          # Visualization utilities
└── requirements.txt          # Project dependencies
```

## Setup

### 1. Create Conda Environment

```bash
# Create a new conda environment
conda create -n leaf-unfolder-ml python=3.9
conda activate leaf-unfolder-ml

# Install required packages
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset should be placed in the `data/fotos hojas bromelias/` directory. The dataset contains pairs of images:
- Folded leaves: `{leafname}.JPG`
- Straightened leaves: `{leafname}F_desdoblada.jpg`

## Training

To train the model with default parameters:

```bash
python run_experiment.py
```

### Hyperparameters

You can customize the training by passing different hyperparameters:

```bash
python run_experiment.py \
    --learning_rate 1e-4 \
    --n_steps 1000 \
    --beta_start 1e-4 \
    --beta_end 0.02 \
    --batch_size 4 \
    --max_epochs 100 \
    --save_examples_every_n_epochs 5 \
    --folded_size 128 \
    --straight_size 256 \
    --padding 300
```

Available hyperparameters:

#### Model Hyperparameters
- `--learning_rate`: Learning rate for the model (default: 1e-4)
- `--n_steps`: Number of diffusion steps (default: 1000)
- `--beta_start`: Starting beta value for noise schedule (default: 1e-4)
- `--beta_end`: Ending beta value for noise schedule (default: 0.02)

#### Training Hyperparameters
- `--batch_size`: Batch size for training (default: 4)
- `--max_epochs`: Maximum number of training epochs (default: 100)
- `--save_examples_every_n_epochs`: Save example predictions every N epochs (default: 5)

#### Data Hyperparameters
- `--folded_size`: Size of folded leaf images (default: 128)
- `--straight_size`: Size of straightened leaf images (default: 256)
- `--padding`: Padding for leaf cropping (default: 300)

### Experiment Organization

Each training run creates a unique experiment name based on the hyperparameters and timestamp. The outputs are organized as follows:

```
example_predictions/
└── YYYYMMDD_HHMMSS_lr1e-4_steps1000_bs4_folded128_straight256/
    └── epoch_*.png

checkpoints/
└── YYYYMMDD_HHMMSS_lr1e-4_steps1000_bs4_folded128_straight256/
    └── leaf-unfolding-*.ckpt

lightning_logs/
└── YYYYMMDD_HHMMSS_lr1e-4_steps1000_bs4_folded128_straight256/
    └── events.out.tfevents.*
```

## Model Architecture

The model uses a U-Net architecture with:
- Time embedding for the diffusion process
- Residual blocks with skip connections
- Downsampling and upsampling paths
- Group normalization and SiLU activation

## Visualization

The `visualization.py` script provides utilities for:
- Loading and preprocessing leaf images
- Cropping and padding images to square format
- Displaying pairs of folded and straightened leaves

## Example Predictions

During training, example predictions are saved in the `example_predictions/` directory. Each example shows:
1. Input folded leaf
2. Target straightened leaf
3. Progressive denoising steps from noise to prediction

## Requirements

- Python 3.9+
- PyTorch
- PyTorch Lightning
- OpenCV
- Matplotlib
- NumPy
- CUDA-compatible GPU (recommended)
