# DETR Training for Utility Inventory Detection

This directory contains the training pipeline for the utility inventory detection model.

## Files

- **`train.py`**: Main training script
- **`detr_model.py`**: DETR model architecture (ResNet50 backbone + Transformer)
- **`dataset.py`**: COCO dataset loader with data augmentation
- **`log_training_metrics.py`**: Training metrics logging to JSON
- **`config.yaml`**: Standard training configuration
- **`config_optimized.yaml`**: Optimized configuration (higher bbox weight, earlier LR drop)
- **`logs/training_log.json`**: Training metrics and loss curves

## Quick Start

### 1. Prerequisites

```bash
# Activate conda environment
conda activate ai-trends

# Install dependencies
pip install -r ../requirements-training.txt
```

### 2. Prepare Dataset

Ensure you've run the data processing scripts (see `../scripts/`):

```bash
# From project root
python scripts/01_download_datasets.py
python scripts/00_extract_datasets.py
python scripts/02_clean_datasets.py
python scripts/03_merge_datasets.py
```

The merged dataset should be in `../datasets/processed/merged/` with train/valid/test splits.

### 3. Start Training

```bash
# Standard configuration
python train.py --config config.yaml

# Or optimized configuration (recommended)
python train.py --config config_optimized.yaml
```

## Configuration Files

### `config.yaml` (Standard)

- **Batch size**: 2
- **Learning rate**: 1e-4
- **Epochs**: 50
- **Bbox loss weight**: 5.0
- **LR drop**: Epoch 40

### `config_optimized.yaml` (Recommended)

- **Batch size**: 2
- **Learning rate**: 1e-4
- **Epochs**: 50
- **Bbox loss weight**: 10.0 (higher for better box regression)
- **LR drop**: Epoch 30 (earlier for faster convergence)

## Training Output

### Model Checkpoints

- **Checkpoints**: `../models/checkpoints/checkpoint_epoch_{N}.pth`
- **Best model**: `../models/final/best_model.pth` (lowest validation loss)
- **Final model**: `../models/final/final_model.pth` (last epoch)

### Training Logs

- **JSON log**: `logs/training_log.json`
  - Contains epoch-by-epoch metrics
  - Used by Streamlit About page
  - Includes: losses, learning rate, epoch time, system info

### Metrics Tracked

- **Classification loss**: Object class prediction error
- **Bbox loss**: Bounding box coordinate error
- **GIoU loss**: Generalized IoU for box overlap
- **Total loss**: Weighted sum of all losses
- **Validation loss**: Same metrics on validation set

## Training Tips

### Device Selection

The script automatically detects available devices:
- **MPS** (Apple Silicon GPU) - fastest
- **CUDA** (NVIDIA GPU) - if available
- **CPU** - fallback (very slow)

### Memory Management

- **Batch size 2**: Optimized for DETRdemo architecture
- **Image resize**: 800px short side (configurable in `dataset.py`)
- **Gradient accumulation**: Not implemented, but can be added

### Monitoring Training

```bash
# Watch training log
tail -f logs/training_log.json

# Or use the plotting script
python ../scripts/plot_training_curves.py
```

### Early Stopping

The script saves the best model based on validation loss. You can:
- Stop training early if validation loss plateaus
- Resume from checkpoint if needed (not implemented, but can be added)

## Model Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Transformer**: 6 encoder + 6 decoder layers
- **Hidden dim**: 256
- **Attention heads**: 8
- **Object queries**: 100
- **Classes**: 3 (insulators, crossarm, utility-pole)

## Training Results

### Best Model (Epoch 48)

- **Validation loss**: 0.3410
- **Training time**: 4.16 hours (50 epochs on M3 Max)
- **Device**: MPS (Apple Silicon GPU)

### Loss Breakdown (Final Epoch)

- Classification loss: ~0.15
- Bbox loss: ~0.10
- GIoU loss: ~0.09
- Total loss: ~0.34

## Troubleshooting

### Out of Memory

- Reduce batch size in `config.yaml`
- Reduce image size in `dataset.py` (default: 800px)
- Use CPU instead of GPU (slower but uses less memory)

### Training Too Slow

- Use GPU (MPS or CUDA) instead of CPU
- Reduce number of epochs
- Use optimized config (converges faster)

### Model Not Improving

- Check dataset quality and annotations
- Increase learning rate (carefully)
- Use optimized config (higher bbox weight)
- Train for more epochs

### Import Errors

```bash
# Ensure you're in the correct directory
cd utility-detr/training

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing dependencies
pip install -r ../requirements-training.txt
```

## Advanced Usage

### Custom Configuration

Create your own config file:

```yaml
num_classes: 3
batch_size: 2
num_epochs: 100
learning_rate: 0.0001
weight_decay: 0.0001
lr_drop: 40
bbox_weight: 10.0
giou_weight: 2.0
device: "mps"  # or "cuda" or "cpu"
```

Then run:
```bash
python train.py --config your_config.yaml
```

### Resume Training

Currently not implemented, but you can modify `train.py` to:
1. Load checkpoint
2. Resume from saved epoch
3. Continue training

## Next Steps

After training:
1. **Evaluate model**: Test on test set (not implemented)
2. **Deploy**: Use `best_model.pth` in Streamlit app
3. **Improve**: Adjust config and retrain if needed

## References

- DETR Paper: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- Original DETR: [Facebook Research](https://github.com/facebookresearch/detr)

