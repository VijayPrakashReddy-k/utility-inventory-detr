# DETR Utility Inventory Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

Training DETR model to detect: **insulators**, **crossarm**, **utility-pole**

## Quick Start

```bash
# Activate conda environment
conda activate ai-trends  # or your env

# Start training
cd utility-detr
python training/train.py --config training/config.yaml
```

## Project Structure

```
utility-detr/
├── datasets/processed/merged/  # Final merged dataset (923 images, 3 classes)
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── final/                  # Final trained models
├── training/
│   ├── train.py               # Main training script
│   ├── detr_model.py          # DETR architecture
│   ├── dataset.py             # Data loader
│   └── config.yaml            # Training configuration
└── scripts/                    # Data processing scripts
```

## Dataset

- **Total**: 923 images, 1,024 annotations
- **Classes**: insulators (599), crossarm (207), utility-pole (218)
- **Splits**: Train (713), Valid (134), Test (76)

## Training

- **Time**: 4.16 hours for 50 epochs (M3 Max)
- **Best Model**: Epoch 48 (validation loss: 0.3410)
- **Output**: `models/final/best_model.pth`
- **Device**: MPS (Apple Silicon GPU)

## Streamlit Deployment

```bash
cd streamlit_deployment
conda activate ai-trends
pip install -r requirements.txt
streamlit run main.py
```

See `streamlit_deployment/README.md` for details.

---

**Status**: ✅ Training Complete | ✅ Streamlit App Ready
