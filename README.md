# DETR Utility Inventory Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

Training DETR model to detect: **insulators**, **crossarm**, **utility-pole**

---

## 🎥 Demo Video

**[📹 Watch Demo Video (68MB)](https://github.com/VijayPrakashReddy-k/utility-inventory-detr/blob/main/assets/demo-compressed.mp4)**  
*Click to view in GitHub's video player*

<!-- To embed video directly in README: -->
<!-- 1. Go to: https://github.com/VijayPrakashReddy-k/utility-inventory-detr/issues/new -->
<!-- 2. Drag and drop the video file into a comment -->
<!-- 3. Copy the generated user-attachments URL -->
<!-- 4. Paste it here in this format: -->
<!-- https://user-images.githubusercontent.com/YOUR-USER-ID/YOUR-VIDEO-ID/demo-compressed.mp4 -->

> **✨ Features Shown:**
> - Advanced OBB Editor with rotation support
> - Real-time angle tracking and editing
> - One-click JSON export with rotation angles
> - Visual feedback and professional UI
> - Select/Edit mode for preserving original detection scores
>
> *Video stored with Git LFS (68MB compressed from 197MB)*
---

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
conda activate ai-trends  # or your env

# Install dependencies
pip install -r requirements-training.txt
```

### 2. Prepare Dataset (Run in order)

```bash
cd utility-detr

# Step 1: Download datasets (or download manually from Roboflow)
python scripts/01_download_datasets.py

# Step 2: Extract ZIP files
python scripts/00_extract_datasets.py

# Step 3: Clean datasets
python scripts/02_clean_datasets.py

# Step 4: Merge datasets
python scripts/03_merge_datasets.py
```

### 3. Train Model

```bash
# Start training
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
