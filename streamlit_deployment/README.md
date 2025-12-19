# Utility Inventory DETR - Streamlit Deployment

Streamlit web app for detecting utility inventory (insulators, crossarms, utility poles) using custom trained DETR model.

## Quick Start

**IMPORTANT**: You MUST use the conda environment where PyTorch is installed!

```bash
# Activate conda environment (where torch is installed)
conda activate ai-trends

# Navigate to streamlit directory
cd utility-detr/streamlit_deployment

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run app
streamlit run main.py
```

**OR use the helper script (recommended):**
```bash
cd utility-detr/streamlit_deployment
./run.sh
```

**OR use conda's python directly:**
```bash
conda activate ai-trends
cd utility-detr/streamlit_deployment
python -m streamlit run main.py
```

App opens at `http://localhost:8501`

**If you get import errors**, make sure you activated the conda environment first!

## Usage

1. **Upload Image**: Select JPG/PNG image of utility infrastructure
2. **Adjust Threshold**: Use sidebar slider (recommended: 0.50-0.70)
3. **View Results**: Detection boxes, statistics, and raw data table

## Model Info

- **Classes**: insulators, crossarm, utility-pole
- **Best Model**: Epoch 48 (val loss: 0.3410)
- **Location**: `../models/final/best_model.pth`
- **Training**: 50 epochs on 923 images

## Troubleshooting

**Import errors?** Use conda environment: `conda activate ai-trends`

**Model not found?** Ensure training completed: `ls ../models/final/best_model.pth`

**Memory issues?** Use smaller images (< 1600x1600 pixels)

## Deployment

### Quick Deploy Options

**1. Streamlit Cloud (Recommended - Free)**
- Push code to GitHub
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect repository and deploy
- See `DEPLOYMENT.md` for detailed steps

**2. Docker**
```bash
cd utility-detr
docker build -f streamlit_deployment/Dockerfile -t utility-detr-app .
docker run -p 8501:8501 utility-detr-app
```

**3. Interactive Deployment Script**
```bash
cd utility-detr/streamlit_deployment
./deploy.sh
```

For detailed deployment instructions, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Files

- `main.py` - Streamlit app
- `about_page.py` - About page with training metrics
- `requirements.txt` - Python dependencies
- `DEPLOYMENT.md` - Complete deployment guide
- `Dockerfile` - Docker container configuration
- `deploy.sh` - Interactive deployment script
