#!/bin/bash
# Run Streamlit app from conda environment

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-trends

# Navigate to streamlit directory
cd "$(dirname "$0")"

# Use conda's python to run streamlit (ensures correct environment)
python -m streamlit run main.py

