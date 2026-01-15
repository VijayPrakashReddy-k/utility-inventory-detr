#!/bin/bash
# Run Streamlit app from conda environment

# Navigate to streamlit directory
cd "$(dirname "$0")"

# Activate conda environment properly
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-trends

# Use the activated environment's python
python -m streamlit run main.py

