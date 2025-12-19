#!/bin/bash
# Deployment script for Utility Inventory Detection Model

set -e

echo "🚀 Utility Inventory Detection Model - Deployment Script"
echo "========================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from streamlit_deployment/ directory"
    exit 1
fi

# Check if model exists
MODEL_PATH="../models/final/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Warning: Model file not found at $MODEL_PATH"
    echo "   Make sure the model has been trained and saved."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if training log exists
if [ ! -f "training_log.json" ]; then
    echo "⚠️  Warning: training_log.json not found"
    echo "   The About page may not display training metrics."
fi

# Deployment options
echo "Select deployment method:"
echo "1) Streamlit Cloud (GitHub + share.streamlit.io)"
echo "2) Docker (local or cloud)"
echo "3) Local Streamlit server"
echo "4) Check deployment readiness"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "📦 Preparing for Streamlit Cloud deployment..."
        echo ""
        echo "Steps to deploy:"
        echo "1. Push your code to GitHub:"
        echo "   cd .."
        echo "   git init"
        echo "   git add ."
        echo "   git commit -m 'Deploy utility inventory detection model'"
        echo "   git remote add origin <your-github-repo-url>"
        echo "   git push -u origin main"
        echo ""
        echo "2. Go to https://share.streamlit.io"
        echo "3. Sign in with GitHub"
        echo "4. Click 'New app'"
        echo "5. Select your repository"
        echo "6. Set Main file path: streamlit_deployment/main.py"
        echo "7. Click 'Deploy'"
        echo ""
        echo "✅ Make sure these files are in your repo:"
        echo "   - streamlit_deployment/main.py"
        echo "   - streamlit_deployment/about_page.py"
        echo "   - streamlit_deployment/requirements.txt"
        echo "   - streamlit_deployment/training_log.json"
        echo "   - models/final/best_model.pth"
        ;;
    2)
        echo ""
        echo "🐳 Building Docker image..."
        cd ..
        if [ -f "streamlit_deployment/Dockerfile" ]; then
            docker build -f streamlit_deployment/Dockerfile -t utility-detr-app .
            echo ""
            echo "✅ Docker image built successfully!"
            echo ""
            echo "To run the container:"
            echo "  docker run -p 8501:8501 utility-detr-app"
            echo ""
            echo "To push to a registry:"
            echo "  docker tag utility-detr-app <your-registry>/utility-detr-app:latest"
            echo "  docker push <your-registry>/utility-detr-app:latest"
        else
            echo "❌ Error: Dockerfile not found"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "🏃 Starting local Streamlit server..."
        echo ""
        # Check if streamlit is installed
        if ! command -v streamlit &> /dev/null; then
            echo "❌ Error: Streamlit not found. Install with: pip install streamlit"
            exit 1
        fi
        
        # Activate conda environment if available
        if command -v conda &> /dev/null; then
            echo "📦 Activating conda environment (if available)..."
            source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
            conda activate ai-trends 2>/dev/null || echo "   (conda env not found, using system Python)"
        fi
        
        echo "🚀 Starting Streamlit..."
        streamlit run main.py
        ;;
    4)
        echo ""
        echo "🔍 Checking deployment readiness..."
        echo ""
        
        # Check required files
        files_ok=true
        
        if [ -f "main.py" ]; then
            echo "✅ main.py found"
        else
            echo "❌ main.py missing"
            files_ok=false
        fi
        
        if [ -f "about_page.py" ]; then
            echo "✅ about_page.py found"
        else
            echo "❌ about_page.py missing"
            files_ok=false
        fi
        
        if [ -f "requirements.txt" ]; then
            echo "✅ requirements.txt found"
        else
            echo "❌ requirements.txt missing"
            files_ok=false
        fi
        
        if [ -f "$MODEL_PATH" ]; then
            echo "✅ Model file found: $MODEL_PATH"
            MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
            echo "   Size: $MODEL_SIZE"
        else
            echo "❌ Model file missing: $MODEL_PATH"
            files_ok=false
        fi
        
        if [ -f "training_log.json" ]; then
            echo "✅ training_log.json found"
        else
            echo "⚠️  training_log.json missing (About page will have limited info)"
        fi
        
        # Check Python dependencies
        echo ""
        echo "📦 Checking Python dependencies..."
        if command -v python3 &> /dev/null; then
            python3 -c "import streamlit; print('✅ streamlit installed')" 2>/dev/null || echo "❌ streamlit not installed"
            python3 -c "import torch; print('✅ torch installed')" 2>/dev/null || echo "❌ torch not installed"
            python3 -c "import torchvision; print('✅ torchvision installed')" 2>/dev/null || echo "❌ torchvision not installed"
            python3 -c "import PIL; print('✅ pillow installed')" 2>/dev/null || echo "❌ pillow not installed"
        else
            echo "⚠️  Python3 not found in PATH"
        fi
        
        echo ""
        if [ "$files_ok" = true ]; then
            echo "✅ All required files are present. Ready for deployment!"
        else
            echo "❌ Some required files are missing. Please fix before deploying."
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✨ Done!"

