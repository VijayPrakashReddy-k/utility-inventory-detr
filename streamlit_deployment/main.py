# ================== Streamlit Deployment for Utility Inventory DETR ==================
# - Custom trained DETR model for utility inventory detection
# - Classes: insulators, crossarm, utility-pole
# - Optimized for CPU/MPS deployment
# ================================================================================

import os
import sys

# Fix OpenMP conflict (same as training script)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check if we're in the right environment - show helpful error if not
# CRITICAL: Don't exit - show error in Streamlit instead
try:
    import torch
except ImportError:
    # Will be handled in main() function
    torch = None

# Hard caps to prevent runaway threading / RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import gc
import base64
import itertools
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import matplotlib - required for visualization
# CRITICAL: Don't exit - show error in Streamlit instead
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend for Streamlit
    import matplotlib.pyplot as plt
except ImportError:
    # Will be handled in main() function
    plt = None

import torchvision.transforms as T

# Add parent directory to path to import detr module
# Get absolute path to training directory
current_file = Path(__file__).resolve()
training_dir = current_file.parent.parent / "training"
training_path = str(training_dir)
if training_path not in sys.path:
    sys.path.insert(0, training_path)

# Import DETR model - try multiple methods
# CRITICAL: Don't exit on failure - handle gracefully in functions
DETRdemo = None
try:
    from detr_model import DETRdemo
except ImportError:
    # Try alternative: add workspace root to path
    workspace_root = current_file.parent.parent.parent
    alt_path = str(workspace_root / "utility-detr" / "training")
    if alt_path not in sys.path:
        sys.path.insert(0, alt_path)
    try:
        from detr_model import DETRdemo
    except ImportError:
        # Last resort: try direct import from training
        import importlib.util
        model_file = training_dir / "detr_model.py"
        if model_file.exists():
            spec = importlib.util.spec_from_file_location("detr_model", model_file)
            if spec and spec.loader:
                detr_model = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(detr_model)
                DETRdemo = detr_model.DETRdemo

# Torch runtime knobs - only set if torch is available
if torch is not None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

# --- UI tweaks ---
# Note: st.set_option() for deprecation warnings removed in newer Streamlit versions

# Default color palette (used if seaborn not available)
DEFAULT_COLORS = [
    (0.000, 0.447, 0.741),  # Blue
    (0.850, 0.325, 0.098),  # Red
    (0.929, 0.694, 0.125),  # Yellow
    (0.494, 0.184, 0.556),  # Purple
    (0.466, 0.674, 0.188),  # Green
    (0.301, 0.745, 0.933),  # Cyan
]

# Use seaborn palette if available, otherwise use default
if HAS_SEABORN:
    palette = itertools.cycle(sns.color_palette())
else:
    palette = itertools.cycle(DEFAULT_COLORS)

# Utility Inventory class names (matching training dataset)
# CRITICAL: Model was trained with category IDs 1, 2, 3 directly
# Model outputs: [0, 1, 2, 3] where:
#   - Index 0 = unused/background (not trained)
#   - Index 1 = insulators (category ID 1)
#   - Index 2 = crossarm (category ID 2)
#   - Index 3 = utility-pole (category ID 3) - but this is also "no object" class
# We exclude index 3 as "no object", so we use indices 0, 1, 2
# But the model learned: 1=insulators, 2=crossarm, 3=utility-pole
# So we need to map: model_output - 1 to get correct class
UTILITY_CLASSES = [
    'insulators',      # category ID 1 → model output index 1
    'crossarm',        # category ID 2 → model output index 2
    'utility-pole'     # category ID 3 → model output index 3 (with stricter filtering)
]

# ---- background image helper (cached) ----
@st.cache_data(show_spinner=False)
def get_base64_of_file(path: str) -> str:
    """Load and encode background image."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def set_page_bg(png_path: str) -> None:
    """Set page background image."""
    try:
        bin_str = get_base64_of_file(png_path)
        if bin_str:
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{bin_str}");
                    background-size: cover;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        pass  # background is optional

# CRITICAL: Don't check background at import time - move to function
# Background image check removed from top-level to prevent file I/O on startup

# ---- transforms ----
# Keep short/long side ~800 to limit memory usage
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- box helpers ----
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center-size format to xyxy format."""
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=1)

def rescale_bboxes(out_bbox: torch.Tensor, size) -> torch.Tensor:
    """Rescale bounding boxes from normalized [0,1] cxcywh to image pixel coordinates xyxy."""
    # out_bbox is in normalized [0,1] cxcywh format
    # Convert to xyxy first, then scale to pixel coordinates
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)  # Convert to normalized xyxy [0,1]
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b * scale  # Scale to pixel coordinates

# ---- drawing helpers ----
def draw_boxes_matplotlib(pil_img, boxes, labels, scores, color_cycle=None, title=None):
    """Draw bounding boxes on image using matplotlib."""
    # local default palette if none is provided
    if color_cycle is None:
        color_cycle = [
            (0.000, 0.447, 0.741),  # Blue
            (0.850, 0.325, 0.098),  # Red
            (0.929, 0.694, 0.125),  # Yellow
            (0.494, 0.184, 0.556),  # Purple
            (0.466, 0.674, 0.188),  # Green
            (0.301, 0.745, 0.933),  # Cyan
        ]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pil_img)

    # repeat colors to cover all boxes
    colors = color_cycle * 100
    for (xmin, ymin, xmax, ymax), name, sc, c in zip(boxes, labels, scores, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2.5))
        ax.text(xmin, ymin, f"{name}: {sc:.2f}", fontsize=13,
                bbox=dict(facecolor="yellow", alpha=0.5))

    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.0)
    return fig

# ---- DETR pipeline ----
@st.cache_resource(show_spinner="Loading model (first time may take 30-60 seconds)...")
def load_utility_detr_model():
    """
    Load the custom trained utility inventory DETR model.
    
    CRITICAL: This function is lazy-loaded - model only loads when first needed.
    This prevents Streamlit Cloud timeout on startup.
    """
    # Force CPU-only mode for Streamlit Cloud compatibility
    device = torch.device("cpu")
    
    # Path to trained model - use absolute path
    current_file = Path(__file__).resolve()
    model_path = current_file.parent.parent / "models" / "final" / "best_model.pth"
    
    # Try alternative paths if first doesn't work
    alternative_paths = [
        model_path,  # Primary: relative to script
        Path("../models/final/best_model.pth").resolve(),  # Relative to cwd
        Path("models/final/best_model.pth").resolve(),  # From cwd
    ]
    
    # Find the first existing path
    model_path = None
    for path in alternative_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        st.error("❌ Model not found!")
        st.info(f"**Script location:** `{current_file}`")
        st.info(f"**Working directory:** `{Path.cwd()}`")
        st.info(f"**Expected locations:**")
        for i, path in enumerate(alternative_paths, 1):
            st.info(f"  {i}. `{path}` {'✅' if path.exists() else '❌'}")
        st.info("\n**Solution:**")
        st.info("1. Ensure training completed: `python training/train.py --config training/config_optimized.yaml`")
        st.info("2. Check model exists: `ls models/final/best_model.pth`")
        return None
    
    # Create model with 3 classes (insulators, crossarm, utility-pole)
    model = DETRdemo(
        num_classes=3,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    
    # Load trained weights - CRITICAL: Always use CPU for Streamlit Cloud
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Show model info if available
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            epoch = checkpoint.get('epoch', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
            st.success(f"✓ Loaded trained model (Epoch {epoch}, Val Loss: {val_loss:.4f})")
        else:
            st.success(f"✓ Loaded trained model from {model_path.name}")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.exception(e)
        return None

def infer_utility_detr(pil_img: Image.Image, model, prob_thresh: float):
    """Run inference on image using utility inventory DETR model."""
    if model is None:
        return pil_img, [], [], [], []
    
    img_rgb = pil_img.convert("RGB")
    original_size = img_rgb.size  # (width, height) - original image size
    orig_w, orig_h = original_size
    
    # Apply the transform to get the actual transformed image
    # T.Resize(800) resizes short side to 800, maintaining aspect ratio
    tensor = transform(img_rgb).unsqueeze(0)
    
    # Get ACTUAL transformed image size from tensor
    # Tensor shape: [batch, channels, height, width]
    tensor_h, tensor_w = tensor.shape[-2], tensor.shape[-1]
    transformed_w = tensor_w  # width
    transformed_h = tensor_h   # height
    transformed_size = (transformed_w, transformed_h)  # (width, height)
    
    # Bound absolute size to avoid huge memory spikes
    assert tensor.shape[-2] <= 1600 and tensor.shape[-1] <= 1600, \
        "Image too large (limit 1600x1600). Please upload a smaller image."

    with torch.inference_mode():
        outputs = model(tensor)

    # Process predictions
    # DETR outputs: pred_logits [batch, 100, num_classes+1], pred_boxes [batch, 100, 4]
    # Model was trained with category IDs 1, 2, 3, so outputs are:
    #   Index 0: background/"no object"
    #   Index 1: insulators (category ID 1)
    #   Index 2: crossarm (category ID 2)
    #   Index 3: utility-pole (category ID 3) - Note: requires stricter filtering to avoid false positives
    pred_logits = outputs["pred_logits"][0]  # [100, 4]
    pred_boxes = outputs["pred_boxes"][0]    # [100, 4] (normalized cxcywh)
    
    # Apply softmax to get probabilities
    probas = pred_logits.softmax(-1)  # [100, 4]
    
    # Process all classes: insulators (1), crossarm (2), utility-pole (3)
    # CRITICAL: Index 3 (utility-pole) conflicts with DETR's "no object" class
    # This causes utility-pole to be over-predicted. We need VERY strict filtering.
    
    background_probs = probas[:, 0]  # [100] - background/"no object"
    insulators_probs = probas[:, 1]  # [100] - insulators
    crossarm_probs = probas[:, 2]    # [100] - crossarm
    utility_pole_probs = probas[:, 3]  # [100] - utility-pole (problematic!)
    
    # CRITICAL: First, filter out queries where background has the highest probability
    # If background is the best class, that query represents "no object" - ignore it completely
    max_probs_all_classes, best_class_all = probas.max(dim=1)  # [100], [100] - includes background
    is_object_query = (best_class_all != 0)  # Only consider queries where an object class wins (not background)
    
    # Find the best OBJECT class for each query (excluding background)
    class_probs_all = probas[:, 1:4]  # [100, 3] - all three object classes
    max_probs_all, class_ids_all = class_probs_all.max(dim=1)  # [100], [100] (0, 1, or 2)
    class_ids_all = class_ids_all + 1  # Map to 1, 2, or 3 (insulators, crossarm, utility-pole)
    
    # Class-specific thresholds (higher = stricter)
    insulators_thresh = prob_thresh
    crossarm_thresh = prob_thresh
    utility_pole_thresh = max(prob_thresh, 0.75)  # High threshold for utility-pole (0.75 minimum)
    
    # Filter insulators: must be an object query, meet threshold, and be better than other classes
    insulators_valid = (
        is_object_query &  # Must be an object query (not background)
        (class_ids_all == 1) &  # Predicted as insulators
        (insulators_probs > insulators_thresh) &  # Above threshold
        (insulators_probs > crossarm_probs + 0.05) &  # 5% better than crossarm
        (insulators_probs > utility_pole_probs + 0.05)  # 5% better than utility-pole
    )
    
    # Filter crossarm: must be an object query, meet threshold, and be better than other classes
    crossarm_valid = (
        is_object_query &  # Must be an object query (not background)
        (class_ids_all == 2) &  # Predicted as crossarm
        (crossarm_probs > crossarm_thresh) &  # Above threshold
        (crossarm_probs > insulators_probs + 0.05) &  # 5% better than insulators
        (crossarm_probs > utility_pole_probs + 0.05)  # 5% better than utility-pole
    )
    
    # Filter utility-pole: EXTREMELY strict (it's often confused with "no object")
    # CRITICAL: Index 3 is used for BOTH utility-pole AND "no object" in DETR
    # TEMPORARY FIX: Disable utility-pole completely until we can retrain with better class mapping
    # The model learned that index 3 = "no object", so it's too ambiguous
    utility_pole_valid = torch.zeros_like(is_object_query, dtype=torch.bool)  # Disable utility-pole for now
    
    # Alternative: If you want to enable it, use VERY strict conditions:
    # utility_pole_valid = (
    #     is_object_query &  # Must be an object query (not background)
    #     (class_ids_all == 3) &  # Predicted as utility-pole
    #     (utility_pole_probs > 0.9) &  # Very high absolute threshold (0.9)
    #     (utility_pole_probs > insulators_probs + 0.4) &  # 40% better than insulators
    #     (utility_pole_probs > crossarm_probs + 0.4) &  # 40% better than crossarm
    #     (background_probs < 0.1)  # Background must be extremely low (< 0.1)
    # )
    
    # Combine all valid detections
    valid_detection = insulators_valid | crossarm_valid | utility_pole_valid
    
    # Use the class predictions
    final_probs = max_probs_all
    final_class_ids = class_ids_all
    
    # Assign to variables for use in filtering (these are used later)
    class_ids = final_class_ids
    max_probs = final_probs
    
    # Apply the combined filtering
    keep = valid_detection
    
    # Apply Non-Maximum Suppression (NMS) to remove duplicate/overlapping detections
    # Limit to top predictions per class to ensure diversity
    if keep.sum() > 0:
        keep_indices = torch.where(keep)[0]
        if len(keep_indices) > 0:
            # Group by class and keep top 3 per class
            class_ids_kept = class_ids[keep_indices]
            unique_classes = torch.unique(class_ids_kept)
            
            final_keep_indices = []
            for cls_id in unique_classes:
                # Get indices for this class
                cls_mask = (class_ids_kept == cls_id)
                cls_indices = keep_indices[cls_mask]
                cls_probs = final_probs[cls_indices]
                
                # Keep only top 1 per class (since there's only one object per class)
                top_cls_index = cls_indices[cls_probs.argsort(descending=True)[0]]
                final_keep_indices.append(top_cls_index.item())
            
            # Create new keep mask
            keep = torch.zeros_like(keep)
            keep[torch.tensor(final_keep_indices, device=keep.device)] = True
    
    if keep.sum() == 0:
        # Debug: show what the model is predicting
        top_k = 5
        top_indices = max_probs.topk(min(top_k, len(max_probs))).indices
        debug_info = []
        for idx in top_indices:
            cls_id = class_ids[idx].item()
            prob = max_probs[idx].item()
            class_idx = cls_id - 1 if cls_id > 0 else 0
            class_name = UTILITY_CLASSES[class_idx] if 0 <= class_idx < len(UTILITY_CLASSES) else f"unknown_{cls_id}"
            debug_info.append(f"Query {idx}: {class_name} (class_id={cls_id}, prob={prob:.3f})")
        return img_rgb, [], [], [], debug_info
    
    # Get filtered predictions
    boxes_norm = pred_boxes[keep]  # Normalized boxes [N, 4] in cxcywh format [0,1]
    scores = max_probs[keep].cpu().numpy().tolist()
    class_ids_filtered = class_ids[keep].cpu().numpy()  # Keep as numpy array
    
    # DEBUG: Print what we're getting
    # print(f"DEBUG: keep.sum()={keep.sum()}, class_ids_filtered={class_ids_filtered}")
    
    # CRITICAL: Model outputs indices 1, 2, 3 (category IDs)
    # Map to class names: 1=insulators, 2=crossarm, 3=utility-pole
    # But our UTILITY_CLASSES array is 0-indexed, so subtract 1
    class_indices = (class_ids_filtered - 1).tolist()  # Convert 1,2,3 → 0,1,2
    
    # DEBUG: Verify mapping
    # print(f"DEBUG: class_ids_filtered={class_ids_filtered}, class_indices={class_indices}")
    
    # CRITICAL FIX: Boxes are normalized to TRANSFORMED image size (800px short side)
    # The model outputs boxes in normalized cxcywh [0,1] relative to transformed image
    # We need to scale them to original image coordinates using the correct scale factors
    
    orig_w, orig_h = original_size
    
    # Convert from normalized cxcywh [0,1] to xyxy [0,1] (relative to transformed image)
    boxes_xyxy_norm = box_cxcywh_to_xyxy(boxes_norm)  # [N, 4] in [0,1] range
    
    # Calculate scale factors: how much larger is the original image compared to transformed
    scale_x = orig_w / transformed_w
    scale_y = orig_h / transformed_h
    
    # Scale boxes from normalized [0,1] (relative to transformed) to original pixel coordinates
    # We do this in one step: normalized * transformed_size * (original_size / transformed_size)
    # Which simplifies to: normalized * original_size (but only if boxes were normalized to original)
    # Since boxes are normalized to transformed, we need: normalized * transformed_size * scale_factor
    
    boxes = []
    for box in boxes_xyxy_norm:
        x1_norm, y1_norm, x2_norm, y2_norm = box.unbind(0)
        
        # First scale to transformed pixel coordinates
        x1_trans = x1_norm.item() * transformed_w
        y1_trans = y1_norm.item() * transformed_h
        x2_trans = x2_norm.item() * transformed_w
        y2_trans = y2_norm.item() * transformed_h
        
        # Then scale to original pixel coordinates
        x1_orig = x1_trans * scale_x
        y1_orig = y1_trans * scale_y
        x2_orig = x2_trans * scale_x
        y2_orig = y2_trans * scale_y
        
        boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
    
    # Clamp boxes to image boundaries and ensure valid coordinates (x2 > x1, y2 > y1)
    boxes = [[max(0, min(b[0], orig_w)), max(0, min(b[1], orig_h)), 
              max(b[0] + 1, min(b[2], orig_w)), max(b[1] + 1, min(b[3], orig_h))] for b in boxes]
    
    # Map class indices to class names
    labels = []
    for i in class_indices:
        if 0 <= i < len(UTILITY_CLASSES):
            labels.append(UTILITY_CLASSES[i])
        else:
            labels.append(f"unknown_{i}")
    
    # DEBUG: Show what classes we detected
    # st.write(f"DEBUG: Detected {len(labels)} objects: {set(labels)}")

    del outputs
    gc.collect()
    return img_rgb, boxes, labels, scores, []

# ---- sidebar ----
def sidebar_controls():
    """Create sidebar controls."""
    with st.sidebar:
        st.subheader("Settings")
        thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05)
        st.caption("Lower threshold → more detections (may include false positives).")
        st.caption("💡 **Default: 0.60** for balanced detection quality.")
        st.caption("Recommended: 0.50-0.70 for utility inventory detection.")
    return thresh

# ---- main app ----
def main():
    """Main detection page."""
    # CRITICAL: Check dependencies first - don't exit, show error in Streamlit
    if torch is None:
        st.error("❌ **PyTorch not found!**")
        st.info("This app requires PyTorch. Please ensure you're running from the correct environment.")
        return
    
    if plt is None:
        st.error("❌ **Matplotlib not found!**")
        st.info("This app requires Matplotlib. Please ensure you're running from the correct environment.")
        return
    
    if DETRdemo is None:
        st.error("❌ **DETR model class not found!**")
        st.info("Could not import DETRdemo. Please check that `training/detr_model.py` exists.")
        return
    
    st.title("Utility Inventory Detection (DETR)")
    st.caption("Custom trained DETR model for detecting insulators, crossarms, and utility poles.")

    prob_thresh = sidebar_controls()
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("Upload a JPG/PNG image of utility infrastructure to run detection.")
        st.info("💡 **Note:** First detection may take 30-60 seconds while the model loads.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Could not read the image. Please try another file.")
        return

    st.image(image, caption="Uploaded image", use_column_width=True)

    # LAZY LOAD: Model only loads when user uploads an image
    # This prevents Streamlit Cloud timeout on startup
    try:
        with st.spinner("Loading model (first time may take 30-60 seconds)..."):
            model = load_utility_detr_model()
        if model is None:
            return
        
        img_rgb, boxes, labels, scores, debug_info = infer_utility_detr(image, model, prob_thresh)
        title = "Utility Inventory Detection (DETR)"
    except AssertionError as e:
        st.error(str(e))
        return
    except RuntimeError as e:
        st.error(f"Inference failed: {e}")
        st.exception(e)
        return

    if len(boxes) == 0:
        st.warning(f"⚠️ No detections above threshold {prob_thresh:.2f}")
        if debug_info:
            with st.expander("🔍 Debug: Top predictions (below threshold)"):
                for info in debug_info:
                    st.text(info)
        st.info("💡 **Try:** Lower the confidence threshold (try 0.10-0.30)")
        return

    # Display results
    fig = draw_boxes_matplotlib(img_rgb, boxes, labels, scores, title=title)
    st.pyplot(fig)

    # Show detection statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(boxes))
    with col2:
        st.metric("Avg Confidence", f"{np.mean(scores):.2f}")
    with col3:
        unique_classes = len(set(labels))
        st.metric("Classes Found", unique_classes)

    # Show class distribution
    if len(labels) > 0:
        st.subheader("Detection Summary")
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            st.write(f"**{class_name}**: {count} detection(s)")

    # Show raw detections table
    with st.expander("Show raw detections"):
        import pandas as pd
        df = pd.DataFrame({
            "label": labels,
            "confidence": [float(x) for x in scores],
            "xmin": [b[0] for b in boxes],
            "ymin": [b[1] for b in boxes],
            "xmax": [b[2] for b in boxes],
            "ymax": [b[3] for b in boxes],
        }).sort_values("confidence", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

# ---- About page ----
def show_about():
    """Show about page with training info."""
    try:
        from about_page import show_about_page
        
        # Find training log - try multiple possible paths
        current_file = Path(__file__)
        # Current file is at: utility-detr/streamlit_deployment/main.py
        # Training log is now in the same directory: utility-detr/streamlit_deployment/training_log.json
        training_log_paths = [
            current_file.parent / "training_log.json",  # Same directory as main.py (preferred)
            current_file.parent.parent / "training" / "logs" / "training_log.json",  # utility-detr/training/logs/
            current_file.parent.parent / "training" / "training_log.json",  # utility-detr/training/
            current_file.parent.parent.parent / "utility-detr" / "training" / "logs" / "training_log.json",  # From workspace root
            Path("/Users/vijayprakashreddy/Documents/SIEAERO/model-build/utility-detr/training/logs/training_log.json"),  # Absolute path
        ]
        
        log_path = None
        for path in training_log_paths:
            if path.exists():
                log_path = str(path)
                break
        
        show_about_page(log_path)
    except ImportError as e:
        st.header("About")
        st.error(f"Error loading about page: {e}")
        st.write("### Utility Inventory Detection Model")
        st.write("**Classes**: insulators, crossarm, utility-pole")
        st.write("**Model**: Custom trained DETR (Detection Transformer)")
        st.write("**Training**: 50 epochs on merged utility inventory dataset")

# ---- main app with navigation ----
def main_with_nav():
    """Main app with navigation sidebar."""
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Detection", "About"], index=0)
    
    if page == "Detection":
        main()
    elif page == "About":
        show_about()

if __name__ == "__main__":
    # Use navigation version
    main_with_nav()

