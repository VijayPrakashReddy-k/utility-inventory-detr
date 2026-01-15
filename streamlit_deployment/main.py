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

# Try to import streamlit-drawable-canvas for visual box editing
# Use the maintained fork from GitHub that fixes Streamlit compatibility
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

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

# Import DETR model - try multiple methods
# CRITICAL: Don't exit on failure - handle gracefully in functions
# First try: import from same directory (streamlit_deployment/detr_model.py)
current_file = Path(__file__).resolve()
DETRdemo = None

try:
    # First try: same directory (for Streamlit Cloud deployment)
    from detr_model import DETRdemo
except ImportError:
    # Fallback 1: try from training directory (for local development)
    training_dir = current_file.parent.parent / "training"
    training_path = str(training_dir)
    if training_path not in sys.path:
        sys.path.insert(0, training_path)
    try:
        from detr_model import DETRdemo
    except ImportError:
        # Fallback 2: try direct file import from training directory
        import importlib.util
        model_file = training_dir / "detr_model.py"
        if model_file.exists():
            spec = importlib.util.spec_from_file_location("detr_model", model_file)
            if spec and spec.loader:
                detr_model = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(detr_model)
                DETRdemo = detr_model.DETRdemo
        else:
            # Fallback 3: try from same directory using direct file import
            local_model_file = current_file.parent / "detr_model.py"
            if local_model_file.exists():
                spec = importlib.util.spec_from_file_location("detr_model", local_model_file)
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

# Background image - will be set in main() function to avoid top-level file I/O

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
    # Use green color for all boxes
    green_color = (0.0, 0.8, 0.0)  # Bright green (RGB normalized)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pil_img)

    # Draw all boxes in green with solid lines (thicker)
    for (xmin, ymin, xmax, ymax), name, sc in zip(boxes, labels, scores):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=green_color, linewidth=4, linestyle='-'))
        ax.text(xmin, ymin, f"{name}: {sc:.2f}", fontsize=14, weight='bold',
                bbox=dict(facecolor="yellow", alpha=0.5))

    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.0)
    return fig

# ---- Canvas Editor Functions ----
def display_advanced_obb_editor(img_rgb, boxes, labels, scores, indices):
    """
    Advanced OBB (Oriented Bounding Box) Editor
    
    Features:
    - Draw horizontal and rotated bounding boxes
    - Click and drag to create boxes
    - Rotate boxes by dragging rotation handle
    - Move boxes by dragging center
    - Resize boxes by dragging corners
    - Delete boxes
    - Support for multiple annotation formats
    - Save edits back to feedback system
    
    Returns:
        JSON string of edited boxes when saved, None otherwise
    """
    import base64
    import json
    from io import BytesIO
    
    # Convert image to base64
    buffered = BytesIO()
    img_rgb.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Prepare existing boxes (convert xyxy to OBB format)
    annotations_json = json.dumps([
        {
            'type': 'obb',
            'cx': (box[0] + box[2]) / 2,  # center x
            'cy': (box[1] + box[3]) / 2,  # center y
            'width': box[2] - box[0],
            'height': box[3] - box[1],
            'angle': 0,  # rotation angle in radians
            'label': label,
            'score': float(score),
            'index': int(idx),
            'color': f'hsl({(int(idx) * 137) % 360}, 70%, 50%)'  # Unique color per box
        }
        for box, label, score, idx in zip(boxes, labels, scores, indices)
    ])
    
    html_code = f"""
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        .editor-container {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px 20px 10px 20px;  /* Reduced bottom padding from 20px to 10px */
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        
        .toolbar {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            align-items: center;
        }}
        
        .tool-btn {{
            padding: 10px 20px;
            border: 2px solid transparent;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
            color: #333;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .tool-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .tool-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }}
        
        .tool-btn.danger {{
            background: #ff4757;
            color: white;
        }}
        
        .tool-btn.danger:hover {{
            background: #ee5a6f;
        }}
        
        .canvas-wrapper {{
            position: relative;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        
        #canvas {{
            display: block;
            cursor: crosshair;
            width: 100%;
            height: auto;
        }}
        
        .info-panel {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .info-item {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .info-label {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .info-value {{
            font-size: 16px;
            font-weight: 700;
            color: #333;
        }}
        
        .label-input {{
            padding: 8px 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }}
        
        .label-input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .instructions {{
            background: #f0f4ff;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
            border-left: 4px solid #667eea;
            margin-bottom: 0;  /* Remove any bottom margin */
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.6; transform: scale(1.2); }}
        }}
    </style>
    
    <div class="editor-container">
        <!-- Toolbar -->
        <div class="toolbar">
            <button class="tool-btn" id="drawBtn" onclick="setMode('draw')">
                <span>✏️</span> Draw Box
            </button>
            <button class="tool-btn active" id="selectBtn" onclick="setMode('select')">
                <span>👆</span> Select/Edit
            </button>
            <button class="tool-btn danger" onclick="deleteSelected()">
                <span>🗑️</span> Delete
            </button>
            <button class="tool-btn danger" onclick="clearAll()">
                <span>🧹</span> Clear All
            </button>
            <button class="tool-btn" onclick="saveBoxes()" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);">
                <span>💾</span> Save Edits
            </button>
            
            <div style="flex: 1;"></div>
            
            <input type="text" id="labelInput" class="label-input" placeholder="Label (e.g., insulators)" value="insulators">
            <input type="number" id="scoreInput" class="label-input" placeholder="Score" value="1.0" min="0" max="1" step="0.01" style="width: 100px;" title="Confidence score for new boxes (0.0-1.0)">
        </div>
        
        <!-- Canvas -->
        <div class="canvas-wrapper">
            <canvas id="canvas" width="{img_rgb.width}" height="{img_rgb.height}"></canvas>
        </div>
        
        <!-- Info Panel -->
        <div class="info-panel">
            <div class="info-item">
                <div class="info-label">Mode</div>
                <div class="info-value" id="modeDisplay">Select/Edit</div>
            </div>
            <div class="info-item">
                <div class="info-label">Total Boxes</div>
                <div class="info-value" id="boxCount">0</div>
            </div>
            <div class="info-item">
                <div class="info-label">Selected</div>
                <div class="info-value" id="selectedInfo">None</div>
            </div>
            <div class="info-item">
                <div class="info-label">Angle</div>
                <div class="info-value" id="angleInfo">0°</div>
            </div>
        </div>
        
        <!-- JSON Output Panel (visible after Save Edits) -->
        <div id="jsonOutputPanel" style="display: none; margin-top: 20px; padding: 20px; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-radius: 12px; border: 2px solid #4CAF50; box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #2E7D32; font-size: 18px;">📋 Copy This JSON</h3>
                <button onclick="copyJsonToClipboard()" id="copyBtn" style="padding: 10px 20px; background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3); transition: all 0.3s ease;">
                    📋 Copy to Clipboard
                </button>
            </div>
            <textarea id="jsonOutput" readonly onclick="this.select()" style="width: 100%; height: 200px; font-family: 'Courier New', monospace; font-size: 12px; padding: 15px; border: 2px solid #81C784; border-radius: 8px; background: #FFFFFF; resize: vertical; cursor: text; line-height: 1.5;"></textarea>
            <div style="margin-top: 15px; padding: 12px; background: rgba(255, 255, 255, 0.95); border-radius: 8px; font-size: 13px; color: #1B5E20; border-left: 4px solid #4CAF50;">
                <div style="margin-bottom: 8px;"><strong>✅ Next Step:</strong> Copy the JSON above, paste it in the <strong>"📥 Paste JSON data here"</strong> field below, then click <strong>"✅ Apply Edited Data"</strong></div>
                <div style="padding: 8px; background: #FFF9C4; border-left: 3px solid #FBC02D; border-radius: 4px; font-size: 12px; color: #F57F17;">
                    <strong>💡 Score Info:</strong><br>
                    • <strong>Original detections</strong> (edited) → Keep real scores (e.g., 0.79, 0.85)<br>
                    • <strong>Newly drawn boxes</strong> (index: -1) → Get placeholder score: 1.0
                </div>
            </div>
        </div>
        
        <!-- Instructions -->
        <div class="instructions">
            <strong>💡 How to Edit Boxes:</strong><br>
            <strong>Draw:</strong> Click "✏️ Draw Box" and drag on image (auto-selects after drawing)<br>
            <strong>🟠 Rotate:</strong> Drag the <span style="color: #ff6b00; font-weight: bold;">ORANGE CIRCLE ↻</span> above box<br>
            <strong>🟢 Resize:</strong> Drag the <span style="color: #00ff00; font-weight: bold;">GREEN SQUARES ⤡</span> at corners<br>
            <strong>🔵 Move:</strong> Drag the <span style="color: #0066ff; font-weight: bold;">BLUE CIRCLE ✛</span> at center<br>
            <strong>Delete:</strong> Press Delete key | <strong>Switch:</strong> Press R key
        </div>
    </div>
    
    <!-- Load Streamlit component library for bidirectional communication -->
    <script>
        // Initialize Streamlit API if not already available
        if (typeof Streamlit === 'undefined') {{
            window.Streamlit = {{
                setComponentValue: function(value) {{
                    // Send message to parent window (Streamlit)
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: value
                    }}, '*');
                }},
                setFrameHeight: function(height) {{
                    window.parent.postMessage({{
                        type: 'streamlit:setFrameHeight',
                        height: height
                    }}, '*');
                }}
            }};
        }}
    </script>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = 'data:image/png;base64,{img_base64}';
        
        let boxes = {annotations_json};
        let mode = 'select';  // Default to Select/Edit mode to edit original detections
        let selectedBox = null;
        let isDragging = false;
        let dragHandle = null;
        let startX, startY;
        let currentBox = null;
        
        img.onload = () => {{
            canvas.style.cursor = 'pointer';  // Set cursor for Select/Edit mode
            render();
            updateInfo();
        }};
        
        function setMode(newMode) {{
            mode = newMode;
            selectedBox = null;
            document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
            if (newMode === 'draw') {{
                document.getElementById('drawBtn').classList.add('active');
                canvas.style.cursor = 'crosshair';
                document.getElementById('modeDisplay').textContent = 'Draw';
            }} else {{
                document.getElementById('selectBtn').classList.add('active');
                canvas.style.cursor = 'pointer';
                document.getElementById('modeDisplay').textContent = 'Select/Edit';
            }}
            render();
            updateInfo();
        }}
        
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            
            // Draw all boxes
            boxes.forEach((box, idx) => {{
                drawOBB(box, idx === selectedBox);
            }});
            
            // Draw current box being created
            if (currentBox) {{
                 ctx.save();
                 ctx.setLineDash([5, 5]);
                 ctx.strokeStyle = '#00ff00';
                 ctx.lineWidth = 6;  // Extra thick for drawing preview
                 const hw = currentBox.width / 2;
                const hh = currentBox.height / 2;
                ctx.translate(currentBox.cx, currentBox.cy);
                ctx.rotate(currentBox.angle);
                ctx.strokeRect(-hw, -hh, currentBox.width, currentBox.height);
                ctx.restore();
            }}
        }}
        
        function drawOBB(box, isSelected) {{
            const hw = box.width / 2;
            const hh = box.height / 2;
            
            ctx.save();
            ctx.translate(box.cx, box.cy);
            ctx.rotate(box.angle);
            
             // Draw box with very thick lines
             ctx.strokeStyle = isSelected ? '#00ff00' : (box.color || '#ff0000');
             ctx.lineWidth = isSelected ? 8 : 6;  // Extra thick for visibility
             ctx.strokeRect(-hw, -hh, box.width, box.height);
            
            // Draw label
            ctx.fillStyle = box.color || '#ff0000';
            ctx.font = 'bold 14px Arial';
            const label = `${{box.label}} ${{box.score ? box.score.toFixed(2) : ''}}`;
            ctx.fillText(label, -hw, -hh - 5);
            
            if (isSelected) {{
                 // Draw corner resize handles with ENLARGE icon (⤡)
                 const cornerSize = 22;  // Larger handles
                 const corners = [
                     {{x: -hw, y: -hh, dir: 'nw'}}, {{x: hw, y: -hh, dir: 'ne'}},
                     {{x: hw, y: hh, dir: 'se'}}, {{x: -hw, y: hh, dir: 'sw'}}
                 ];
                 corners.forEach(c => {{
                     // Green rounded square background
                     ctx.fillStyle = '#00ff00';
                     ctx.strokeStyle = 'white';
                     ctx.lineWidth = 4;  // Thicker outline
                     ctx.beginPath();
                     ctx.roundRect(c.x - cornerSize/2, c.y - cornerSize/2, cornerSize, cornerSize, 3);
                     ctx.fill();
                     ctx.stroke();
                     
                     // Draw ENLARGE/EXPAND icon with arrows pointing outward
                     ctx.strokeStyle = 'black';
                     ctx.lineWidth = 2.5;
                     ctx.lineCap = 'round';
                     
                     // Diagonal double arrows pointing outward (⤡ icon)
                     const arrowSize = 6;
                     const offsetX = (c.dir === 'nw' || c.dir === 'sw') ? -1 : 1;
                     const offsetY = (c.dir === 'nw' || c.dir === 'ne') ? -1 : 1;
                     
                     // Main diagonal line
                     ctx.beginPath();
                     ctx.moveTo(c.x - arrowSize * offsetX * 0.5, c.y - arrowSize * offsetY * 0.5);
                     ctx.lineTo(c.x + arrowSize * offsetX * 0.5, c.y + arrowSize * offsetY * 0.5);
                     
                     // Arrow head 1
                     ctx.moveTo(c.x + arrowSize * offsetX * 0.5, c.y + arrowSize * offsetY * 0.5);
                     ctx.lineTo(c.x + arrowSize * offsetX * 0.5 - 3 * offsetX, c.y + arrowSize * offsetY * 0.5);
                     ctx.moveTo(c.x + arrowSize * offsetX * 0.5, c.y + arrowSize * offsetY * 0.5);
                     ctx.lineTo(c.x + arrowSize * offsetX * 0.5, c.y + arrowSize * offsetY * 0.5 - 3 * offsetY);
                     
                     // Arrow head 2 (opposite direction for resize feel)
                     ctx.moveTo(c.x - arrowSize * offsetX * 0.5, c.y - arrowSize * offsetY * 0.5);
                     ctx.lineTo(c.x - arrowSize * offsetX * 0.5 + 3 * offsetX, c.y - arrowSize * offsetY * 0.5);
                     ctx.moveTo(c.x - arrowSize * offsetX * 0.5, c.y - arrowSize * offsetY * 0.5);
                     ctx.lineTo(c.x - arrowSize * offsetX * 0.5, c.y - arrowSize * offsetY * 0.5 + 3 * offsetY);
                     
                     ctx.stroke();
                 }});
                
                 // Draw rotation handle with ROTATE icon (↻)
                 const rotY = -hh - 30;
                 const rotRadius = 16;  // Larger rotation handle
                 ctx.beginPath();
                 ctx.arc(0, rotY, rotRadius, 0, 2 * Math.PI);
                 ctx.fillStyle = '#ff6b00';
                 ctx.fill();
                 ctx.strokeStyle = 'white';
                 ctx.lineWidth = 4;  // Thicker outline
                 ctx.stroke();
                 
                 // Draw CIRCULAR ROTATION arrow (↻ icon)
                 ctx.strokeStyle = 'white';
                 ctx.lineWidth = 3;
                 ctx.lineCap = 'round';
                 
                 // Circular arrow (270 degrees)
                 ctx.beginPath();
                 ctx.arc(0, rotY, 7, -Math.PI * 0.7, Math.PI * 0.7, false);
                 ctx.stroke();
                 
                 // Arrow head (pointing clockwise)
                 ctx.fillStyle = 'white';
                 ctx.beginPath();
                 ctx.moveTo(Math.cos(Math.PI * 0.7) * 7, rotY + Math.sin(Math.PI * 0.7) * 7);
                 ctx.lineTo(Math.cos(Math.PI * 0.7) * 7 - 5, rotY + Math.sin(Math.PI * 0.7) * 7 - 2);
                 ctx.lineTo(Math.cos(Math.PI * 0.7) * 7 - 3, rotY + Math.sin(Math.PI * 0.7) * 7 + 4);
                 ctx.closePath();
                 ctx.fill();
                 
                 // Add small circle in center for better visibility
                 ctx.beginPath();
                 ctx.arc(0, rotY, 2, 0, 2 * Math.PI);
                 ctx.fillStyle = 'white';
                 ctx.fill();
                
                 // Draw center MOVE/ADJUST handle with 4-way arrows icon (⥮)
                 const centerSize = 20;  // Larger center handle
                 ctx.beginPath();
                 ctx.arc(0, 0, centerSize/2, 0, 2 * Math.PI);
                 ctx.fillStyle = '#0066ff';
                 ctx.fill();
                 ctx.strokeStyle = 'white';
                 ctx.lineWidth = 4;  // Thicker outline
                 ctx.stroke();
                 
                 // Draw MOVE icon with 4-directional arrows (⥮)
                 ctx.strokeStyle = 'white';
                 ctx.lineWidth = 2.5;
                 ctx.lineCap = 'round';
                 ctx.lineJoin = 'round';
                 
                 const arrowLen = 6;
                 const arrowHeadSize = 3;
                 
                 // Draw 4 arrows pointing outward
                 // Up arrow
                 ctx.beginPath();
                 ctx.moveTo(0, 0);
                 ctx.lineTo(0, -arrowLen);
                 ctx.moveTo(-arrowHeadSize, -arrowLen + arrowHeadSize);
                 ctx.lineTo(0, -arrowLen);
                 ctx.lineTo(arrowHeadSize, -arrowLen + arrowHeadSize);
                 ctx.stroke();
                 
                 // Down arrow
                 ctx.beginPath();
                 ctx.moveTo(0, 0);
                 ctx.lineTo(0, arrowLen);
                 ctx.moveTo(-arrowHeadSize, arrowLen - arrowHeadSize);
                 ctx.lineTo(0, arrowLen);
                 ctx.lineTo(arrowHeadSize, arrowLen - arrowHeadSize);
                 ctx.stroke();
                 
                 // Left arrow
                 ctx.beginPath();
                 ctx.moveTo(0, 0);
                 ctx.lineTo(-arrowLen, 0);
                 ctx.moveTo(-arrowLen + arrowHeadSize, -arrowHeadSize);
                 ctx.lineTo(-arrowLen, 0);
                 ctx.lineTo(-arrowLen + arrowHeadSize, arrowHeadSize);
                 ctx.stroke();
                 
                 // Right arrow
                 ctx.beginPath();
                 ctx.moveTo(0, 0);
                 ctx.lineTo(arrowLen, 0);
                 ctx.moveTo(arrowLen - arrowHeadSize, -arrowHeadSize);
                 ctx.lineTo(arrowLen, 0);
                 ctx.lineTo(arrowLen - arrowHeadSize, arrowHeadSize);
                 ctx.stroke();
                 
                 // Center dot for better visibility
                 ctx.beginPath();
                 ctx.arc(0, 0, 2, 0, 2 * Math.PI);
                 ctx.fillStyle = 'white';
                 ctx.fill();
            }}
            
            ctx.restore();
        }}
        
        function getMousePos(e) {{
            const rect = canvas.getBoundingClientRect();
            return {{
                x: (e.clientX - rect.left) * (canvas.width / rect.width),
                y: (e.clientY - rect.top) * (canvas.height / rect.height)
            }};
        }}
        
        function getHandle(pos, box) {{
            const hw = box.width / 2;
            const hh = box.height / 2;
            
            // Transform point to box local coordinates
            const dx = pos.x - box.cx;
            const dy = pos.y - box.cy;
            const cos = Math.cos(-box.angle);
            const sin = Math.sin(-box.angle);
            const lx = dx * cos - dy * sin;
            const ly = dx * sin + dy * cos;
            
            // Check rotation handle (larger hit area matching new size)
            if (Math.hypot(lx, ly + hh + 25) < 16) return 'rotate';
            
            // Check center (larger hit area matching new size)
            if (Math.hypot(lx, ly) < 20) return 'center';
            
            // Check corners (larger hit area matching new size)
            const corners = [
                {{x: -hw, y: -hh, name: 'nw'}}, {{x: hw, y: -hh, name: 'ne'}},
                {{x: hw, y: hh, name: 'se'}}, {{x: -hw, y: hh, name: 'sw'}}
            ];
            for (let c of corners) {{
                if (Math.hypot(lx - c.x, ly - c.y) < 22) return c.name;
            }}
            
            // Check if inside box
            if (Math.abs(lx) <= hw && Math.abs(ly) <= hh) return 'center';
            
            return null;
        }}
        
        canvas.addEventListener('mousedown', (e) => {{
            const pos = getMousePos(e);
            
            if (mode === 'draw') {{
                isDragging = true;
                startX = pos.x;
                startY = pos.y;
                currentBox = {{
                    cx: pos.x,
                    cy: pos.y,
                    width: 0,
                    height: 0,
                    angle: 0
                }};
            }} else if (mode === 'select') {{
                // Find clicked box
                for (let i = boxes.length - 1; i >= 0; i--) {{
                    const handle = getHandle(pos, boxes[i]);
                    if (handle) {{
                        selectedBox = i;
                        dragHandle = handle;
                        isDragging = true;
                        startX = pos.x;
                        startY = pos.y;
                        render();
                        updateInfo();
                        return;
                    }}
                }}
                selectedBox = null;
                render();
                updateInfo();
            }}
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            
            const pos = getMousePos(e);
            
            if (mode === 'draw' && currentBox) {{
                const dx = pos.x - startX;
                const dy = pos.y - startY;
                currentBox.width = Math.abs(dx);
                currentBox.height = Math.abs(dy);
                currentBox.cx = startX + dx / 2;
                currentBox.cy = startY + dy / 2;
                render();
            }} else if (mode === 'select' && selectedBox !== null) {{
                const box = boxes[selectedBox];
                const dx = pos.x - startX;
                const dy = pos.y - startY;
                
                if (dragHandle === 'center') {{
                    // Move the box
                    box.cx += dx;
                    box.cy += dy;
                }} else if (dragHandle === 'rotate') {{
                    // Rotate the box
                    const angle = Math.atan2(pos.y - box.cy, pos.x - box.cx);
                    box.angle = angle + Math.PI / 2;
                }} else {{
                    // Resize from corners
                    // Transform mouse movement to box local coordinates
                    const cos = Math.cos(-box.angle);
                    const sin = Math.sin(-box.angle);
                    const localDx = dx * cos - dy * sin;
                    const localDy = dx * sin + dy * cos;
                    
                    // Resize based on which corner
                    if (dragHandle === 'se') {{
                        box.width = Math.max(20, box.width + localDx * 2);
                        box.height = Math.max(20, box.height + localDy * 2);
                    }} else if (dragHandle === 'sw') {{
                        box.width = Math.max(20, box.width - localDx * 2);
                        box.height = Math.max(20, box.height + localDy * 2);
                    }} else if (dragHandle === 'ne') {{
                        box.width = Math.max(20, box.width + localDx * 2);
                        box.height = Math.max(20, box.height - localDy * 2);
                    }} else if (dragHandle === 'nw') {{
                        box.width = Math.max(20, box.width - localDx * 2);
                        box.height = Math.max(20, box.height - localDy * 2);
                    }}
                }}
                
                startX = pos.x;
                startY = pos.y;
                render();
                updateInfo();
            }}
        }});
        
        canvas.addEventListener('mouseup', (e) => {{
            if (mode === 'draw' && currentBox && currentBox.width > 10 && currentBox.height > 10) {{
                const newBoxIndex = boxes.length;
                const scoreValue = parseFloat(document.getElementById('scoreInput').value) || 1.0;
                boxes.push({{
                    type: 'obb',
                    cx: currentBox.cx,
                    cy: currentBox.cy,
                    width: currentBox.width,
                    height: currentBox.height,
                    angle: 0,
                    label: document.getElementById('labelInput').value || 'object',
                    score: Math.max(0, Math.min(1, scoreValue)),  // Clamp between 0 and 1
                    color: `hsl(${{(boxes.length * 137) % 360}}, 70%, 50%)`,
                    index: -1
                }});
                
                // Auto-switch to select mode and select the newly created box
                setMode('select');
                selectedBox = newBoxIndex;
                
                updateInfo();
            }}
            
            isDragging = false;
            currentBox = null;
            dragHandle = null;
            render();
        }});
        
        function deleteSelected() {{
            if (selectedBox !== null) {{
                boxes.splice(selectedBox, 1);
                selectedBox = null;
                render();
                updateInfo();
            }}
        }}
        
        function clearAll() {{
            if (confirm('Clear all boxes?')) {{
                boxes = [];
                selectedBox = null;
                render();
                updateInfo();
            }}
        }}
        
        function updateInfo() {{
            document.getElementById('boxCount').textContent = boxes.length;
            if (selectedBox !== null && selectedBox < boxes.length) {{
                const box = boxes[selectedBox];
                document.getElementById('selectedInfo').textContent = `Box ${{selectedBox + 1}} (${{box.label}})`;
                document.getElementById('angleInfo').textContent = `${{(box.angle * 180 / Math.PI).toFixed(1)}}°`;
            }} else {{
                document.getElementById('selectedInfo').textContent = 'None';
                document.getElementById('angleInfo').textContent = '0°';
            }}
        }}
        
        function saveBoxes() {{
            // Confirmation dialog
            const message = boxes.length === 0 
                ? 'No boxes to save. Do you want to clear all detections?'
                : 'Save ' + boxes.length + ' box(es) and update feedback?\\n\\nThis will overwrite the current detections.';
            
            if (!confirm(message)) {{
                return; // User cancelled
            }}
            
            // Convert OBB boxes to axis-aligned bounding boxes for feedback
            const savedBoxes = boxes.map(box => {{
                // Calculate the four corners of the rotated box
                const hw = box.width / 2;
                const hh = box.height / 2;
                const cos = Math.cos(box.angle);
                const sin = Math.sin(box.angle);
                
                const corners = [
                    {{x: -hw, y: -hh}},
                    {{x: hw, y: -hh}},
                    {{x: hw, y: hh}},
                    {{x: -hw, y: hh}}
                ];
                
                // Transform corners to image coordinates
                const transformedCorners = corners.map(c => {{
                    const rotX = c.x * cos - c.y * sin;
                    const rotY = c.x * sin + c.y * cos;
                    return {{
                        x: box.cx + rotX,
                        y: box.cy + rotY
                    }};
                }});
                
                // Find axis-aligned bounding box
                const xs = transformedCorners.map(c => c.x);
                const ys = transformedCorners.map(c => c.y);
                const xmin = Math.max(0, Math.min(...xs));
                const ymin = Math.max(0, Math.min(...ys));
                const xmax = Math.min({img_rgb.width}, Math.max(...xs));
                const ymax = Math.min({img_rgb.height}, Math.max(...ys));
                
                return {{
                    index: box.index,
                    label: box.label,
                    score: box.score,
                    bbox: [xmin, ymin, xmax, ymax],
                    obb: {{
                        cx: box.cx,
                        cy: box.cy,
                        width: box.width,
                        height: box.height,
                        angle: box.angle
                    }}
                }};
            }});
            
            // Send to Streamlit
            Streamlit.setComponentValue(JSON.stringify(savedBoxes));
            
            // Display coordinates to user
            let coordsText = 'Saved ' + savedBoxes.length + ' box(es):\\n\\n';
            savedBoxes.forEach((item, idx) => {{
                const bbox = item.bbox;
                const obb = item.obb;
                coordsText += 'Box ' + (idx + 1) + ' (' + item.label + '):\\n';
                coordsText += '  Axis-Aligned Box:\\n';
                coordsText += '    xmin: ' + Math.round(bbox[0]) + '\\n';
                coordsText += '    ymin: ' + Math.round(bbox[1]) + '\\n';
                coordsText += '    xmax: ' + Math.round(bbox[2]) + '\\n';
                coordsText += '    ymax: ' + Math.round(bbox[3]) + '\\n';
                coordsText += '  Oriented Box (OBB):\\n';
                coordsText += '    center_x: ' + Math.round(obb.cx) + '\\n';
                coordsText += '    center_y: ' + Math.round(obb.cy) + '\\n';
                coordsText += '    width: ' + Math.round(obb.width) + '\\n';
                coordsText += '    height: ' + Math.round(obb.height) + '\\n';
                coordsText += '    angle: ' + (obb.angle * 180 / Math.PI).toFixed(2) + '° (' + obb.angle.toFixed(4) + ' rad)\\n\\n';
            }});
            
            alert(coordsText);
            
            // Show JSON in visible panel for easy copying (NO CONSOLE NEEDED!)
            const jsonOutputPanel = document.getElementById('jsonOutputPanel');
            const jsonOutput = document.getElementById('jsonOutput');
            jsonOutput.value = JSON.stringify(savedBoxes, null, 2);
            jsonOutputPanel.style.display = 'block';
            
            // Scroll to JSON panel so user sees it
            setTimeout(() => {{
                jsonOutputPanel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}, 300);
            
            // Also log to console as backup
            console.log('\\n========================================');
            console.log('=== OBB EDITOR - JSON OUTPUT ===');
            console.log('========================================');
            console.log(JSON.stringify(savedBoxes, null, 2));
            console.log('========================================\\n');
            
            // Visual feedback
            const saveBtn = event.currentTarget;
            const originalHTML = saveBtn.innerHTML;
            saveBtn.innerHTML = '<span>✅</span> Saved!';
            saveBtn.style.background = '#4CAF50';
            setTimeout(() => {{
                saveBtn.innerHTML = originalHTML;
                saveBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            }}, 1500);
        }}
        
        // Function to copy JSON to clipboard
        function copyJsonToClipboard() {{
            const jsonOutput = document.getElementById('jsonOutput');
            const copyBtn = document.getElementById('copyBtn');
            
            // Select and copy the text
            jsonOutput.select();
            jsonOutput.setSelectionRange(0, 99999); // For mobile devices
            
            try {{
                document.execCommand('copy');
                
                // Visual feedback
                const originalHTML = copyBtn.innerHTML;
                copyBtn.innerHTML = '✅ Copied!';
                copyBtn.style.background = 'linear-gradient(135deg, #2E7D32 0%, #43A047 100%)';
                
                setTimeout(() => {{
                    copyBtn.innerHTML = originalHTML;
                    copyBtn.style.background = 'linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%)';
                }}, 2000);
            }} catch (err) {{
                // Fallback for modern browsers
                navigator.clipboard.writeText(jsonOutput.value).then(() => {{
                    const originalHTML = copyBtn.innerHTML;
                    copyBtn.innerHTML = '✅ Copied!';
                    copyBtn.style.background = 'linear-gradient(135deg, #2E7D32 0%, #43A047 100%)';
                    
                    setTimeout(() => {{
                        copyBtn.innerHTML = originalHTML;
                        copyBtn.style.background = 'linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%)';
                    }}, 2000);
                }}).catch(err => {{
                    alert('Failed to copy. Please manually select and copy the JSON.');
                }});
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Delete' && selectedBox !== null) {{
                deleteSelected();
            }} else if (e.key === 'r' || e.key === 'R') {{
                setMode(mode === 'draw' ? 'select' : 'draw');
            }}
        }});
    </script>
    """
    
    # Display the component
    st.components.v1.html(html_code, height=img_rgb.height + 280, scrolling=False)
    
    st.info("💡 **New! Easy JSON Copy** - No browser console needed! JSON appears directly in the editor after clicking 'Save Edits'.")
    
    # Manual data entry (primary method since automatic capture doesn't work with st.components.v1.html)
    st.markdown("### 📋 Apply Edited Data")
    st.markdown("""
    **Step 1:** Edit boxes in the canvas above (rotate, resize, move as needed)
    
    **Step 2:** Click **💾 Save Edits** button in the editor
    
    **Step 3:** A **green panel** with JSON will appear below the canvas
    
    **Step 4:** Click **📋 Copy to Clipboard** button (or manually select and copy the JSON)
    
    **Step 5:** Paste the JSON below and click **✅ Apply Edited Data**
    """)
    
    manual_json = st.text_area(
        "📥 Paste JSON data here:",
        placeholder='[{"index": 0, "label": "insulators", "score": 0.79, "bbox": [10, 20, 100, 150], "obb": {"cx": 55, "cy": 85, "width": 90, "height": 130, "angle": -0.2720}}]',
        height=120,
        key="manual_obb_json",
        help="Copy the JSON from browser console and paste here"
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("✅ Apply Edited Data", type="primary", use_container_width=True):
            if manual_json.strip():
                try:
                    import json
                    edited_data = json.loads(manual_json)
                    st.session_state.obb_edited_data = edited_data
                    
                    # Show what was captured
                    rotated_count = sum(1 for box in edited_data if box['obb']['angle'] != 0)
                    st.success(f"✅ Applied {len(edited_data)} box(es)!")
                    if rotated_count > 0:
                        st.info(f"🔄 {rotated_count} box(es) have rotation angles")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON format: {e}")
            else:
                st.warning("⚠️ Please paste JSON data first")
    
    with col2:
        st.caption("💡 If you don't see JSON in console, make sure you clicked 'Save Edits' in the canvas first!")

def display_coco_editor(img_rgb, boxes, labels, scores, indices):
    """Simple COCO Editor - Just load image and show boxes"""
    import base64
    import json
    from io import BytesIO
    
    buffered = BytesIO()
    img_rgb.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    boxes_json = json.dumps([
        {'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3]),
         'label': label, 'score': float(score), 'index': int(idx)}
        for box, label, score, idx in zip(boxes, labels, scores, indices)
    ])
    
    html_code = f"""
    <style>
        body {{ margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: Arial, sans-serif; }}
        .toolbar {{ background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; margin-bottom: 15px; display: flex; gap: 10px; flex-wrap: wrap; }}
        .btn {{ padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px; transition: all 0.3s; }}
        .btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .btn-draw {{ background: #4CAF50; color: white; }}
        .btn-select {{ background: #2196F3; color: white; }}
        .btn-delete {{ background: #FF9800; color: white; }}
        .btn-clear {{ background: #f44336; color: white; }}
        .btn-save {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .active {{ box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.5); }}
        .wrapper {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); }}
        #c {{ display: block; cursor: crosshair; max-width: 100%; height: auto; }}
        .info {{ background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px; margin-top: 15px; }}
    </style>
    
    <div class="toolbar">
        <button class="btn btn-draw active" id="drawBtn" onclick="setMode('draw')">✏️ Draw Box</button>
        <button class="btn btn-select" id="selectBtn" onclick="setMode('select')">👆 Select/Edit</button>
        <button class="btn btn-delete" onclick="deleteSelected()">🗑️ Delete</button>
        <button class="btn btn-clear" onclick="clearAll()">🧹 Clear All</button>
        <button class="btn btn-save" onclick="saveBoxes()">💾 Save Edits</button>
        <div style="flex: 1;"></div>
        <select class="btn" id="labelSelect" style="background: white; border: 2px solid #ddd;">
            <option value="insulators">Insulators</option>
            <option value="crossarm">Crossarm</option>
            <option value="utility-pole">Utility-Pole</option>
        </select>
    </div>
    
    <div class="wrapper">
        <canvas id="c" width="{img_rgb.width}" height="{img_rgb.height}"></canvas>
    </div>
    
    <div class="info">
        <strong>Mode:</strong> <span id="mode">Draw</span> | 
        <strong>Boxes:</strong> <span id="boxCount">0</span> | 
        <strong>Selected:</strong> <span id="selected">None</span>
    </div>
    
    <script>
    var c = document.getElementById('c');
    var ctx = c.getContext('2d');
    var img = new Image();
    var boxes = {boxes_json};
    var mode = 'draw';
    var selectedBox = -1;
    
    var boxes = {boxes_json};
    var mode = 'draw';
    var selectedBox = -1;
    
    img.onload = function() {{ 
        c.width = img.width;
        c.height = img.height;
        render(); 
    }};
    img.src = 'data:image/png;base64,{img_base64}';
    
    function render() {{
        ctx.clearRect(0, 0, c.width, c.height);
        ctx.drawImage(img, 0, 0);
        
        // Draw boxes
        boxes.forEach((box, idx) => {{
            ctx.strokeStyle = idx === selectedBox ? '#00FF00' : '#FF0000';
            ctx.lineWidth = idx === selectedBox ? 6 : 4;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            
            ctx.fillStyle = idx === selectedBox ? '#00FF00' : '#FF0000';
            ctx.font = 'bold 16px Arial';
            ctx.fillText(box.label + ' ' + box.score.toFixed(2), box.x1, Math.max(20, box.y1 - 5));
        }});
        
        document.getElementById('boxCount').textContent = boxes.length;
        document.getElementById('selected').textContent = selectedBox >= 0 ? 'Box ' + (selectedBox + 1) : 'None';
    }}
    
    function setMode(newMode) {{
        mode = newMode;
        selectedBox = -1;
        document.getElementById('drawBtn').classList.toggle('active', newMode === 'draw');
        document.getElementById('selectBtn').classList.toggle('active', newMode === 'select');
        document.getElementById('mode').textContent = newMode === 'draw' ? 'Draw' : 'Select/Edit';
        c.style.cursor = newMode === 'draw' ? 'crosshair' : 'pointer';
        render();
    }}
    
    function deleteSelected() {{
        if (selectedBox >= 0) {{
            boxes.splice(selectedBox, 1);
            selectedBox = -1;
            render();
        }}
    }}
    
    function clearAll() {{
        if (confirm('Clear all boxes?')) {{
            boxes = [];
            selectedBox = -1;
            render();
        }}
    }}
    
    function setModeOLD(newMode) {{
        mode = newMode;
        selectedBox = -1;
        document.getElementById('drawBtn').classList.toggle('active', newMode === 'draw');
        document.getElementById('selectBtn').classList.toggle('active', newMode === 'select');
        document.getElementById('mode').textContent = newMode === 'draw' ? 'Draw' : 'Select/Edit';
        c.style.cursor = newMode === 'draw' ? 'crosshair' : 'pointer';
        render();
    }}
    
    function deleteSelected() {{
        if (selectedBox >= 0) {{
            boxes.splice(selectedBox, 1);
            selectedBox = -1;
            render();
        }}
    }}
    
    function clearAll() {{
        if (confirm('Clear all boxes?')) {{
            boxes = [];
            selectedBox = -1;
            render();
        }}
    }}
    
    function saveBoxes() {{
        if (boxes.length === 0) {{
            alert('No boxes to save!');
            return;
        }}
        
        // Format box coordinates
        let coordsText = 'Saved ' + boxes.length + ' box(es):\\n\\n';
        boxes.forEach((box, idx) => {{
            coordsText += 'Box ' + (idx + 1) + ' (' + box.label + '):\\n';
            coordsText += '  xmin: ' + Math.round(box.x1) + '\\n';
            coordsText += '  ymin: ' + Math.round(box.y1) + '\\n';
            coordsText += '  xmax: ' + Math.round(box.x2) + '\\n';
            coordsText += '  ymax: ' + Math.round(box.y2) + '\\n';
            coordsText += '  width: ' + Math.round(box.x2 - box.x1) + '\\n';
            coordsText += '  height: ' + Math.round(box.y2 - box.y1) + '\\n\\n';
        }});
        
        alert(coordsText);
        
        // Also log to console for copy-paste
        console.log('=== BOUNDING BOX COORDINATES ===');
        boxes.forEach((box, idx) => {{
            console.log('Box ' + (idx + 1) + ' (' + box.label + '):', {{
                xmin: Math.round(box.x1),
                ymin: Math.round(box.y1),
                xmax: Math.round(box.x2),
                ymax: Math.round(box.y2),
                width: Math.round(box.x2 - box.x1),
                height: Math.round(box.y2 - box.y1),
                label: box.label,
                score: box.score
            }});
        }});
        console.log('=== JSON FORMAT ===');
        console.log(JSON.stringify(boxes, null, 2));
    }}
    
    // Mouse event handlers
    let isDrawing = false;
    let startX, startY;
    let currentBox = null;
    
    c.addEventListener('mousedown', (e) => {{
        const rect = c.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        if (mode === 'draw') {{
            isDrawing = true;
            startX = x;
            startY = y;
            currentBox = {{x1: x, y1: y, x2: x, y2: y}};
        }} else {{
            // Select mode - check if clicking on a box
            selectedBox = -1;
            for (let i = boxes.length - 1; i >= 0; i--) {{
                const box = boxes[i];
                if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {{
                    selectedBox = i;
                    break;
                }}
            }}
            render();
        }}
    }});
    
    c.addEventListener('mousemove', (e) => {{
        if (mode === 'draw' && isDrawing) {{
            const rect = c.getBoundingClientRect();
            currentBox.x2 = e.clientX - rect.left;
            currentBox.y2 = e.clientY - rect.top;
            
            // Render with current box preview
            ctx.clearRect(0, 0, c.width, c.height);
            ctx.drawImage(img, 0, 0);
            
            // Draw existing boxes
            boxes.forEach((box, idx) => {{
                ctx.strokeStyle = '#FF0000';
                ctx.lineWidth = 4;
                ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                ctx.fillStyle = '#FF0000';
                ctx.font = 'bold 16px Arial';
                ctx.fillText(box.label + ' ' + box.score.toFixed(2), box.x1, Math.max(20, box.y1 - 5));
            }});
            
            // Draw current box being drawn
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(currentBox.x1, currentBox.y1, currentBox.x2 - currentBox.x1, currentBox.y2 - currentBox.y1);
            ctx.setLineDash([]);
        }}
    }});
    
    c.addEventListener('mouseup', (e) => {{
        if (mode === 'draw' && isDrawing) {{
            isDrawing = false;
            if (currentBox && Math.abs(currentBox.x2 - currentBox.x1) > 10 && Math.abs(currentBox.y2 - currentBox.y1) > 10) {{
                const selectedLabel = document.getElementById('labelSelect').value;
                boxes.push({{
                    x1: Math.min(currentBox.x1, currentBox.x2),
                    y1: Math.min(currentBox.y1, currentBox.y2),
                    x2: Math.max(currentBox.x1, currentBox.x2),
                    y2: Math.max(currentBox.y1, currentBox.y2),
                    label: selectedLabel,
                    score: 1.0,
                    index: -1
                }});
            }}
            currentBox = null;
            render();
        }}
    }});
    </script>
    """
    
    # Calculate proper height
    max_display_width = 1200
    aspect_ratio = img_rgb.height / img_rgb.width
    display_height = min(img_rgb.height, int(max_display_width * aspect_ratio))
    
    st.components.v1.html(html_code, height=display_height + 220)


def display_coco_editor_FULL_OLD(img_rgb, boxes, labels, scores, indices):
    """OLD VERSION"""
    import base64
    import json
    from io import BytesIO
    
    buffered = BytesIO()
    img_rgb.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    boxes_json = json.dumps([
        {'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3]),
         'label': label, 'score': float(score), 'index': int(idx)}
        for box, label, score, idx in zip(boxes, labels, scores, indices)
    ])
    
    html_code_old = f"""
    <style>
        .editor-container {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 100%;
            margin: 0 auto;
        }}
        
        .toolbar {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .tool-btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(255,255,255,0.2);
            color: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        
        .tool-btn:hover {{
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }}
        
        .tool-btn.active {{
            background: white;
            color: #667eea;
        }}
        
        .tool-btn.danger {{
            background: rgba(244,67,54,0.8);
        }}
        
        .canvas-wrapper {{
            position: relative;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        
        #cocoCanvas {{
            display: block;
            cursor: crosshair;
        }}
        
        .info-panel {{
            margin-top: 15px;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .info-item {{
            flex: 1;
            min-width: 120px;
        }}
        
        .info-label {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .info-value {{
            font-size: 16px;
            font-weight: 700;
            color: #333;
        }}
        
        .label-input {{
            padding: 8px 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 6px;
            font-size: 14px;
            background: white;
            font-weight: 600;
            cursor: pointer;
        }}
        
        .instructions {{
            background: #f0f4ff;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
            border-left: 4px solid #667eea;
            margin-top: 10px;
        }}
    </style>
    
    <div class="editor-container">
        <!-- Toolbar -->
        <div class="toolbar">
            <button class="tool-btn active" id="drawBtn" onclick="setMode('draw')">
                <span>✏️</span> Draw Box
            </button>
            <button class="tool-btn" id="selectBtn" onclick="setMode('select')">
                <span>👆</span> Select/Edit
            </button>
            <button class="tool-btn danger" onclick="deleteSelected()">
                <span>🗑️</span> Delete
            </button>
            <button class="tool-btn danger" onclick="clearAll()">
                <span>🧹</span> Clear All
            </button>
            <button class="tool-btn" onclick="saveBoxes()" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);">
                <span>💾</span> Save Edits
            </button>
            
            <div style="flex: 1;"></div>
            
            <select class="label-input" id="labelSelect">
                <option value="insulators">Insulators</option>
                <option value="crossarm">Crossarm</option>
                <option value="utility-pole">Utility-Pole</option>
            </select>
        </div>
        
        <!-- Canvas -->
        <div class="canvas-wrapper">
            <canvas id="cocoCanvas" width="{img_rgb.width}" height="{img_rgb.height}"></canvas>
        </div>
        
        <!-- Info Panel -->
        <div class="info-panel">
            <div class="info-item">
                <div class="info-label">Mode</div>
                <div class="info-value" id="modeDisplay">Draw</div>
            </div>
            <div class="info-item">
                <div class="info-label">Total Boxes</div>
                <div class="info-value" id="boxCount">0</div>
            </div>
            <div class="info-item">
                <div class="info-label">Selected</div>
                <div class="info-value" id="selectedInfo">None</div>
            </div>
        </div>
        
        <!-- Instructions -->
        <div class="instructions">
            <strong>💡 How to Edit Boxes:</strong><br>
            <strong>Draw:</strong> Click "✏️ Draw Box" and drag on image<br>
            <strong>✋ Move:</strong> Select mode → Click and drag box<br>
            <strong>🔲 Resize:</strong> Select mode → Drag corners or edges<br>
            <strong>Delete:</strong> Press Delete key or use Delete button
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('cocoCanvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = 'data:image/png;base64,{img_base64}';
        
        let boxes = {boxes_json};
        let mode = 'draw';
        let selectedBox = -1;
        let isDragging = false;
        let dragType = null;
        let startX, startY;
        let currentBox = null;
        
        img.onload = () => {{
            render();
            updateInfo();
        }};
        
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            
            // Draw all boxes
            boxes.forEach((box, idx) => {{
                const isSelected = idx === selectedBox;
                ctx.strokeStyle = isSelected ? '#00FF00' : '#FF0000';
                ctx.lineWidth = isSelected ? 6 : 4;
                ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                
                // Draw label
                ctx.fillStyle = isSelected ? '#00FF00' : '#FF0000';
                ctx.font = 'bold 16px Arial';
                const labelText = box.label + ' ' + box.score.toFixed(2);
                ctx.fillText(labelText, box.x1, Math.max(20, box.y1 - 5));
                
                // Draw resize handles if selected
                if (isSelected) {{
                    const handleSize = 12;
                    ctx.fillStyle = '#FFFFFF';
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 2;
                    
                    // Corner handles
                    const corners = [
                        {{x: box.x1, y: box.y1}},
                        {{x: box.x2, y: box.y1}},
                        {{x: box.x1, y: box.y2}},
                        {{x: box.x2, y: box.y2}}
                    ];
                    corners.forEach(c => {{
                        ctx.fillRect(c.x - handleSize/2, c.y - handleSize/2, handleSize, handleSize);
                        ctx.strokeRect(c.x - handleSize/2, c.y - handleSize/2, handleSize, handleSize);
                    }});
                    
                    // Edge handles
                    const edges = [
                        {{x: (box.x1 + box.x2) / 2, y: box.y1}},
                        {{x: (box.x1 + box.x2) / 2, y: box.y2}},
                        {{x: box.x1, y: (box.y1 + box.y2) / 2}},
                        {{x: box.x2, y: (box.y1 + box.y2) / 2}}
                    ];
                    edges.forEach(e => {{
                        ctx.fillRect(e.x - handleSize/2, e.y - handleSize/2, handleSize, handleSize);
                        ctx.strokeRect(e.x - handleSize/2, e.y - handleSize/2, handleSize, handleSize);
                    }});
                }}
            }});
            
            // Draw current box being drawn
            if (currentBox && mode === 'draw') {{
                ctx.strokeStyle = '#0000FF';
                ctx.lineWidth = 3;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(currentBox.x1, currentBox.y1, currentBox.x2 - currentBox.x1, currentBox.y2 - currentBox.y1);
                ctx.setLineDash([]);
            }}
            
            updateInfo();
        }}
        
        function setMode(newMode) {{
            mode = newMode;
            selectedBox = -1;
            document.getElementById('modeDisplay').textContent = newMode === 'draw' ? 'Draw' : 'Select/Edit';
            
            const drawBtn = document.getElementById('drawBtn');
            const selectBtn = document.getElementById('selectBtn');
            
            // Update button styles
            if (newMode === 'draw') {{
                drawBtn.classList.add('active');
                selectBtn.classList.remove('active');
                canvas.style.cursor = 'crosshair';
            }} else {{
                drawBtn.classList.remove('active');
                selectBtn.classList.add('active');
                canvas.style.cursor = 'default';
            }}
            
            render();
        }}
        
        function getMousePos(e) {{
            const rect = canvas.getBoundingClientRect();
            return {{
                x: (e.clientX - rect.left) * (canvas.width / rect.width),
                y: (e.clientY - rect.top) * (canvas.height / rect.height)
            }};
        }}
        
        function getHandle(pos, box) {{
            const handleSize = 12;
            const threshold = handleSize;
            
            // Check corners
            if (Math.abs(pos.x - box.x1) < threshold && Math.abs(pos.y - box.y1) < threshold) return 'resize-nw';
            if (Math.abs(pos.x - box.x2) < threshold && Math.abs(pos.y - box.y1) < threshold) return 'resize-ne';
            if (Math.abs(pos.x - box.x1) < threshold && Math.abs(pos.y - box.y2) < threshold) return 'resize-sw';
            if (Math.abs(pos.x - box.x2) < threshold && Math.abs(pos.y - box.y2) < threshold) return 'resize-se';
            
            // Check edges
            const midX = (box.x1 + box.x2) / 2;
            const midY = (box.y1 + box.y2) / 2;
            if (Math.abs(pos.x - midX) < threshold && Math.abs(pos.y - box.y1) < threshold) return 'resize-n';
            if (Math.abs(pos.x - midX) < threshold && Math.abs(pos.y - box.y2) < threshold) return 'resize-s';
            if (Math.abs(pos.x - box.x1) < threshold && Math.abs(pos.y - midY) < threshold) return 'resize-w';
            if (Math.abs(pos.x - box.x2) < threshold && Math.abs(pos.y - midY) < threshold) return 'resize-e';
            
            // Check if inside box (for moving)
            if (pos.x >= box.x1 && pos.x <= box.x2 && pos.y >= box.y1 && pos.y <= box.y2) return 'move';
            
            return null;
        }}
        
        canvas.addEventListener('mousedown', (e) => {{
            const pos = getMousePos(e);
            
            if (mode === 'draw') {{
                isDragging = true;
                startX = pos.x;
                startY = pos.y;
                currentBox = {{x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y}};
            }} else if (mode === 'select') {{
                // Check if clicking on a box
                for (let i = boxes.length - 1; i >= 0; i--) {{
                    const handle = getHandle(pos, boxes[i]);
                    if (handle) {{
                        selectedBox = i;
                        isDragging = true;
                        dragType = handle;
                        startX = pos.x;
                        startY = pos.y;
                        render();
                        return;
                    }}
                }}
                selectedBox = -1;
                render();
            }}
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            const pos = getMousePos(e);
            
            if (!isDragging) {{
                // Update cursor based on hover
                if (mode === 'select' && selectedBox >= 0) {{
                    const handle = getHandle(pos, boxes[selectedBox]);
                    if (handle) {{
                        if (handle === 'move') canvas.style.cursor = 'move';
                        else if (handle.includes('nw') || handle.includes('se')) canvas.style.cursor = 'nwse-resize';
                        else if (handle.includes('ne') || handle.includes('sw')) canvas.style.cursor = 'nesw-resize';
                        else if (handle.includes('n') || handle.includes('s')) canvas.style.cursor = 'ns-resize';
                        else if (handle.includes('e') || handle.includes('w')) canvas.style.cursor = 'ew-resize';
                    }} else {{
                        canvas.style.cursor = 'default';
                    }}
                }}
                return;
            }}
            
            if (mode === 'draw') {{
                currentBox.x2 = pos.x;
                currentBox.y2 = pos.y;
                render();
            }} else if (mode === 'select' && selectedBox >= 0) {{
                const box = boxes[selectedBox];
                const dx = pos.x - startX;
                const dy = pos.y - startY;
                
                if (dragType === 'move') {{
                    box.x1 += dx;
                    box.y1 += dy;
                    box.x2 += dx;
                    box.y2 += dy;
                }} else if (dragType === 'resize-nw') {{
                    box.x1 = pos.x;
                    box.y1 = pos.y;
                }} else if (dragType === 'resize-ne') {{
                    box.x2 = pos.x;
                    box.y1 = pos.y;
                }} else if (dragType === 'resize-sw') {{
                    box.x1 = pos.x;
                    box.y2 = pos.y;
                }} else if (dragType === 'resize-se') {{
                    box.x2 = pos.x;
                    box.y2 = pos.y;
                }} else if (dragType === 'resize-n') {{
                    box.y1 = pos.y;
                }} else if (dragType === 'resize-s') {{
                    box.y2 = pos.y;
                }} else if (dragType === 'resize-w') {{
                    box.x1 = pos.x;
                }} else if (dragType === 'resize-e') {{
                    box.x2 = pos.x;
                }}
                
                // Ensure x1 < x2 and y1 < y2
                if (box.x1 > box.x2) [box.x1, box.x2] = [box.x2, box.x1];
                if (box.y1 > box.y2) [box.y1, box.y2] = [box.y2, box.y1];
                
                startX = pos.x;
                startY = pos.y;
                render();
            }}
        }});
        
        canvas.addEventListener('mouseup', (e) => {{
            if (mode === 'draw' && isDragging && currentBox) {{
                const width = Math.abs(currentBox.x2 - currentBox.x1);
                const height = Math.abs(currentBox.y2 - currentBox.y1);
                
                if (width > 10 && height > 10) {{
                    boxes.push({{
                        x1: Math.min(currentBox.x1, currentBox.x2),
                        y1: Math.min(currentBox.y1, currentBox.y2),
                        x2: Math.max(currentBox.x1, currentBox.x2),
                        y2: Math.max(currentBox.y1, currentBox.y2),
                        label: document.getElementById('labelSelect').value,
                        score: 1.0,
                        index: boxes.length > 0 ? Math.max(...boxes.map(b => b.index)) + 1 : 0
                    }});
                }}
                currentBox = null;
            }}
            
            isDragging = false;
            dragType = null;
            render();
        }});
        
        function deleteSelected() {{
            if (selectedBox >= 0) {{
                boxes.splice(selectedBox, 1);
                selectedBox = -1;
                render();
            }}
        }}
        
        function clearAll() {{
            if (confirm('Clear all boxes?')) {{
                boxes = [];
                selectedBox = -1;
                render();
            }}
        }}
        
        function saveBoxes() {{
            // Confirmation dialog
            const message = boxes.length === 0 
                ? 'No boxes to save. Do you want to clear all detections?'
                : 'Save ' + boxes.length + ' axis-aligned box(es) and update feedback?\\n\\nThis will overwrite the current detections.';
            
            if (!confirm(message)) {{
                return; // User cancelled
            }}
            
            const savedBoxes = boxes.map(box => ({{
                index: box.index,
                label: box.label,
                score: box.score,
                bbox: [
                    Math.max(0, Math.min(box.x1, box.x2)),
                    Math.max(0, Math.min(box.y1, box.y2)),
                    Math.min({img_rgb.width}, Math.max(box.x1, box.x2)),
                    Math.min({img_rgb.height}, Math.max(box.y1, box.y2))
                ]
            }})));
            
            // Send to Streamlit
            Streamlit.setComponentValue(JSON.stringify(savedBoxes));
            
            // Visual feedback
            const saveBtn = event.currentTarget;
            const originalHTML = saveBtn.innerHTML;
            saveBtn.innerHTML = '✅ Saved!';
            saveBtn.style.background = '#4CAF50';
            setTimeout(() => {{
                saveBtn.innerHTML = originalHTML;
                saveBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            }}, 1500);
        }}
        
        function updateInfo() {{
            document.getElementById('boxCount').textContent = boxes.length;
            if (selectedBox >= 0 && selectedBox < boxes.length) {{
                const box = boxes[selectedBox];
                document.getElementById('selectedInfo').textContent = 'Box ' + (selectedBox + 1) + ' (' + box.label + ')';
            }} else {{
                document.getElementById('selectedInfo').textContent = 'None';
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Delete' && selectedBox >= 0) {{
                deleteSelected();
            }} else if (e.key === 'd' || e.key === 'D') {{
                setMode('draw');
            }} else if (e.key === 's' || e.key === 'S') {{
                setMode('select');
            }}
        }});
        
        // Initialize
        setMode('draw');
    </script>
    """
    
    st.markdown("### 📐 COCO Editor (Axis-Aligned Boxes)")
    st.components.v1.html(html_code, height=min(img_rgb.height + 280, 900))


def display_html5_canvas_editor(img_rgb, boxes, labels, scores, indices):
    """
    HTML5 Canvas implementation with mouse drawing - NO external dependencies!
    Provides full interactive editing: draw, move, resize boxes with mouse
    """
    import base64
    import json
    from io import BytesIO
    
    # Convert image to base64
    buffered = BytesIO()
    img_rgb.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Prepare existing boxes as JSON with indices for feedback tracking
    boxes_json = json.dumps([
        {
            'x1': float(box[0]), 'y1': float(box[1]), 
            'x2': float(box[2]), 'y2': float(box[3]),
            'label': label, 'score': float(score), 'index': int(idx)
        }
        for box, label, score, idx in zip(boxes, labels, scores, indices)
    ])
    
    # Initialize session state for canvas edits
    canvas_key = f"canvas_edits_{hash(img_base64[:100])}"
    if canvas_key not in st.session_state:
        st.session_state[canvas_key] = {'boxes': boxes_json}
    
    # HTML/CSS/JavaScript for interactive canvas
    html_code = f"""
    <div style="border: 2px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; margin-bottom: 20px;">
        <div style="margin-bottom: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <button onclick="setMode('draw')" id="drawBtn" 
                    style="padding: 10px 20px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 6px; font-weight: bold;">
                🖊️ Draw Box
            </button>
            <button onclick="setMode('move')" id="moveBtn" 
                    style="padding: 10px 20px; cursor: pointer; background: #2196F3; color: white; border: none; border-radius: 6px; font-weight: bold;">
                ✋ Move/Resize
            </button>
            <button onclick="deleteSelected()" id="deleteBtn"
                    style="padding: 10px 20px; cursor: pointer; background: #FF9800; color: white; border: none; border-radius: 6px; font-weight: bold;">
                ❌ Delete Selected
            </button>
            <button onclick="clearAll()" 
                    style="padding: 10px 20px; cursor: pointer; background: #f44336; color: white; border: none; border-radius: 6px; font-weight: bold;">
                🗑️ Clear All
            </button>
            <button onclick="saveBoxes()" 
                    style="padding: 10px 20px; cursor: pointer; background: #9C27B0; color: white; border: none; border-radius: 6px; font-weight: bold;">
                💾 Save Changes
            </button>
        </div>
        
        <canvas id="myCanvas" width="{img_rgb.width}" height="{img_rgb.height}" 
                style="border: 2px solid #ccc; cursor: crosshair; background: white; display: block; max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></canvas>
        
        <div id="info" style="margin-top: 15px; padding: 12px; background: white; border-radius: 6px; font-family: 'Segoe UI', sans-serif; font-size: 14px; border-left: 4px solid #2196F3;">
            <strong>Mode: Draw</strong> | Click and drag to create boxes | Press <kbd>Delete</kbd> to remove selected box
        </div>
        
        <div id="boxInfo" style="margin-top: 10px; padding: 10px; background: #e3f2fd; border-radius: 4px; font-size: 12px; display: none;">
            <strong>Selected:</strong> <span id="selectedLabel"></span>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('myCanvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = 'data:image/png;base64,{img_base64}';
        
        let boxes = {boxes_json};
        let mode = 'draw';
        let isDrawing = false;
        let startX, startY;
        let currentBox = null;
        let selectedBox = null;
        let isResizing = false;
        let resizeHandle = null;
        
        const boxColor = '#00FF00';  // Green color for all boxes
        
        img.onload = function() {{
            redraw();
        }};
        
        function redraw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            
            // Draw all boxes in green with solid lines (thicker)
            boxes.forEach((box, idx) => {{
                ctx.strokeStyle = boxColor;
                ctx.lineWidth = 5;  // Increased thickness
                ctx.setLineDash([]);  // Ensure solid lines
                ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                
                // Draw label background
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.fillRect(box.x1, Math.max(0, box.y1 - 20), 150, 18);
                
                // Draw label text
                ctx.fillStyle = boxColor;
                ctx.font = 'bold 14px Arial';
                ctx.fillText(`${{box.label}}: ${{box.score.toFixed(2)}}`, box.x1 + 2, Math.max(15, box.y1 - 5));
                
                // Draw handles if selected
                if (selectedBox === idx) {{
                    drawHandles(box);
                }}
            }});
            
            // Draw current box being drawn
            if (currentBox) {{
                ctx.strokeStyle = '#FF0000';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(currentBox.x1, currentBox.y1, 
                              currentBox.x2 - currentBox.x1, 
                              currentBox.y2 - currentBox.y1);
                ctx.setLineDash([]);
            }}
        }}
        
        function drawHandles(box) {{
            ctx.fillStyle = '#FF0000';
            ctx.strokeStyle = '#FFFFFF';
            const handleSize = 10;
            const handles = [
                {{x: box.x1, y: box.y1, type: 'nw'}},
                {{x: box.x2, y: box.y1, type: 'ne'}},
                {{x: box.x1, y: box.y2, type: 'sw'}},
                {{x: box.x2, y: box.y2, type: 'se'}}
            ];
            handles.forEach(h => {{
                ctx.beginPath();
                ctx.arc(h.x, h.y, handleSize/2, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }});
        }}
        
        function getHandleAt(x, y, box) {{
            const handleSize = 10;
            const handles = [
                {{x: box.x1, y: box.y1, type: 'nw'}},
                {{x: box.x2, y: box.y1, type: 'ne'}},
                {{x: box.x1, y: box.y2, type: 'sw'}},
                {{x: box.x2, y: box.y2, type: 'se'}}
            ];
            for (let h of handles) {{
                if (Math.abs(x - h.x) < handleSize && Math.abs(y - h.y) < handleSize) {{
                    return h.type;
                }}
            }}
            return null;
        }}
        
        function setMode(newMode) {{
            mode = newMode;
            selectedBox = null;
            canvas.style.cursor = mode === 'draw' ? 'crosshair' : 'pointer';
            document.getElementById('info').innerHTML = 
                mode === 'draw' ? 
                '<strong>Mode: Draw</strong> | Click and drag to create boxes | Press <kbd>Delete</kbd> to remove selected box' : 
                '<strong>Mode: Move/Resize</strong> | Click boxes to select, drag to move, drag corners to resize';
            document.getElementById('boxInfo').style.display = 'none';
            redraw();
        }}
        
        function clearAll() {{
            if (confirm('Clear all boxes? This cannot be undone.')) {{
                boxes = [];
                selectedBox = null;
                redraw();
            }}
        }}
        
        function deleteSelected() {{
            if (selectedBox !== null) {{
                boxes.splice(selectedBox, 1);
                selectedBox = null;
                document.getElementById('boxInfo').style.display = 'none';
                redraw();
            }} else {{
                alert('Please select a box first');
            }}
        }}
        
        function saveBoxes() {{
            // Store boxes in a way Streamlit can access
            // This will be handled via Streamlit's component communication
            const boxesJson = JSON.stringify(boxes);
            // Use parent.postMessage to send data to Streamlit
            if (window.parent) {{
                window.parent.postMessage({{
                    type: 'canvas_boxes',
                    boxes: boxesJson
                }}, '*');
            }}
            alert(`Saved ${{boxes.length}} box(es)!`);
        }}
        
        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            
            if (mode === 'draw') {{
                isDrawing = true;
                startX = x;
                startY = y;
                currentBox = {{x1: x, y1: y, x2: x, y2: y}};
            }} else {{
                // Check if clicking on a resize handle first
                if (selectedBox !== null) {{
                    const box = boxes[selectedBox];
                    const handle = getHandleAt(x, y, box);
                    if (handle) {{
                        isResizing = true;
                        resizeHandle = handle;
                        startX = x;
                        startY = y;
                        return;
                    }}
                }}
                
                // Check if clicking on a box
                for (let i = boxes.length - 1; i >= 0; i--) {{
                    const box = boxes[i];
                    if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {{
                        selectedBox = i;
                        startX = x;
                        startY = y;
                        document.getElementById('boxInfo').style.display = 'block';
                        document.getElementById('selectedLabel').textContent = 
                            `${{box.label}} (conf: ${{box.score.toFixed(2)}})`;
                        redraw();
                        return;
                    }}
                }}
                selectedBox = null;
                document.getElementById('boxInfo').style.display = 'none';
                redraw();
            }}
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            
            if (mode === 'draw' && isDrawing) {{
                currentBox.x2 = x;
                currentBox.y2 = y;
                redraw();
            }} else if (mode === 'move' && selectedBox !== null) {{
                const box = boxes[selectedBox];
                
                if (isResizing && resizeHandle) {{
                    // Resize box
                    if (resizeHandle.includes('n')) box.y1 = y;
                    if (resizeHandle.includes('s')) box.y2 = y;
                    if (resizeHandle.includes('w')) box.x1 = x;
                    if (resizeHandle.includes('e')) box.x2 = x;
                    
                    // Ensure valid box
                    if (box.x2 < box.x1) {{ const temp = box.x1; box.x1 = box.x2; box.x2 = temp; }}
                    if (box.y2 < box.y1) {{ const temp = box.y1; box.y1 = box.y2; box.y2 = temp; }}
                }} else if (e.buttons === 1) {{
                    // Move box
                    const dx = x - startX;
                    const dy = y - startY;
                    box.x1 += dx;
                    box.y1 += dy;
                    box.x2 += dx;
                    box.y2 += dy;
                    startX = x;
                    startY = y;
                }}
                
                // Update cursor
                const handle = getHandleAt(x, y, box);
                canvas.style.cursor = handle ? 'nwse-resize' : 'move';
                
                redraw();
            }}
        }});
        
        canvas.addEventListener('mouseup', (e) => {{
            if (mode === 'draw' && isDrawing) {{
                isDrawing = false;
                if (currentBox && Math.abs(currentBox.x2 - currentBox.x1) > 10 && 
                    Math.abs(currentBox.y2 - currentBox.y1) > 10) {{
                    boxes.push({{
                        x1: Math.min(currentBox.x1, currentBox.x2),
                        y1: Math.min(currentBox.y1, currentBox.y2),
                        x2: Math.max(currentBox.x1, currentBox.x2),
                        y2: Math.max(currentBox.y1, currentBox.y2),
                        label: 'New Box',
                        score: 1.0,
                        index: -1
                    }});
                }}
                currentBox = null;
                redraw();
            }}
            isResizing = false;
            resizeHandle = null;
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Delete' && selectedBox !== null) {{
                deleteSelected();
            }}
        }});
    </script>
    """
    
    # Display the HTML canvas
    st.markdown("### ✏️ Interactive Editor (HTML5 Canvas)")
    st.markdown("**Instructions:**")
    st.markdown("- 🖊️ **Draw Mode**: Click and drag to draw new boxes")
    st.markdown("- ✋ **Move/Resize Mode**: Click boxes to select, drag to move, drag corners to resize")
    st.markdown("- ❌ **Delete**: Select a box and click Delete button or press Delete key")
    st.markdown("- 💾 **Save**: Click Save Changes to apply edits")
    
    st.components.v1.html(html_code, height=img_rgb.height + 250)
    
    # Note: HTML5 canvas edits would need to be captured via Streamlit components communication
    # For now, users can still use the coordinate editing below
    st.info("💡 **Note:** Use the box editing interface below to adjust coordinates manually, or use the canvas above for visual editing.")

def display_canvas_editor_alternative(img_rgb, boxes, labels, scores, indices):
    """
    Alternative implementation without streamlit-drawable-canvas
    Uses Streamlit's native image display and PIL ImageDraw for box editing
    """
    from PIL import ImageDraw
    
    # Initialize session state for selected box
    if 'selected_box' not in st.session_state:
        st.session_state.selected_box = None
    
    # Create a copy of the image to draw on
    img_with_boxes = img_rgb.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Draw all boxes with labels in green
    green_color = 'green'
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Draw rectangle in green with solid lines
        draw.rectangle(box, outline=green_color, width=3)
        # Draw label
        text = f"{label}: {score:.2f}"
        # Ensure text doesn't go off image
        text_y = max(0, box[1] - 15)
        draw.text((box[0], text_y), text, fill=green_color)
    
    # Display the image
    st.image(img_with_boxes, use_container_width=True)
    
    # Box editing interface
    st.markdown("### ✏️ Edit Bounding Boxes")
    st.caption("Select a box to edit its coordinates or delete it")
    
    if len(boxes) == 0:
        st.info("No boxes to edit. All detections have been removed.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Select box to edit
        box_options = [f"Box {i+1}: {label} (conf: {score:.2f})" for i, (label, score) in enumerate(zip(labels, scores))]
        selected = st.selectbox("Select box to edit:", box_options, key="box_selector")
        selected_idx = int(selected.split(":")[0].split()[1]) - 1  # Convert to 0-indexed
        st.session_state.selected_box = selected_idx
    
    with col2:
        if st.button("🗑️ Delete Selected Box", type="secondary", use_container_width=True):
            # Mark for removal in feedback
            det_key = f"det_{indices[selected_idx]}"
            if det_key not in st.session_state.feedback_data.get('detection_feedback', {}):
                st.session_state.feedback_data['detection_feedback'][det_key] = {
                    "index": indices[selected_idx],
                    "action": "remove",
                    "label": labels[selected_idx],
                    "bbox": boxes[selected_idx],
                    "confidence": scores[selected_idx]
                }
            else:
                st.session_state.feedback_data['detection_feedback'][det_key]['action'] = "remove"
            st.success("Box marked for removal. Changes will be saved with feedback.")
            st.rerun()
    
    # Edit selected box coordinates
    if st.session_state.selected_box is not None:
        idx = st.session_state.selected_box
        if idx < len(boxes):
            st.markdown(f"**Editing: {labels[idx]}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            current_box = boxes[idx]
            img_width, img_height = img_rgb.size
            
            with col1:
                x1 = st.number_input("X Min", value=int(current_box[0]), min_value=0, max_value=img_width, key=f"x1_{idx}")
            with col2:
                y1 = st.number_input("Y Min", value=int(current_box[1]), min_value=0, max_value=img_height, key=f"y1_{idx}")
            with col3:
                x2 = st.number_input("X Max", value=int(current_box[2]), min_value=0, max_value=img_width, key=f"x2_{idx}")
            with col4:
                y2 = st.number_input("Y Max", value=int(current_box[3]), min_value=0, max_value=img_height, key=f"y2_{idx}")
            
            # Validate box
            if x2 > x1 and y2 > y1:
                if st.button("💾 Update Box", type="primary", use_container_width=True):
                    new_bbox = [float(x1), float(y1), float(x2), float(y2)]
                    # Update feedback
                    det_key = f"det_{indices[idx]}"
                    if det_key not in st.session_state.feedback_data.get('detection_feedback', {}):
                        st.session_state.feedback_data['detection_feedback'][det_key] = {
                            "index": indices[idx],
                            "action": "wrong_box",
                            "label": labels[idx],
                            "bbox": boxes[idx],
                            "confidence": scores[idx]
                        }
                    st.session_state.feedback_data['detection_feedback'][det_key]['corrected_bbox'] = new_bbox
                    st.session_state.feedback_data['detection_feedback'][det_key]['action'] = "wrong_box"
                    st.success("✅ Box updated! Changes will be saved with feedback.")
                    st.rerun()
            else:
                st.warning("⚠️ Invalid box: X Max must be > X Min and Y Max must be > Y Min")

def display_canvas_editor(img_rgb, boxes, labels, scores, indices):
    """Display interactive canvas for visual bounding box editing."""
    if not HAS_CANVAS:
        return
    
    # CRITICAL FIX: Convert numpy array to PIL Image if needed
    # st_canvas expects PIL.Image.Image, not numpy array
    # This fixes: ValueError: The truth value of an array with more than one element is ambiguous
    if isinstance(img_rgb, np.ndarray):
        # Convert numpy array to PIL Image
        # Handle different array formats: uint8 [0-255] or float [0-1]
        if img_rgb.dtype == np.uint8:
            background_image = Image.fromarray(img_rgb, mode='RGB')
        elif img_rgb.max() <= 1.0:
            # Float array in [0, 1] range - convert to uint8
            img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
            background_image = Image.fromarray(img_rgb_uint8, mode='RGB')
        else:
            # Float array in [0, 255] range - convert to uint8
            background_image = Image.fromarray(img_rgb.astype(np.uint8), mode='RGB')
    elif isinstance(img_rgb, Image.Image):
        # Already a PIL Image - use directly
        background_image = img_rgb
    else:
        raise TypeError(f"img_rgb must be PIL.Image.Image or numpy.ndarray, got {type(img_rgb)}")
    
    # Ensure it's RGB mode
    if background_image.mode != 'RGB':
        background_image = background_image.convert('RGB')
    
    img_width, img_height = background_image.size
    
    # Convert boxes to canvas format (x, y, width, height)
    # Canvas uses top-left corner + width/height, our boxes are [xmin, ymin, xmax, ymax]
    canvas_objects = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        # Color based on label
        color_map = {
            "insulators": "#1f77b4",  # Blue
            "crossarm": "#ff7f0e",    # Orange
            "utility-pole": "#2ca02c" # Green
        }
        color = color_map.get(label, "#d62728")  # Default red
        
        canvas_objects.append({
            "type": "rect",
            "x": float(xmin),
            "y": float(ymin),
            "width": float(width),
            "height": float(height),
            "stroke": color,
            "strokeWidth": 2,
            "fill": "transparent",
            "left": float(xmin),
            "top": float(ymin),
            "scaleX": 1.0,
            "scaleY": 1.0,
            "angle": 0,
            "skewX": 0,
            "skewY": 0,
            "rx": 0,
            "ry": 0,
            "objectId": f"det_{indices[i]}",  # Use original index
            "label": label,
            "confidence": float(score)
        })
    
    # Display canvas
    st.write("**✏️ Interactive Editor:** Drag, resize, or delete boxes directly on the image")
    st.caption("💡 Tip: Click and drag boxes to move them, drag corners to resize. Double-click to delete.")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
        stroke_width=2,
        stroke_color="#FF0000",  # Red stroke
        background_image=background_image,  # PIL Image (not numpy array)
        height=min(800, img_height),  # Limit height for display
        width=min(1200, img_width),   # Limit width for display
        drawing_mode="transform",  # Allow moving/resizing existing objects
        point_display_radius=3 if st.session_state.get("show_points", False) else 0,
        key="detection_canvas",
        initial_drawing={
            "version": "4.4.0",
            "objects": canvas_objects
        } if canvas_objects else None,
        display_toolbar=True,
    )
    
    # Process canvas edits
    if canvas_result.json_data is not None:
        canvas_objects_updated = canvas_result.json_data.get("objects", [])
        
        # Track which detections are still in canvas (not deleted)
        active_det_ids = set()
        
        # Update session state with canvas edits
        for obj in canvas_objects_updated:
            if obj.get("type") == "rect":
                det_id = obj.get("objectId", "")
                if det_id.startswith("det_"):
                    active_det_ids.add(det_id)
                    try:
                        det_index = int(det_id.split("_")[1])
                        det_key = f"det_{det_index}"
                        
                        # Get current feedback or create new
                        if det_key not in st.session_state.feedback_data.get('detection_feedback', {}):
                            st.session_state.feedback_data['detection_feedback'][det_key] = {
                                "index": det_index,
                                "action": None,
                                "label": obj.get("label", "unknown"),
                                "bbox": [0, 0, 0, 0],
                                "confidence": obj.get("confidence", 0.0)
                            }
                        
                        # Extract bbox from canvas object
                        # Canvas uses x, y, width, height
                        x = obj.get("left", obj.get("x", 0))
                        y = obj.get("top", obj.get("y", 0))
                        width = obj.get("width", 0) * obj.get("scaleX", 1.0)
                        height = obj.get("height", 0) * obj.get("scaleY", 1.0)
                        
                        # Convert to [xmin, ymin, xmax, ymax]
                        xmin = max(0, min(x, img_width))
                        ymin = max(0, min(y, img_height))
                        xmax = max(xmin + 1, min(x + width, img_width))
                        ymax = max(ymin + 1, min(y + height, img_height))
                        
                        new_bbox = [xmin, ymin, xmax, ymax]
                        
                        # Update feedback with corrected bbox
                        current_feedback = st.session_state.feedback_data['detection_feedback'][det_key]
                        current_feedback['corrected_bbox'] = new_bbox
                        if current_feedback.get('action') is None:
                            current_feedback['action'] = "wrong_box"
                        
                    except (ValueError, IndexError):
                        continue
        
        # Mark deleted detections (removed from canvas) as "remove"
        original_det_ids = {obj.get("objectId") for obj in canvas_objects if obj.get("objectId", "").startswith("det_")}
        deleted_det_ids = original_det_ids - active_det_ids
        
        for det_id in deleted_det_ids:
            try:
                det_index = int(det_id.split("_")[1])
                det_key = f"det_{det_index}"
                if det_key in st.session_state.feedback_data.get('detection_feedback', {}):
                    st.session_state.feedback_data['detection_feedback'][det_key]['action'] = "remove"
            except (ValueError, IndexError):
                continue
        
        # Show update message if canvas was edited
        if len(canvas_objects_updated) != len(canvas_objects):
            st.info("💾 Canvas edits saved! Boxes have been updated.")
    
    # Show canvas controls
    with st.expander("🎨 Canvas Controls"):
        st.write("**How to use:**")
        st.write("- **Move box:** Click and drag the box")
        st.write("- **Resize box:** Click and drag the corners/edges")
        st.write("- **Delete box:** Select box and press Delete key, or use Remove button below")
        st.write("- **Add new box:** Switch to 'Rect' mode in toolbar, draw new box")
        
        if st.button("🔄 Reset Canvas to Original Detections"):
            # Clear canvas edits from session state
            for det_key in list(st.session_state.feedback_data.get('detection_feedback', {}).keys()):
                feedback = st.session_state.feedback_data['detection_feedback'][det_key]
                if isinstance(feedback, dict):
                    feedback.pop('corrected_bbox', None)
                    if feedback.get('action') == 'wrong_box' and 'corrected_label' not in feedback:
                        feedback['action'] = None
            st.rerun()

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
    
    # CRITICAL: Model outputs indices 1, 2, 3 (category IDs)
    # Map to class names: 1=insulators, 2=crossarm, 3=utility-pole
    # But our UTILITY_CLASSES array is 0-indexed, so subtract 1
    class_indices = (class_ids_filtered - 1).tolist()  # Convert 1,2,3 → 0,1,2
    
    # DEBUG: Verify mapping
    # print(f"DEBUG: class_ids_filtered={class_ids_filtered}, class_indices={class_indices}")
    
    # During training, boxes were normalized to TRANSFORMED image size (after resize to 800px)
    # So during inference: normalize → transformed pixels → original pixels
    
    orig_w, orig_h = original_size
    
    # Convert normalized cxcywh [0,1] to xyxy [0,1] (relative to transformed image)
    boxes_xyxy_norm = box_cxcywh_to_xyxy(boxes_norm)
    
    # Scale from normalized [0,1] to TRANSFORMED pixel coordinates
    boxes_transformed = []
    for box in boxes_xyxy_norm:
        x1, y1, x2, y2 = box.cpu().tolist()
        boxes_transformed.append([
            x1 * transformed_w,
            y1 * transformed_h,
            x2 * transformed_w,
            y2 * transformed_h
        ])
    
    # Scale from transformed coordinates to original coordinates
    scale_x = orig_w / transformed_w
    scale_y = orig_h / transformed_h
    
    boxes = []
    for box in boxes_transformed:
        boxes.append([
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ])
    
    # Clamp to image boundaries
    boxes_clamped = []
    for b in boxes:
        x1 = max(0, min(b[0], orig_w))
        y1 = max(0, min(b[1], orig_h))
        x2 = min(max(x1 + 1, b[2]), orig_w)
        y2 = min(max(y1 + 1, b[3]), orig_h)
        boxes_clamped.append([x1, y1, x2, y2])
    
    boxes = boxes_clamped
    
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
    # Initialize session state for feedback (must be done early)
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = {
            'overall_feedback': None,
            'detection_feedback': {},
            'user_notes': ''
        }
    
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
    
    # Set background image if available (lazy load - not at import time)
    bg_path = Path(__file__).parent / "background.png"
    if bg_path.exists():
        set_page_bg(str(bg_path))
    
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

    # ============================================================================
    # MODULE 2: DETECTION RESULTS
    # ============================================================================
    st.markdown("---")
    st.markdown("## 📊 Module 2: Detection Results")
    
    # Display image with detections using matplotlib
    fig = draw_boxes_matplotlib(img_rgb, boxes, labels, scores, title=title)
    st.pyplot(fig)
    
    # Detection summary
    st.markdown("### Detection Summary")
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    for cls, count in sorted(class_counts.items()):
        st.write(f"**{cls}**: {count} detection(s)")
    
    # Show raw detections (expandable)
    with st.expander("Show raw detections"):
        import pandas as pd
        df = pd.DataFrame({
            "index": list(range(len(boxes))),
            "label": labels,
            "confidence": [float(x) for x in scores],
            "xmin": [int(b[0]) for b in boxes],
            "ymin": [int(b[1]) for b in boxes],
            "xmax": [int(b[2]) for b in boxes],
            "ymax": [int(b[3]) for b in boxes],
        }).sort_values("confidence", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
    
    # ============================================================================
    # MODULE 3: CANVAS EDITOR (HITL) + EXPORT FEEDBACK
    # ============================================================================
    st.markdown("---")
    st.markdown("## ✏️ Module 3: Canvas Editor (Human-In-The-Loop)")
    st.caption("Edit bounding boxes, labels, and provide feedback to improve the model")
    
    # OBB Editor for editing
    st.markdown("### 🎨 Edit Detections")
    
    # Initialize session state for edited boxes
    if 'obb_edited_data' not in st.session_state:
        st.session_state.obb_edited_data = None
    
    # Display the OBB editor (captures return value from Save button)
    display_advanced_obb_editor(img_rgb, boxes, labels, scores, list(range(len(boxes))))
    
    # Export Annotations Section
    st.markdown("---")
    st.markdown("### 💾 Export Annotations")
    
    # Status indicator
    if st.session_state.obb_edited_data:
        rotated_count = sum(1 for box in st.session_state.obb_edited_data if box['obb']['angle'] != 0)
        if rotated_count > 0:
            st.success(f"✅ **Using Edited Data** ({len(st.session_state.obb_edited_data)} box(es), {rotated_count} with rotation)")
        else:
            st.info(f"ℹ️ **Using Edited Data** ({len(st.session_state.obb_edited_data)} box(es), no rotation)")
        
        if st.button("🔄 Reset to Original Detections", key="reset_to_original"):
            st.session_state.obb_edited_data = None
            st.rerun()
    else:
        st.warning("⚠️ **Using Original Detections** (no rotation angles)")
    
    # Determine which data to use for export
    if st.session_state.obb_edited_data:
        # Use edited data from OBB canvas
        edited_boxes = st.session_state.obb_edited_data
        st.success(f"✅ Using {len(edited_boxes)} edited box(es) from canvas")
        
        # Show rotation info
        rotated_count = sum(1 for box in edited_boxes if box['obb']['angle'] != 0)
        if rotated_count > 0:
            st.info(f"🔄 {rotated_count} box(es) have rotation angles")
            with st.expander("📐 View Rotation Angles"):
                for i, box in enumerate(edited_boxes):
                    angle_rad = box['obb']['angle']
                    angle_deg = angle_rad * 180 / 3.14159
                    if angle_rad != 0:
                        st.write(f"**Box {i+1} ({box['label']})**: {angle_deg:.1f}° ({angle_rad:.4f} rad)")
        else:
            st.caption("All boxes are axis-aligned (no rotation)")
        
        # Prepare detections from edited OBB data
        detections_data = []
        for item in edited_boxes:
            # Get axis-aligned bbox
            bbox = item['bbox']
            obb = item['obb']
            
            detections_data.append({
                'id': item.get('index', 0) + 1,
                'bbox': bbox,
                'label': item['label'],
                'score': float(item['score']),
                'category_id': UTILITY_CLASSES.index(item['label']) + 1 if item['label'] in UTILITY_CLASSES else 1,
                'obb': obb  # Include OBB data with rotation
            })
    else:
        # Use original model predictions
        st.info("💡 **Tip:** Edit boxes in the canvas above and click 'Save Edits' to export with rotation angles")
        
        # Prepare detections from original predictions
        detections_data = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            detections_data.append({
                'id': i + 1,
                'bbox': box,
                'label': label,
                'score': float(score),
                'category_id': UTILITY_CLASSES.index(label) + 1 if label in UTILITY_CLASSES else 1,
                'obb': {
                    'cx': (x1 + x2) / 2,
                    'cy': (y1 + y2) / 2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'angle': 0.0
                }
            })
    
    # Image info for COCO format
    image_info = {
        'id': 1,
        'file_name': uploaded.name if hasattr(uploaded, 'name') else 'image.jpg',
        'width': img_rgb.width,
        'height': img_rgb.height
    }
    
    # Category info for COCO format
    categories = [
        {'id': i+1, 'name': name, 'supercategory': 'utility'}
        for i, name in enumerate(UTILITY_CLASSES)
    ]
    
    # Export buttons
    st.markdown("#### 📤 Choose Export Format:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export COCO JSON", type="primary", use_container_width=True):
            import json
            from datetime import datetime
            
            # Build COCO format
            coco_data = {
                'info': {
                    'description': 'Utility Inventory Detection - Edited Annotations',
                    'version': '1.0',
                    'date_created': datetime.now().isoformat()
                },
                'images': [image_info],
                'annotations': [],
                'categories': categories
            }
            
            # Add annotations
            for det in detections_data:
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1
                
                coco_data['annotations'].append({
                    'id': det['id'],
                    'image_id': 1,
                    'category_id': det['category_id'],
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'area': float(width * height),
                    'iscrowd': 0,
                    'score': det['score']
                })
            
            # Create download
            coco_json = json.dumps(coco_data, indent=2)
            st.download_button(
                label="⬇️ Download COCO JSON",
                data=coco_json,
                file_name=f"annotations_coco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ COCO format ready for download!")
    
    with col2:
        if st.button("📥 Export OBB JSON", use_container_width=True):
            import json
            from datetime import datetime
            
            # Build OBB format (with rotation support)
            obb_data = {
                'info': {
                    'description': 'Utility Inventory Detection - OBB Annotations',
                    'version': '1.0',
                    'format': 'obb',
                    'date_created': datetime.now().isoformat()
                },
                'images': [image_info],
                'annotations': [],
                'categories': categories
            }
            
            # Add annotations (use OBB data with actual rotation angles from editor)
            for det in detections_data:
                # Use the OBB data that includes rotation angle
                obb_info = det['obb']
                
                obb_data['annotations'].append({
                    'id': det['id'],
                    'image_id': 1,
                    'category_id': det['category_id'],
                    'obb': {
                        'cx': float(obb_info['cx']),
                        'cy': float(obb_info['cy']),
                        'width': float(obb_info['width']),
                        'height': float(obb_info['height']),
                        'angle': float(obb_info['angle'])  # Actual rotation angle in radians from editor
                    },
                    'score': det['score']
                })
            
            # Create download
            obb_json = json.dumps(obb_data, indent=2)
            st.download_button(
                label="⬇️ Download OBB JSON",
                data=obb_json,
                file_name=f"annotations_obb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ OBB format ready for download!")
    
    # Info about export formats
    with st.expander("ℹ️ About Export Formats"):
        st.markdown("""
        **COCO JSON**: Standard format for object detection. Compatible with:
        - PyTorch DETR training
        - Detectron2
        - MMDetection
        - Most modern detection frameworks
        
        **OBB JSON**: Oriented Bounding Box format with rotation support. Use for:
        - Rotated object detection
        - Aerial image detection
        - Custom training pipelines that support rotation
        - Includes rotation angles in radians for precise orientation
        
        **💡 Tip**: After exporting, merge these annotations with your existing dataset for retraining!
        """)

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

