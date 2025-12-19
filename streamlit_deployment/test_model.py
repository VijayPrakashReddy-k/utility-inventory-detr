#!/usr/bin/env python3
"""
Test the trained DETR model on a test image and compare with ground truth.
"""

import os
# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import json
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from detr_model import DETRdemo

# Utility classes
UTILITY_CLASSES = ['insulators', 'crossarm', 'utility-pole']

# Transform
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def box_cxcywh_to_xyxy(x):
    """Convert boxes from center-size format to xyxy format."""
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=1)

def rescale_bboxes(out_bbox, size):
    """Rescale bounding boxes from normalized [0,1] to image pixel coordinates."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b * scale

def load_model():
    """Load the trained model."""
    model_path = Path(__file__).parent.parent / "models" / "final" / "best_model.pth"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    model = DETRdemo(num_classes=3, hidden_dim=256, nheads=8, 
                     num_encoder_layers=6, num_decoder_layers=6)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model

def test_image(model, image_path, annotations_file, prob_thresh=0.3):
    """Test model on a single image."""
    print(f"\n{'='*70}")
    print(f"Testing: {image_path.name}")
    print(f"{'='*70}")
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    print(f"Image size: {img.size}")
    
    # Load ground truth annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Find annotations for this image
    image_info = None
    for img_info in coco_data['images']:
        if img_info['file_name'] == image_path.name:
            image_info = img_info
            break
    
    if not image_info:
        print(f"WARNING: Image {image_path.name} not found in annotations")
        return
    
    img_id = image_info['id']
    gt_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    print(f"\n📊 Ground Truth:")
    print(f"  Number of objects: {len(gt_annotations)}")
    for ann in gt_annotations:
        cat_id = ann['category_id']
        cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), 'unknown')
        bbox = ann['bbox']  # [x, y, w, h]
        print(f"    - {cat_name} (id={cat_id}): bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    
    # Run inference
    tensor = transform(img).unsqueeze(0)
    
    with torch.inference_mode():
        outputs = model(tensor)
    
    # Process predictions
    pred_logits = outputs["pred_logits"][0]  # [100, 4]
    pred_boxes = outputs["pred_boxes"][0]    # [100, 4] normalized cxcywh
    
    probas = pred_logits.softmax(-1)  # [100, 4]
    class_probs = probas[:, :-1]  # Remove "no object" class [100, 3]
    max_probs, class_ids = class_probs.max(dim=1)  # [100], [100]
    
    # Filter by threshold
    keep = max_probs > prob_thresh
    
    print(f"\n🔍 Model Predictions (threshold={prob_thresh}):")
    print(f"  Detections above threshold: {keep.sum().item()}")
    
    if keep.sum() > 0:
        boxes_norm = pred_boxes[keep]
        scores = max_probs[keep].cpu().numpy()
        class_ids_filtered = class_ids[keep].cpu().numpy()
        
        boxes = rescale_bboxes(boxes_norm, img.size).tolist()
        # Clamp boxes to image boundaries
        img_w, img_h = img.size
        boxes = [[max(0, min(b[0], img_w)), max(0, min(b[1], img_h)), 
                  max(0, min(b[2], img_w)), max(0, min(b[3], img_h))] for b in boxes]
        # Map model indices (1,2,3) to class array indices (0,1,2)
        class_indices = class_ids_filtered - 1
        labels = [UTILITY_CLASSES[i] for i in class_indices]
        
        print(f"\n  Detected objects:")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            print(f"    {i+1}. {label}: confidence={score:.3f}, bbox=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    else:
        print(f"  ⚠️  No detections above threshold!")
        # Show top 5 predictions
        top_k = 5
        top_indices = max_probs.topk(min(top_k, len(max_probs))).indices
        print(f"\n  Top {top_k} predictions (below threshold):")
        for idx in top_indices:
            cls_id = class_ids[idx].item()
            prob = max_probs[idx].item()
            box_norm = pred_boxes[idx]
            box = rescale_bboxes(box_norm.unsqueeze(0), img.size)[0].tolist()
            print(f"    - {UTILITY_CLASSES[cls_id]}: {prob:.3f}, bbox=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Show all predictions (top 10)
    print(f"\n📈 All Predictions (top 10):")
    top_10_indices = max_probs.topk(min(10, len(max_probs))).indices
    for idx in top_10_indices:
        cls_id = class_ids[idx].item()
        prob = max_probs[idx].item()
        box_norm = pred_boxes[idx]
        box = rescale_bboxes(box_norm.unsqueeze(0), img.size)[0].tolist()
        status = "✓" if prob > prob_thresh else " "
        print(f"  {status} {UTILITY_CLASSES[cls_id]}: {prob:.4f}, bbox=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

def main():
    """Main test function."""
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Find test images
    test_dir = Path(__file__).parent.parent / "datasets" / "processed" / "merged" / "test"
    annotations_file = test_dir / "_annotations.coco.json"
    
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return
    
    if not annotations_file.exists():
        print(f"ERROR: Annotations file not found: {annotations_file}")
        return
    
    # Get test images
    test_images = list(test_dir.glob("*.jpg"))[:3]  # Test first 3 images
    
    if not test_images:
        print(f"ERROR: No test images found in {test_dir}")
        return
    
    print(f"\n🧪 Testing on {len(test_images)} images...")
    
    # Test each image
    for img_path in test_images:
        test_image(model, img_path, annotations_file, prob_thresh=0.3)
    
    print(f"\n{'='*70}")
    print("Test complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

