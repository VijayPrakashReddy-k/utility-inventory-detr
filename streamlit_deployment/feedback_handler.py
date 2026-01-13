"""
Human-In-The-Loop (HITL) Feedback Handler

Stores and manages user feedback for model improvement.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import base64
from PIL import Image
import io


class FeedbackHandler:
    """Handle collection and storage of user feedback."""
    
    def __init__(self, feedback_dir: str = "feedback"):
        """
        Initialize feedback handler.
        
        Args:
            feedback_dir: Directory to store feedback files
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate hash for image to use as unique identifier."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return hashlib.md5(img_bytes.read()).hexdigest()
    
    def save_feedback(
        self,
        image: Image.Image,
        detections: List[Dict],
        overall_feedback: Optional[str] = None,
        detection_feedback: Optional[List[Dict]] = None,
        user_notes: Optional[str] = None
    ) -> str:
        """
        Save user feedback for an image and its detections.
        
        Args:
            image: PIL Image object
            detections: List of detection dicts with keys: label, confidence, bbox (xmin, ymin, xmax, ymax)
            overall_feedback: Overall feedback for the image ("correct", "incorrect", "partial")
            detection_feedback: List of feedback for individual detections
            user_notes: Optional user notes
            
        Returns:
            Path to saved feedback file
        """
        image_hash = self._get_image_hash(image)
        timestamp = datetime.now().isoformat()
        
        # Convert image to base64 for storage
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        feedback_data = {
            "timestamp": timestamp,
            "image_hash": image_hash,
            "image_base64": image_base64,
            "image_size": image.size,  # (width, height)
            "detections": detections,
            "overall_feedback": overall_feedback,
            "detection_feedback": detection_feedback or [],
            "user_notes": user_notes,
            "feedback_id": f"{image_hash}_{timestamp.replace(':', '-').split('.')[0]}"
        }
        
        # Save to JSON file
        feedback_file = self.feedback_dir / f"feedback_{feedback_data['feedback_id']}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return str(feedback_file)
    
    def load_all_feedback(self) -> List[Dict]:
        """Load all feedback files."""
        feedback_files = list(self.feedback_dir.glob("feedback_*.json"))
        all_feedback = []
        
        for file in feedback_files:
            try:
                with open(file, 'r') as f:
                    all_feedback.append(json.load(f))
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        return sorted(all_feedback, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def export_to_coco(self, output_path: Optional[str] = None) -> str:
        """
        Export feedback to COCO format for retraining.
        
        Only exports feedback marked as corrections (incorrect detections that were fixed).
        
        Args:
            output_path: Path to save COCO JSON file
            
        Returns:
            Path to exported COCO file
        """
        if output_path is None:
            output_path = self.feedback_dir / "feedback_coco_export.json"
        else:
            output_path = Path(output_path)
        
        all_feedback = self.load_all_feedback()
        
        # Build COCO format
        coco_data = {
            "info": {
                "description": "HITL Feedback Export",
                "version": "1.0",
                "year": datetime.now().year
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "insulators", "supercategory": "utility"},
                {"id": 2, "name": "crossarm", "supercategory": "utility"},
                {"id": 3, "name": "utility-pole", "supercategory": "utility"}
            ]
        }
        
        category_map = {
            "insulators": 1,
            "crossarm": 2,
            "utility-pole": 3
        }
        
        image_id = 1
        annotation_id = 1
        
        for feedback in all_feedback:
            # Only export if user provided corrections
            if feedback.get('overall_feedback') != 'incorrect':
                continue
            
            # Add image
            width, height = feedback['image_size']
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": f"feedback_{feedback['feedback_id']}.png",
                "license": 0,
                "date_captured": feedback['timestamp']
            })
            
            # Add corrected annotations from detection_feedback
            for det_feedback in feedback.get('detection_feedback', []):
                if det_feedback.get('action') == 'correct':
                    # User corrected this detection
                    label = det_feedback.get('corrected_label', det_feedback.get('label'))
                    bbox = det_feedback.get('corrected_bbox', det_feedback.get('bbox'))
                    
                    if label in category_map and bbox:
                        xmin, ymin, xmax, ymax = bbox
                        # COCO format: [x_min, y_min, width, height]
                        coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[label],
                            "bbox": coco_bbox,
                            "area": (xmax - xmin) * (ymax - ymin),
                            "iscrowd": 0
                        })
                        annotation_id += 1
            
            image_id += 1
        
        # Save COCO file
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return str(output_path)
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback."""
        all_feedback = self.load_all_feedback()
        
        stats = {
            "total_feedback": len(all_feedback),
            "overall_correct": 0,
            "overall_incorrect": 0,
            "overall_partial": 0,
            "detections_reviewed": 0,
            "detections_corrected": 0,
            "detections_removed": 0
        }
        
        for feedback in all_feedback:
            overall = feedback.get('overall_feedback', '')
            if overall == 'correct':
                stats["overall_correct"] += 1
            elif overall == 'incorrect':
                stats["overall_incorrect"] += 1
            elif overall == 'partial':
                stats["overall_partial"] += 1
            
            for det_feedback in feedback.get('detection_feedback', []):
                stats["detections_reviewed"] += 1
                action = det_feedback.get('action', '')
                if action == 'correct':
                    stats["detections_corrected"] += 1
                elif action == 'remove':
                    stats["detections_removed"] += 1
        
        return stats

