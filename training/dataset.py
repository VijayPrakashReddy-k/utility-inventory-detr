"""
COCO dataset loader for DETR training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F


class CocoDetection(Dataset):
    """
    COCO format dataset for object detection.
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms=None,
        return_masks=False,
    ):
        """
        Args:
            img_folder: Path to folder containing images
            ann_file: Path to COCO format JSON annotation file
            transforms: Optional transform pipeline
            return_masks: Whether to return masks (for panoptic, not used here)
        """
        self.img_folder = Path(img_folder)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        self.return_masks = return_masks
        
        # Load annotations
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build mappings
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Create list of image IDs
        self.ids = list(self.images.keys())
        
        print(f"Loaded {len(self.ids)} images from {self.ann_file}")
        print(f"Categories: {[c['name'] for c in self.categories.values()]}")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.img_folder / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.annotations.get(img_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
        
        # Create target dict (DETR format)
        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'iscrowd': iscrowd,
        }
        
        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


class Compose:
    """Compose transforms that handle both image and target."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL image and target to tensors."""
    def __call__(self, image, target):
        # Convert image to tensor
        image = F.to_tensor(image)
        
        # Normalize image
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and adjust boxes."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)
            if 'boxes' in target and len(target['boxes']) > 0:
                w = image.width
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # Flip x coordinates
                target['boxes'] = boxes
        return image, target


class Resize:
    """Resize image and adjust boxes."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        image = F.resize(image, self.size)
        new_w, new_h = image.size
        
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            # Scale boxes to new image size
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            
            # CRITICAL: Normalize boxes to [0, 1] range (DETR requirement)
            boxes[:, 0] /= new_w
            boxes[:, 2] /= new_w
            boxes[:, 1] /= new_h
            boxes[:, 3] /= new_h
            
            # Clamp to [0, 1] to ensure valid range
            boxes = torch.clamp(boxes, 0.0, 1.0)
            target['boxes'] = boxes
        
        return image, target


class ColorJitter:
    """Apply color jitter augmentation."""
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image, target):
        import random
        from torchvision.transforms import functional as F_transform
        
        if random.random() < 0.5:  # Apply 50% of the time
            image = F_transform.adjust_brightness(image, 1.0 + random.uniform(-self.brightness, self.brightness))
            image = F_transform.adjust_contrast(image, 1.0 + random.uniform(-self.contrast, self.contrast))
            image = F_transform.adjust_saturation(image, 1.0 + random.uniform(-self.saturation, self.saturation))
            image = F_transform.adjust_hue(image, random.uniform(-self.hue, self.hue))
        
        return image, target


def make_coco_transforms(image_set: str, image_size: int = 800):
    """
    Create transform pipeline for COCO dataset.
    
    Args:
        image_set: 'train' or 'val'
        image_size: Target image size
    """
    transforms = []
    
    if image_set == 'train':
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(ColorJitter())  # Add color augmentation for training
    
    transforms.append(Resize(image_size))
    transforms.append(ToTensor())
    
    return Compose(transforms)


class CocoCollate:
    """
    Custom collate function for COCO dataset.
    Handles variable number of objects per image.
    """
    
    def __call__(self, batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, targets

