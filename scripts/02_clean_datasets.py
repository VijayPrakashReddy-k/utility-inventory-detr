#!/usr/bin/env python3
"""
Clean datasets: Remove null labels and validate annotations.

This script:
1. Removes annotations with 'null' category
2. Validates bounding boxes are within image bounds
3. Reports statistics about each dataset
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

def load_coco_json(json_path: Path) -> Dict:
    """Load COCO format JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_coco_json(data: Dict, json_path: Path):
    """Save COCO format JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def clean_annotations(coco_data: Dict, remove_null: bool = True) -> Tuple[Dict, Dict]:
    """
    Clean COCO dataset annotations.
    
    Returns:
        cleaned_data: Cleaned COCO data
        stats: Statistics about cleaning process
    """
    stats = {
        'total_images': len(coco_data.get('images', [])),
        'total_annotations_before': len(coco_data.get('annotations', [])),
        'null_annotations_removed': 0,
        'invalid_boxes_removed': 0,
        'categories': Counter(),
    }
    
    # Find null category ID
    null_category_id = None
    for cat in coco_data.get('categories', []):
        if cat.get('name', '').lower() in ['null', 'none', 'background']:
            null_category_id = cat['id']
            break
    
    # Clean categories (remove null if found)
    cleaned_categories = []
    for cat in coco_data.get('categories', []):
        if cat['id'] != null_category_id:
            cleaned_categories.append(cat)
    
    # Clean annotations
    cleaned_annotations = []
    image_dict = {img['id']: img for img in coco_data.get('images', [])}
    
    for ann in coco_data.get('annotations', []):
        # Remove null category annotations
        if remove_null and ann['category_id'] == null_category_id:
            stats['null_annotations_removed'] += 1
            continue
        
        # Validate bounding box
        img = image_dict.get(ann['image_id'])
        if not img:
            stats['invalid_boxes_removed'] += 1
            continue
        
        # COCO format: [x, y, width, height]
        x, y, w, h = ann['bbox']
        img_w, img_h = img['width'], img['height']
        
        # Check if box is within image bounds
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            stats['invalid_boxes_removed'] += 1
            continue
        
        # Check if box has valid dimensions
        if w <= 0 or h <= 0:
            stats['invalid_boxes_removed'] += 1
            continue
        
        cleaned_annotations.append(ann)
        # Track category usage
        cat_name = next((c['name'] for c in cleaned_categories if c['id'] == ann['category_id']), 'unknown')
        stats['categories'][cat_name] += 1
    
    # Build cleaned data
    cleaned_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': cleaned_categories,
        'images': coco_data.get('images', []),
        'annotations': cleaned_annotations,
    }
    
    stats['total_annotations_after'] = len(cleaned_annotations)
    stats['total_categories'] = len(cleaned_categories)
    
    return cleaned_data, stats

def process_dataset(raw_dir: Path, output_dir: Path, dataset_name: str):
    """Process a single dataset directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"  Skipping {split} (directory not found)")
            continue
        
        json_file = split_dir / '_annotations.coco.json'
        if not json_file.exists():
            print(f"  Skipping {split} (no annotations file found)")
            continue
        
        print(f"\n  Processing {split} split...")
        
        # Load and clean
        coco_data = load_coco_json(json_file)
        cleaned_data, stats = clean_annotations(coco_data)
        
        # Print statistics
        print(f"    Images: {stats['total_images']}")
        print(f"    Annotations: {stats['total_annotations_before']} → {stats['total_annotations_after']}")
        print(f"    Removed null: {stats['null_annotations_removed']}")
        print(f"    Removed invalid boxes: {stats['invalid_boxes_removed']}")
        print(f"    Categories: {stats['total_categories']}")
        print(f"    Category distribution:")
        for cat, count in stats['categories'].most_common():
            print(f"      - {cat}: {count}")
        
        # Save cleaned data
        output_split_dir = output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        output_json = output_split_dir / '_annotations.coco.json'
        save_coco_json(cleaned_data, output_json)
        
        # Copy images (symlink to save space, or copy if needed)
        import shutil
        for img_file in split_dir.glob('*.jpg'):
            if not (output_split_dir / img_file.name).exists():
                shutil.copy2(img_file, output_split_dir / img_file.name)
        for img_file in split_dir.glob('*.png'):
            if not (output_split_dir / img_file.name).exists():
                shutil.copy2(img_file, output_split_dir / img_file.name)
        
        print(f"    ✓ Saved cleaned data to {output_json}")

def main():
    """Main cleaning script."""
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'datasets' / 'raw'
    output_dir = project_root / 'datasets' / 'processed'
    
    if not raw_dir.exists():
        print(f"ERROR: Raw datasets directory not found: {raw_dir}")
        print("Please download datasets first (see scripts/01_download_datasets.md)")
        return
    
    # Process each dataset
    datasets = ['insulators', 'crossarm', 'utility-pole']
    
    for dataset_name in datasets:
        dataset_raw_dir = raw_dir / dataset_name
        if not dataset_raw_dir.exists():
            print(f"\n⚠️  Dataset '{dataset_name}' not found in {dataset_raw_dir}")
            print("   Skipping...")
            continue
        
        dataset_output_dir = output_dir / dataset_name
        process_dataset(dataset_raw_dir, dataset_output_dir, dataset_name)
    
    print(f"\n{'='*60}")
    print("✅ Dataset cleaning complete!")
    print(f"Cleaned datasets saved to: {output_dir}")
    print(f"\nNext step: Run merge script: python scripts/03_merge_datasets.py")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

