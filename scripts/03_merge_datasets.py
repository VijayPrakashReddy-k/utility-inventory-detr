#!/usr/bin/env python3
"""
Merge multiple COCO datasets into one unified dataset.

This script:
1. Loads cleaned datasets from processed/
2. Unifies category IDs across all datasets
3. Ensures unique image and annotation IDs
4. Creates train/valid/test splits
5. Saves merged dataset ready for training
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

def load_coco_json(json_path: Path) -> Dict:
    """Load COCO format JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_coco_json(data: Dict, json_path: Path):
    """Save COCO format JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def build_category_map(datasets: List[Dict]) -> Dict[str, int]:
    """
    Build unified category mapping across all datasets.
    
    Returns:
        category_name_to_id: Mapping from category name to unified ID
    """
    all_categories = {}
    category_id = 1  # Start from 1 (0 is reserved for background in some formats)
    
    for dataset in datasets:
        for cat in dataset.get('categories', []):
            cat_name = cat['name'].lower().strip()
            if cat_name not in all_categories:
                all_categories[cat_name] = category_id
                category_id += 1
    
    return all_categories

def merge_datasets(
    dataset_dirs: List[Path],
    output_dir: Path,
    category_map: Dict[str, int]
) -> Dict:
    """
    Merge multiple COCO datasets into one.
    
    Args:
        dataset_dirs: List of paths to cleaned dataset directories
        output_dir: Output directory for merged dataset
        category_map: Unified category name -> ID mapping
    
    Returns:
        Statistics about the merge
    """
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'categories': len(category_map),
        'category_distribution': defaultdict(int),
    }
    
    # Build reverse mapping (id -> name) for stats
    id_to_name = {v: k for k, v in category_map.items()}
    
    # Unified dataset structure
    merged = {
        'info': {
            'description': 'Merged Utility Inventory Dataset',
            'version': '1.0',
        },
        'licenses': [],
        'categories': [
            {'id': cat_id, 'name': cat_name, 'supercategory': 'utility'}
            for cat_name, cat_id in sorted(category_map.items(), key=lambda x: x[1])
        ],
        'images': [],
        'annotations': [],
    }
    
    # Track IDs to ensure uniqueness
    image_id_counter = 1
    annotation_id_counter = 1
    seen_image_hashes = set()
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        split_images = []
        split_annotations = []
        
        for dataset_dir in dataset_dirs:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            json_file = split_dir / '_annotations.coco.json'
            if not json_file.exists():
                continue
            
            print(f"  Loading from {dataset_dir.name}...")
            coco_data = load_coco_json(json_file)
            
            # Build old category ID -> new category ID mapping
            old_to_new_cat = {}
            for cat in coco_data.get('categories', []):
                old_id = cat['id']
                cat_name = cat['name'].lower().strip()
                if cat_name in category_map:
                    old_to_new_cat[old_id] = category_map[cat_name]
            
            # Process images
            image_id_map = {}  # old_id -> new_id
            for img in coco_data.get('images', []):
                # Create hash from filename to detect duplicates
                img_hash = hash(img.get('file_name', ''))
                if img_hash in seen_image_hashes:
                    continue  # Skip duplicate images
                
                seen_image_hashes.add(img_hash)
                new_img_id = image_id_counter
                image_id_map[img['id']] = new_img_id
                
                new_img = img.copy()
                new_img['id'] = new_img_id
                split_images.append(new_img)
                image_id_counter += 1
            
            # Process annotations
            for ann in coco_data.get('annotations', []):
                old_img_id = ann['image_id']
                if old_img_id not in image_id_map:
                    continue  # Image was skipped (duplicate)
                
                old_cat_id = ann['category_id']
                if old_cat_id not in old_to_new_cat:
                    continue  # Category not in unified map
                
                new_ann = ann.copy()
                new_ann['id'] = annotation_id_counter
                new_ann['image_id'] = image_id_map[old_img_id]
                new_ann['category_id'] = old_to_new_cat[old_cat_id]
                
                split_annotations.append(new_ann)
                annotation_id_counter += 1
                
                # Track category distribution
                cat_name = id_to_name[old_to_new_cat[old_cat_id]]
                stats['category_distribution'][cat_name] += 1
        
        # Save split
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        split_data = {
            'info': merged['info'],
            'licenses': [],
            'categories': merged['categories'],
            'images': split_images,
            'annotations': split_annotations,
        }
        
        split_json = split_output_dir / '_annotations.coco.json'
        save_coco_json(split_data, split_json)
        
        print(f"  ✓ Saved {len(split_images)} images, {len(split_annotations)} annotations")
        
        # Copy images (you may want to use symlinks to save space)
        import shutil
        for dataset_dir in dataset_dirs:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            for img_file in split_dir.glob('*.jpg'):
                dest = split_output_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
            for img_file in split_dir.glob('*.png'):
                dest = split_output_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
        
        stats['total_images'] += len(split_images)
        stats['total_annotations'] += len(split_annotations)
    
    return stats

def main():
    """Main merge script."""
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'datasets' / 'processed'
    output_dir = project_root / 'datasets' / 'processed' / 'merged'
    
    if not processed_dir.exists():
        print(f"ERROR: Processed datasets directory not found: {processed_dir}")
        print("Please run cleaning script first: python scripts/02_clean_datasets.py")
        return
    
    # Find all cleaned datasets
    dataset_dirs = []
    dataset_names = ['insulators', 'crossarm', 'utility-pole']
    
    for name in dataset_names:
        dataset_dir = processed_dir / name
        if dataset_dir.exists():
            dataset_dirs.append(dataset_dir)
            print(f"✓ Found dataset: {name}")
        else:
            print(f"⚠️  Dataset not found: {name}")
    
    if not dataset_dirs:
        print("ERROR: No cleaned datasets found!")
        return
    
    # Load all datasets to build category map
    print("\nBuilding unified category map...")
    all_datasets = []
    for dataset_dir in dataset_dirs:
        for split in ['train', 'valid', 'test']:
            json_file = dataset_dir / split / '_annotations.coco.json'
            if json_file.exists():
                all_datasets.append(load_coco_json(json_file))
    
    category_map = build_category_map(all_datasets)
    
    print(f"\nUnified categories ({len(category_map)}):")
    for cat_name, cat_id in sorted(category_map.items(), key=lambda x: x[1]):
        print(f"  {cat_id}: {cat_name}")
    
    # Confirm with user (in production, you might want to make this interactive)
    print(f"\n{'='*60}")
    print("Merging datasets...")
    print(f"{'='*60}")
    
    # Merge
    stats = merge_datasets(dataset_dirs, output_dir, category_map)
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("✅ Dataset merging complete!")
    print(f"{'='*60}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Categories: {stats['categories']}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count}")
    
    print(f"\nMerged dataset saved to: {output_dir}")
    print(f"\nNext step: Set up training code (see training/ directory)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

