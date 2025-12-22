#!/usr/bin/env python3
"""
Extract downloaded Roboflow COCO dataset ZIP files to the correct directory structure.
"""

import zipfile
import shutil
from pathlib import Path
import sys

def extract_roboflow_zip(zip_path: Path, output_dir: Path, dataset_name: str):
    """
    Extract Roboflow COCO dataset ZIP to the expected structure.
    
    Roboflow COCO exports typically have structure:
    - train/
    - valid/
    - test/ (optional)
    Each with _annotations.coco.json and images
    """
    print(f"\n{'='*60}")
    print(f"Extracting: {dataset_name}")
    print(f"From: {zip_path.name}")
    print(f"To: {output_dir}")
    print(f"{'='*60}")
    
    if not zip_path.exists():
        print(f"ERROR: ZIP file not found: {zip_path}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract ZIP
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            
            # Find the base directory (Roboflow often includes a folder in the ZIP)
            base_dirs = set()
            for name in file_list:
                if '/' in name:
                    base_dirs.add(name.split('/')[0])
            
            # If there's a single base directory, we'll extract and move contents
            if len(base_dirs) == 1:
                base_dir = list(base_dirs)[0]
                print(f"  Found base directory: {base_dir}")
            else:
                base_dir = None
            
            # Extract all files
            zip_ref.extractall(output_dir.parent)
            
            # Move files to correct location
            if base_dir:
                extracted_base = output_dir.parent / base_dir
                if extracted_base.exists() and extracted_base != output_dir:
                    # Move contents from base_dir to output_dir
                    for item in extracted_base.iterdir():
                        dest = output_dir / item.name
                        if item.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(item), str(dest))
                        else:
                            if dest.exists():
                                dest.unlink()
                            shutil.move(str(item), str(dest))
                    # Remove empty base directory
                    if extracted_base.exists():
                        extracted_base.rmdir()
            
            # Also check if files were extracted directly to output_dir
            temp_extract = output_dir.parent
            for item in temp_extract.iterdir():
                if item.is_dir() and item.name in ['train', 'valid', 'test']:
                    dest = output_dir / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
            
        print(f"  ✓ Extraction complete")
        
        # Verify structure
        splits = ['train', 'valid', 'test']
        found_splits = []
        for split in splits:
            split_dir = output_dir / split
            if split_dir.exists():
                json_file = split_dir / '_annotations.coco.json'
                if json_file.exists():
                    found_splits.append(split)
                    # Count images
                    img_count = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
                    print(f"  ✓ {split}: {img_count} images, annotations found")
                else:
                    print(f"  ⚠️  {split}: directory exists but no _annotations.coco.json")
            else:
                print(f"  ⚠️  {split}: directory not found")
        
        if found_splits:
            print(f"  ✅ Dataset structure looks good!")
            return True
        else:
            print(f"  ⚠️  Warning: No valid splits found. Check the ZIP structure.")
            return False
            
    except Exception as e:
        print(f"  ERROR extracting {zip_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main extraction script."""
    project_root = Path(__file__).parent.parent
    workspace_root = project_root.parent
    
    # Find ZIP files in workspace root
    zip_files = {
        'insulators': workspace_root / 'Insulators.v5i.coco.zip',
        'crossarm': workspace_root / 'song crossarm.v6i.coco.zip',
        'utility-pole': workspace_root / 'utility-pole.v4i.coco.zip',
    }
    
    # Also check in project root
    for name, zip_path in list(zip_files.items()):
        if not zip_path.exists():
            alt_path = project_root / zip_path.name
            if alt_path.exists():
                zip_files[name] = alt_path
    
    print("="*60)
    print("Extracting Roboflow COCO Dataset ZIP Files")
    print("="*60)
    
    raw_dir = project_root / 'datasets' / 'raw'
    success_count = 0
    
    for dataset_name, zip_path in zip_files.items():
        if not zip_path.exists():
            print(f"\n⚠️  ZIP file not found: {zip_path.name}")
            print(f"   Please ensure the ZIP file is in the workspace root or project root")
            continue
        
        output_dir = raw_dir / dataset_name
        if extract_roboflow_zip(zip_path, output_dir, dataset_name):
            success_count += 1
    
    print(f"\n{'='*60}")
    if success_count == len(zip_files):
        print("✅ All datasets extracted successfully!")
        print(f"\nExtracted datasets are in: {raw_dir}")
        print(f"\nNext step: Run cleaning script:")
        print(f"  python scripts/02_clean_datasets.py")
    else:
        print(f"⚠️  Extracted {success_count}/{len(zip_files)} datasets")
        print("   Please check the extraction results above")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

