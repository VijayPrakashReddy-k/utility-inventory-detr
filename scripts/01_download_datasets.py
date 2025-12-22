#!/usr/bin/env python3
"""
Download Roboflow COCO dataset ZIP files.

This script downloads the three utility inventory datasets from Roboflow.
You need a Roboflow API key to download datasets.

Usage:
    python scripts/01_download_datasets.py

Environment Variables:
    ROBOFLOW_API_KEY: Your Roboflow API key (required)
    
Or set it in the script below.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Optional

# Roboflow dataset download URLs
# Note: These are public datasets, but you may need to download manually from Roboflow
# or use Roboflow API if you have access

DATASET_INFO = {
    'insulators': {
        'name': 'Insulators',
        'url': 'https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3',
        'download_url': None,  # Public datasets require manual download or API
        'expected_filename': 'Insulators.v5i.coco.zip',
    },
    'crossarm': {
        'name': 'Crossarm',
        'url': 'https://universe.roboflow.com/project-91iyv/song-crossarm-zqkmo/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true',
        'download_url': None,
        'expected_filename': 'song crossarm.v6i.coco.zip',
    },
    'utility-pole': {
        'name': 'Utility-pole',
        'url': 'https://universe.roboflow.com/project-6kpfk/utility-pole-hdbuh/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true',
        'download_url': None,
        'expected_filename': 'utility-pole.v4i.coco.zip',
    }
}

def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file from URL to output path."""
    try:
        print(f"  Downloading to: {output_path}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False

def download_with_roboflow_api(workspace: str, project: str, version: int, api_key: str, output_path: Path) -> bool:
    """
    Download dataset using Roboflow API.
    
    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number
        api_key: Roboflow API key
        output_path: Where to save the ZIP file
    """
    # Roboflow API endpoint for downloading datasets
    api_url = f"https://api.roboflow.com/dataset/{workspace}/{project}/{version}/download"
    params = {
        'api_key': api_key,
        'format': 'coco'
    }
    
    try:
        print(f"  Downloading via Roboflow API...")
        response = requests.get(api_url, params=params, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"  ✓ Download complete")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    """Main download script."""
    project_root = Path(__file__).parent.parent
    workspace_root = project_root.parent
    
    print("="*60)
    print("Download Roboflow COCO Dataset ZIP Files")
    print("="*60)
    print()
    print("NOTE: These are public Roboflow datasets.")
    print("You have two options:")
    print()
    print("Option 1: Manual Download (Recommended)")
    print("  1. Visit each dataset URL below")
    print("  2. Click 'Download' and select 'COCO' format")
    print("  3. Save ZIP files to workspace root or project root")
    print()
    print("Option 2: Roboflow API (if you have API key)")
    print("  Set ROBOFLOW_API_KEY environment variable")
    print("  Or edit this script to add your API key")
    print()
    print("="*60)
    print()
    
    # Check for API key
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        print("⚠️  No ROBOFLOW_API_KEY found - using manual download instructions")
        print()
    
    # Check for existing ZIP files
    print("Checking for existing ZIP files...")
    found_files = []
    missing_files = []
    
    for dataset_id, info in DATASET_INFO.items():
        filename = info['expected_filename']
        
        # Check in workspace root
        zip_path = workspace_root / filename
        if not zip_path.exists():
            # Check in project root
            zip_path = project_root / filename
        
        if zip_path.exists():
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Found: {filename} ({size_mb:.1f} MB)")
            found_files.append(dataset_id)
        else:
            print(f"  ✗ Missing: {filename}")
            missing_files.append(dataset_id)
            print(f"     URL: {info['url']}")
    
    print()
    
    if not missing_files:
        print("✅ All ZIP files already exist!")
        print(f"\nNext step: Run extraction script:")
        print(f"  python scripts/00_extract_datasets.py")
        return
    
    print(f"⚠️  Missing {len(missing_files)} ZIP file(s)")
    print()
    print("To download manually:")
    print()
    for dataset_id in missing_files:
        info = DATASET_INFO[dataset_id]
        print(f"  {info['name']}:")
        print(f"    URL: {info['url']}")
        print(f"    Expected filename: {info['expected_filename']}")
        print(f"    Save to: {workspace_root} or {project_root}")
        print()
    
    print("After downloading, run:")
    print("  python scripts/00_extract_datasets.py")

if __name__ == '__main__':
    main()

