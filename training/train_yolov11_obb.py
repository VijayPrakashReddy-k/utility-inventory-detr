"""
Wrapper to run YOLOv11-OBB training on the converted dataset.

This script will:
- Create a `data.yaml` file pointing to the YOLOv11-OBB dataset created by
  `scripts/04_convert_coco_to_yolov11_obb.py` (default: `datasets/processed/yolov11_obb`).
- Attempt to invoke an external `yolov11` CLI if available. If not present,
  it will print the training command the user can run manually.

Note: This repo does not vendor the YOLOv11 code. Install the YOLOv11-OBB
implementation you intend to use and ensure its CLI is on PATH.
"""
import argparse
import yaml
from pathlib import Path
import subprocess
import sys

from ultralytics import YOLO

def build_data_yaml(yolov11_root: Path, names_file: Path, out_yaml: Path) -> Path:
    """Write a simple data.yaml for YOLO training tools.

    The file contains paths for `train`, `val` and `test` (optional) and
    number of classes and path to names.

    Args:
        yolov11_root: Root directory of YOLOv11 dataset (with train/valid/test subdirs)
        names_file: Path to names.txt file
        out_yaml: Path to write data.yaml file
    Returns:
        Path to the written data.yaml file
    """
    train_imgs = yolov11_root / 'train' / 'images'
    val_imgs = yolov11_root / 'valid' / 'images'
    test_imgs = yolov11_root / 'test' / 'images'

    # Count classes from names file
    with open(names_file, 'r') as f:
        names = [l.strip() for l in f.readlines() if l.strip()]

    data = {
        'train': str(train_imgs).replace('\\', '/'),
        'val': str(val_imgs).replace('\\', '/'),
        'test': str(test_imgs).replace('\\', '/'),
        'nc': len(names),
        'names': names
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(data, f)

    print(f"Wrote data YAML to {out_yaml}")
    return out_yaml


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11-OBB using converted dataset')
    parser.add_argument('--dataset', type=str, default='datasets/processed/yolov11_obb',
                        help='Root of converted YOLOv11 dataset (images/labels per split)')
    parser.add_argument('--names', type=str, default='datasets/processed/yolov11_obb/names.txt',
                        help='Path to names.txt for classes')
    parser.add_argument('--out', type=str, default='models/yolov11_obb', help='Where to save checkpoints')
    parser.add_argument('--weights', type=str, default='yolov11m-obb.pt', help='Pretrained OBB weights (Ultralytics format)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    yolov11_root = project_root / args.dataset
    names_file = project_root / args.names
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = out_dir / 'data.yaml'
    if not names_file.exists():
        print(f"ERROR: names file not found: {names_file}")
        sys.exit(1)

    build_data_yaml(yolov11_root, names_file, data_yaml)
    
    print('\nLaunching training via Ultralytics Python API')
    model = YOLO(args.weights)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        workers=args.workers,
        project=str(out_dir),
    )
    print(f"Training with weights={args.weights}, epochs={args.epochs}, imgsz={args.img}")
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
