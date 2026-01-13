"""
Utilities to convert COCO-format datasets to a YOLOv11-OBB compatible layout.

This module creates per-image label files with lines in the format:
  <class_idx> x1 y1 x2 y2 x3 y3 x4 y4

Notes:
- For axis-aligned COCO bboxes, the 4 corners are calculated from the bbox.
- Coordinates are normalized to [0,1] relative to image width/height.
- COCO category IDs are expected to be 1-based; output class indices are 0-based.
"""
from pathlib import Path
import json
import shutil
from typing import Dict, List


def _ann_to_yolov11_line(ann: Dict, img_info: Dict, category_id_offset: int = 1) -> str:
    """Convert a single COCO annotation to a YOLOv11-OBB label line.

    Args:
        ann: COCO annotation dict (expects 'bbox' key)
        img_info: COCO image dict (expects 'width' and 'height' and 'file_name')
        category_id_offset: value to subtract from COCO category_id to make it 0-based

    Returns:
        A string line for the label file, e.g. "0 x1 y1 x2 y2 x3 y3 x4 y4"
    """
    bbox = ann.get('bbox', None)
    if bbox is None:
        return ''
    x, y, w, h = bbox
    img_w = img_info.get('width', None)
    img_h = img_info.get('height', None)
    if not img_w or not img_h:
        raise ValueError(f"Image width/height missing for image: {img_info}")

    # Calculate 4 corners of the bounding box (top-left, top-right, bottom-right, bottom-left)
    x1, y1 = x, y                    # Top-left
    x2, y2 = x + w, y                # Top-right
    x3, y3 = x + w, y + h            # Bottom-right
    x4, y4 = x, y + h                # Bottom-left

    # Normalize coordinates to [0, 1]
    x1_n = x1 / img_w
    y1_n = y1 / img_h
    x2_n = x2 / img_w
    y2_n = y2 / img_h
    x3_n = x3 / img_w
    y3_n = y3 / img_h
    x4_n = x4 / img_w
    y4_n = y4 / img_h

    # Convert category id to 0-based index
    class_idx = ann.get('category_id', 0) - category_id_offset

    return f"{class_idx} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x3_n:.6f} {y3_n:.6f} {x4_n:.6f} {y4_n:.6f}"


def convert_coco_json_to_yolov11_folder(coco_json_path: str, images_dir: str, out_dir: str, category_id_offset: int = 1):
    """Convert COCO JSON annotations into YOLOv11-OBB label files.

    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images referenced by COCO
        out_dir: Output directory. Will contain `images/` and `labels/` subfolders.
        category_id_offset: Integer to subtract from COCO category ids to make 0-based classes.
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco.get('images', [])}
    annotations_by_image = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann)

    img_out_dir = out_dir / 'images'
    lbl_out_dir = out_dir / 'labels'
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_info in images.items():
        file_name = img_info.get('file_name')
        if not file_name:
            continue

        src_img = Path(images_dir) / file_name
        if not src_img.exists():
            # try relative path in same folder as coco json
            alt = Path(coco_json_path).parent / file_name
            if alt.exists():
                src_img = alt
            else:
                print(f"Warning: image file not found for {file_name}, skipping")
                continue

        dest_img = img_out_dir / Path(file_name).name
        if not dest_img.exists():
            shutil.copy2(src_img, dest_img)

        anns = annotations_by_image.get(img_id, [])
        label_lines: List[str] = []
        for ann in anns:
            line = _ann_to_yolov11_line(ann, img_info, category_id_offset=category_id_offset)
            if line:
                label_lines.append(line)

        label_file = lbl_out_dir / (Path(file_name).stem + '.txt')
        with open(label_file, 'w') as lf:
            lf.write('\n'.join(label_lines))

    print(f"Converted {len(images)} images -> {out_dir} (images + labels)")


def build_names_file_from_coco(coco_json_path: str, out_path: str, category_id_offset: int = 1):
    """Create a `names` file listing class names in index order (0-based).

    Args:
        coco_json_path: Path to COCO JSON
        out_path: Path to write names file (one name per line)
        category_id_offset: subtract this from COCO id to get 0-based index
    """
    coco_json_path = Path(coco_json_path)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    cats = coco.get('categories', [])
    # Build mapping id->name
    max_id = max((c.get('id', 0) for c in cats), default=0)
    names = [''] * (max_id - category_id_offset + 1) if max_id >= category_id_offset else []
    for c in cats:
        idx = c.get('id', 0) - category_id_offset
        if idx >= 0:
            names[idx] = c.get('name', '')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(names))

    print(f"Wrote names file to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert COCO JSON to YOLOv11-OBB dataset layout')
    parser.add_argument('--coco', required=True, help='Path to COCO JSON file')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--out', required=True, help='Output directory for YOLOv11 dataset')
    args = parser.parse_args()
    convert_coco_json_to_yolov11_folder(args.coco, args.images, args.out)
