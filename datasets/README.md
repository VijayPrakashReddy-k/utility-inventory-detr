# Utility Inventory Detection Datasets

This directory contains the datasets used for training the utility inventory detection model.

## Dataset Structure

### Source Datasets (from Roboflow)

1. **Insulators**: [Roboflow Dataset](https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3)
   - Total used: 602 images, 599 annotations

2. **Crossarm**: [Roboflow Dataset](https://universe.roboflow.com/project-91iyv/song-crossarm-zqkmo/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
   - Total used: 182 images, 207 annotations

3. **Utility-pole**: [Roboflow Dataset](https://universe.roboflow.com/project-6kpfk/utility-pole-hdbuh/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
   - Total used: 139 images, 218 annotations

### Merged Dataset

The final training dataset is in `processed/merged/`:

- **Train**: 713 images, 809 annotations
- **Valid**: 134 images, 137 annotations
- **Test**: 76 images, 78 annotations
- **Total**: 923 images, 1,024 annotations

## Dataset Format

All datasets are in **COCO JSON format**:
- `_annotations.coco.json`: Contains image metadata, annotations, and category definitions
- Images: JPG format (excluded from git due to size)

## Processing Scripts

See `../scripts/` for data processing scripts (run in order):

1. **`01_download_datasets.py`**: Download Roboflow ZIP files (or download manually)
   - Checks for existing ZIP files
   - Provides download links if missing
   - Can use Roboflow API if you have an API key

2. **`00_extract_datasets.py`**: Extract downloaded ZIP files
   - Extracts ZIP files to `datasets/raw/` directory
   - Verifies dataset structure

3. **`02_clean_datasets.py`**: Clean and validate individual datasets
   - Removes null annotations
   - Validates bounding boxes
   - Saves cleaned datasets to `datasets/processed/`

4. **`03_merge_datasets.py`**: Merge multiple datasets into unified format
   - Unifies category IDs
   - Ensures unique image/annotation IDs
   - Creates final merged dataset in `datasets/processed/merged/`

## Note

- **Images are NOT included** in this repository (too large, ~923 images)
- Only **metadata JSON files** are included for reference
- To get the full dataset, download from Roboflow using the links above

