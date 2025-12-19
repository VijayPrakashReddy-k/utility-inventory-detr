"""
About page for Utility Inventory Detection Model.
Displays dataset statistics, training information, system configuration, and improvement recommendations.
"""

import json
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def load_training_log(log_path: str):
    """Load training log JSON file."""
    if not log_path:
        return None
    try:
        log_file = Path(log_path)
        if not log_file.exists():
            # Try same directory as this file
            current_file = Path(__file__)
            local_log = current_file.parent / "training_log.json"
            if local_log.exists():
                log_file = local_log
            else:
                return None
        with open(log_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading training log: {e}")
        return None


def get_dataset_stats():
    """Get dataset statistics from merged dataset."""
    base_path = Path(__file__).parent.parent / "datasets" / "processed" / "merged"
    
    stats = {
        'train': {'images': 0, 'annotations': 0, 'per_class': {}},
        'valid': {'images': 0, 'annotations': 0, 'per_class': {}},
        'test': {'images': 0, 'annotations': 0, 'per_class': {}}
    }
    
    for split in ['train', 'valid', 'test']:
        json_path = base_path / split / "_annotations.coco.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            stats[split]['images'] = len(data['images'])
            stats[split]['annotations'] = len(data['annotations'])
            
            # Count per class
            for ann in data['annotations']:
                cat_id = ann['category_id']
                cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
                stats[split]['per_class'][cat_name] = stats[split]['per_class'].get(cat_name, 0) + 1
    
    return stats


def show_about_page(log_path: str = None):
    """Display comprehensive about page with all sections."""
    
    st.title("About Utility Inventory Detection Model")
    
    # ============================================================================
    # CRITICAL ISSUE: Utility-Pole Detection Problem
    # ============================================================================
    st.header("⚠️ Critical Training Issue: Utility-Pole Detection")
    
    st.error("""
    **Problem:** Utility-pole detection is **disabled** due to a class index conflict.
    
    **Root Cause:**
    - DETR uses index `num_classes` (3 in our case) for "no object" queries
    - Utility-pole was also assigned to index 3
    - The model learned that index 3 = "no object", causing utility-pole to be over-predicted
    - Even with strict filtering (40% margin, 0.85 threshold), false positives persisted
    
    **Current Status:** Utility-pole detection is **temporarily disabled** to prevent false positives. Only insulators and crossarm are currently being detected.
    
    **Solution:** Retrain with utility-pole mapped to index 4 (use 4-class setup) to avoid the conflict.
    """)
    
    st.divider()
    
    # ============================================================================
    # DATASET SECTION
    # ============================================================================
    st.header("📊 Dataset Statistics")
    
    # Dataset sources
    st.subheader("Dataset Sources")
    st.write("The model was trained on merged datasets from Roboflow. **Note:** Only a subset of each dataset was downloaded and used:")
    
    # Get actual numbers used from individual datasets
    base_path = Path(__file__).parent.parent / "datasets" / "processed"
    dataset_info = {
        "Insulators": {
            "url": "https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3",
            "images_used": 0,
            "annotations_used": 0
        },
        "Crossarm": {
            "url": "https://universe.roboflow.com/project-91iyv/song-crossarm-zqkmo/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true",
            "images_used": 0,
            "annotations_used": 0
        },
        "Utility-pole": {
            "url": "https://universe.roboflow.com/project-6kpfk/utility-pole-hdbuh/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true",
            "images_used": 0,
            "annotations_used": 0
        }
    }
    
    # Get actual numbers from source datasets (ALL splits: train, valid, test)
    dataset_name_map = {
        "Insulators": "insulators",
        "Crossarm": "crossarm",
        "Utility-pole": "utility-pole"
    }
    
    for class_name in dataset_info.keys():
        dataset_name = dataset_name_map[class_name]
        # Sum across all splits (train, valid, test)
        images_total = 0
        annotations_total = 0
        for split in ['train', 'valid', 'test']:
            source_path = base_path / dataset_name / split / "_annotations.coco.json"
            if source_path.exists():
                with open(source_path, 'r') as f:
                    data = json.load(f)
                images_total += len(data['images'])
                annotations_total += len(data['annotations'])
        
        dataset_info[class_name]["images_used"] = images_total
        dataset_info[class_name]["annotations_used"] = annotations_total
    
    # Display with actual numbers
    for class_name, info in dataset_info.items():
        st.write(f"- **{class_name}:** [{info['url']}]({info['url']})")
        st.caption(f"  → Used: {info['images_used']} images, {info['annotations_used']} annotations (subset of full dataset)")
    
    # Show totals verification
    total_source_images = sum(info['images_used'] for info in dataset_info.values())
    total_source_annotations = sum(info['annotations_used'] for info in dataset_info.values())
    st.info(f"**Total from source datasets (all splits):** {total_source_images} images, {total_source_annotations} annotations")
    
    st.divider()
    
    dataset_stats = get_dataset_stats()
    
    # Overall statistics
    total_images = sum(stats['images'] for stats in dataset_stats.values())
    total_annotations = sum(stats['annotations'] for stats in dataset_stats.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Total Annotations", total_annotations)
    with col3:
        st.metric("Number of Classes", 3)
    
    # Split-wise statistics
    st.subheader("Split-wise Distribution")
    
    splits_data = []
    for split_name, split_stats in dataset_stats.items():
        splits_data.append({
            'Split': split_name.upper(),
            'Images': split_stats['images'],
            'Annotations': split_stats['annotations'],
            'Avg Annotations/Image': round(split_stats['annotations'] / split_stats['images'], 2) if split_stats['images'] > 0 else 0
        })
    
    splits_df = pd.DataFrame(splits_data)
    st.dataframe(splits_df, use_container_width=True, hide_index=True)
    
    # Class distribution
    st.subheader("Class Distribution")
    
    # Combine all splits for total class counts
    total_per_class = {}
    for split_stats in dataset_stats.values():
        for class_name, count in split_stats['per_class'].items():
            total_per_class[class_name] = total_per_class.get(class_name, 0) + count
    
    # Create visualization
    if total_per_class:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        classes = list(total_per_class.keys())
        counts = list(total_per_class.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        ax1.bar(classes, counts, color=colors[:len(classes)])
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Annotations')
        ax1.set_title('Total Annotations per Class')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors[:len(classes)], startangle=90)
        ax2.set_title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed per-split class distribution
        st.subheader("Per-Split Class Distribution")
        class_dist_data = []
        for split_name, split_stats in dataset_stats.items():
            for class_name in ['insulators', 'crossarm', 'utility-pole']:
                count = split_stats['per_class'].get(class_name, 0)
                class_dist_data.append({
                    'Split': split_name.upper(),
                    'Class': class_name,
                    'Count': count
                })
        
        class_dist_df = pd.DataFrame(class_dist_data)
        pivot_df = class_dist_df.pivot(index='Class', columns='Split', values='Count').fillna(0)
        st.dataframe(pivot_df, use_container_width=True)
        
        # Data imbalance analysis
        st.subheader("⚠️ Data Imbalance Analysis")
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        st.warning(f"""
        **Imbalance Ratio:** {imbalance_ratio:.2f}:1 (max:min)
        
        - **Most common class:** {classes[counts.index(max_count)]} ({max_count} annotations)
        - **Least common class:** {classes[counts.index(min_count)]} ({min_count} annotations)
        
        **Impact:**
        - The model may be biased towards the majority class (insulators)
        - Minority classes (crossarm, utility-pole) may have lower detection accuracy
        - Consider data augmentation or class weighting during training
        """)
    
    st.divider()
    
    # ============================================================================
    # TRAINING SECTION
    # ============================================================================
    st.header("🏋️ Training Information")
    
    # Try to find training log if not provided
    if not log_path:
        # Try common paths - check same directory first (preferred location)
        current_file = Path(__file__)
        possible_paths = [
            current_file.parent / "training_log.json",  # Same directory as about_page.py (preferred)
            current_file.parent.parent / "training" / "logs" / "training_log.json",
            current_file.parent.parent / "training" / "training_log.json",
            current_file.parent.parent.parent / "utility-detr" / "training" / "logs" / "training_log.json",
            Path("/Users/vijayprakashreddy/Documents/SIEAERO/model-build/utility-detr/training/logs/training_log.json"),
        ]
        for path in possible_paths:
            if path.exists():
                log_path = str(path)
                break
    
    if log_path and Path(log_path).exists():
        training_log = load_training_log(log_path)
        
        if training_log:
            # System info
            if 'system_info' in training_log:
                sys_info = training_log['system_info']
                st.subheader("Training System")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Platform:** {sys_info.get('platform', 'N/A')}")
                    st.write(f"**Processor:** {sys_info.get('processor', 'N/A')}")
                    st.write(f"**CPU:** {sys_info.get('cpu', 'N/A')}")
                with col2:
                    st.write(f"**Python:** {sys_info.get('python_version', 'N/A')}")
                    st.write(f"**PyTorch:** {sys_info.get('pytorch_version', 'N/A')}")
                    st.write(f"**MPS Available:** {sys_info.get('mps_available', False)}")
            
            # Training duration
            if 'total_training_time_hours' in training_log:
                hours = training_log['total_training_time_hours']
                st.subheader("Training Duration")
                st.metric("Total Training Time", f"{hours:.2f} hours ({hours * 60:.1f} minutes)")
            
            # Training dates
            if 'training_start' in training_log and 'training_end' in training_log:
                start = datetime.fromisoformat(training_log['training_start'])
                end = datetime.fromisoformat(training_log['training_end'])
                duration = end - start
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Start:** {start.strftime('%Y-%m-%d %H:%M:%S')}")
                with col2:
                    st.write(f"**End:** {end.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Training configuration
            st.subheader("Training Configuration")
            if 'dataset_info' in training_log:
                ds_info = training_log['dataset_info']
                config_col1, config_col2 = st.columns(2)
                with config_col1:
                    st.write(f"**Training Images:** {ds_info.get('train_images', 'N/A')}")
                    st.write(f"**Validation Images:** {ds_info.get('val_images', 'N/A')}")
                    st.write(f"**Number of Classes:** {ds_info.get('num_classes', 'N/A')}")
                with config_col2:
                    st.write(f"**Classes:** {', '.join(ds_info.get('classes', []))}")
            
            # Final metrics
            if 'epochs' in training_log and len(training_log['epochs']) > 0:
                final_epoch = training_log['epochs'][-1]
                st.subheader("Final Training Metrics (Epoch 50)")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Train Loss", f"{final_epoch.get('train_loss', 0):.4f}")
                    st.metric("Val Loss", f"{final_epoch.get('val_loss', 0):.4f}")
                with metrics_col2:
                    st.metric("Classification Loss", f"{final_epoch.get('train_loss_ce', 0):.4f}")
                    st.metric("BBox Loss", f"{final_epoch.get('train_loss_bbox', 0):.4f}")
                with metrics_col3:
                    st.metric("GIoU Loss", f"{final_epoch.get('train_loss_giou', 0):.4f}")
                    st.metric("Learning Rate", f"{final_epoch.get('learning_rate', 0):.2e}")
            
            # Loss curves
            if 'epochs' in training_log and len(training_log['epochs']) > 0:
                st.subheader("Training & Validation Loss Curves")
                
                epochs = [e['epoch'] for e in training_log['epochs']]
                train_losses = [e['train_loss'] for e in training_log['epochs']]
                val_losses = [e['val_loss'] for e in training_log['epochs']]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
                ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Progress')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Best epoch
                best_epoch_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
                best_epoch = epochs[best_epoch_idx]
                best_val_loss = val_losses[best_epoch_idx]
                st.info(f"**Best Model:** Epoch {best_epoch} (Validation Loss: {best_val_loss:.4f})")
        else:
            st.warning("Training log could not be loaded.")
    else:
        st.warning("Training log file not found.")
    
    st.divider()
    
    # ============================================================================
    # INFRASTRUCTURE SECTION
    # ============================================================================
    st.header("💻 Infrastructure & System Configuration")
    
    if log_path and Path(log_path).exists():
        training_log = load_training_log(log_path)
        if training_log and 'system_info' in training_log:
            sys_info = training_log['system_info']
            
            infra_col1, infra_col2 = st.columns(2)
            
            with infra_col1:
                st.subheader("Hardware")
                st.write(f"**CPU:** {sys_info.get('cpu', 'N/A')}")
                st.write(f"**Processor Architecture:** {sys_info.get('processor', 'N/A')}")
                st.write(f"**Platform:** {sys_info.get('platform', 'N/A')}")
            
            with infra_col2:
                st.subheader("Software")
                st.write(f"**Python Version:** {sys_info.get('python_version', 'N/A')}")
                st.write(f"**PyTorch Version:** {sys_info.get('pytorch_version', 'N/A')}")
                st.write(f"**CUDA Available:** {sys_info.get('cuda_available', False)}")
                st.write(f"**MPS Available:** {sys_info.get('mps_available', False)}")
            
            if sys_info.get('mps_available'):
                st.success("✅ Training was accelerated using Apple Silicon MPS (Metal Performance Shaders)")
            elif sys_info.get('cuda_available'):
                st.success("✅ Training was accelerated using CUDA")
            else:
                st.info("ℹ️ Training was performed on CPU")
    
    st.divider()
    
    # ============================================================================
    # IMPROVEMENTS NEEDED SECTION
    # ============================================================================
    st.header("🔧 Improvements Needed")
    
    st.subheader("1. Detection & Classification Improvements")
    
    improvement_tasks = [
        {
            "Priority": "🔴 Critical",
            "Issue": "Utility-Pole Detection",
            "Description": "Retrain model with different class mapping to fix index 3 conflict with 'no object' class",
            "Action": "Map utility-pole to index 4 instead of 3, or use 4-class setup where utility-pole gets its own index"
        },
        {
            "Priority": "🟡 High",
            "Issue": "Bounding Box Accuracy",
            "Description": "Bounding boxes are sometimes incorrectly positioned or sized",
            "Action": "Improve box regression loss, increase bbox_weight in training config, add more training data"
        },
        {
            "Priority": "🟡 High",
            "Issue": "Classification Accuracy",
            "Description": "Some objects are misclassified (e.g., insulators detected as crossarm)",
            "Action": "Increase classification loss weight, add more diverse training examples, use data augmentation"
        },
        {
            "Priority": "🟢 Medium",
            "Issue": "Confidence Calibration",
            "Description": "Confidence scores may not accurately reflect detection quality",
            "Action": "Implement temperature scaling, use focal loss, or calibrate confidence thresholds"
        }
    ]
    
    for task in improvement_tasks:
        with st.expander(f"{task['Priority']} - {task['Issue']}"):
            st.write(f"**Description:** {task['Description']}")
            st.write(f"**Recommended Action:** {task['Action']}")
    
    st.subheader("2. Data Imbalance Improvements")
    
    imbalance_tasks = [
        {
            "Issue": "Class Distribution Imbalance",
            "Current": "Insulators: 417, Crossarm: 180, Utility-pole: 212 (train split)",
            "Impact": "Model biased towards insulators, lower accuracy for crossarm",
            "Solutions": [
                "Collect more crossarm training data",
                "Use class weighting in loss function (increase weight for minority classes)",
                "Apply data augmentation more aggressively for minority classes",
                "Use focal loss to focus on hard examples",
                "Oversample minority classes during training"
            ]
        },
        {
            "Issue": "Validation Set Imbalance",
            "Current": "Validation set has no utility-pole annotations",
            "Impact": "Cannot properly evaluate utility-pole detection performance",
            "Solutions": [
                "Redistribute dataset splits to ensure all classes in validation set",
                "Use stratified splitting to maintain class distribution across splits",
                "Add utility-pole images to validation set"
            ]
        }
    ]
    
    for task in imbalance_tasks:
        with st.expander(f"📊 {task['Issue']}"):
            st.write(f"**Current State:** {task['Current']}")
            st.write(f"**Impact:** {task['Impact']}")
            st.write("**Recommended Solutions:**")
            for solution in task['Solutions']:
                st.write(f"  - {solution}")
    
    st.subheader("3. Model Architecture Improvements")
    
    architecture_improvements = [
        "Consider using RF-DETR (Robust Faster DETR) for better performance and faster inference",
        "Increase hidden_dim from 256 to 512 for richer feature representations",
        "Add more encoder/decoder layers (currently 6 each)",
        "Use pre-trained weights from COCO (already implemented)",
        "Experiment with different backbone architectures (ResNet101, EfficientNet)",
        "Implement learning rate warmup for better convergence"
    ]
    
    for improvement in architecture_improvements:
        st.write(f"  - {improvement}")
    
    st.divider()
    
    # Footer
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

