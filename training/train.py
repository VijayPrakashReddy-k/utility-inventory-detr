#!/usr/bin/env python3
"""
DETR Training Script for Utility Inventory Dataset
"""

import argparse
import yaml
import json
from pathlib import Path
import sys
import os

# Workaround for OpenMP duplicate library issue (common with conda)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.detr_model import DETRdemo
from training.dataset import CocoDetection, make_coco_transforms, CocoCollate
from training.log_training_metrics import TrainingLogger
import time


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
    x0, y0, x1, y1 = x.unbind(1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    boxes1, boxes2: [N, 4] in xyxy format
    """
    # intersection
    inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter = inter.clamp(min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    
    # union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / union
    
    # GIoU
    c_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    c_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    c_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    c_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    giou = iou - (c_area - union) / c_area
    
    return giou


class SetCriterion(nn.Module):
    """DETR loss computation."""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Bounding box loss (L1 + GIoU)."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Convert to cxcywh format
        src_boxes_cxcy = box_xyxy_to_cxcywh(src_boxes)
        target_boxes_cxcy = box_xyxy_to_cxcywh(target_boxes)

        loss_bbox = nn.functional.l1_loss(src_boxes_cxcy, target_boxes_cxcy, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes_cxcy),
            box_cxcywh_to_xyxy(target_boxes_cxcy)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """Returns the sum of all losses."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we do this computation for each layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                losses.update(self.loss_labels(aux_outputs, targets, indices, num_boxes))
                losses.update(self.loss_boxes(aux_outputs, targets, indices, num_boxes))

        return losses


class HungarianMatcher(nn.Module):
    """Hungarian matcher for bipartite matching."""
    
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Perform the matching."""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch*num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch*num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def linear_sum_assignment(cost_matrix):
    """Simple Hungarian algorithm implementation."""
    try:
        from scipy.optimize import linear_sum_assignment
        i, j = linear_sum_assignment(cost_matrix)
        return i, j
    except ImportError:
        # Fallback: simple greedy matching (not optimal but works)
        import numpy as np
        cost_np = cost_matrix.numpy()
        i, j = [], []
        used_j = set()
        for row_idx in range(len(cost_np)):
            best_j = None
            best_cost = float('inf')
            for col_idx in range(len(cost_np[row_idx])):
                if col_idx not in used_j and cost_np[row_idx, col_idx] < best_cost:
                    best_cost = cost_np[row_idx, col_idx]
                    best_j = col_idx
            if best_j is not None:
                i.append(row_idx)
                j.append(best_j)
                used_j.add(best_j)
        return np.array(i), np.array(j)


def load_pretrained_weights(model, pretrained_path=None):
    """Load pretrained DETR weights (COCO)."""
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out classification head if num_classes differs
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if 'linear_class' in k:
                    # Skip classification head - will be randomly initialized
                    continue
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("✓ Loaded pretrained weights (classification head randomly initialized)")
    else:
        # Try to download from torch hub
        try:
            print("Attempting to download pretrained weights from torch hub...")
            state_dict = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
                map_location='cpu', check_hash=True
            )
            model_dict = model.state_dict()
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and 'linear_class' not in k:
                    if model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
            print("✓ Loaded pretrained weights from torch hub")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
            print("   Training from scratch...")


def train_epoch(model, criterion, data_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_dicts = []
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        # images is already a tensor from CocoCollate
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        total_loss += losses.item()
        loss_dicts.append({k: v.item() for k, v in loss_dict.items()})
        
        # Update progress bar
        pbar.set_postfix({'loss': losses.item()})
    
    avg_loss = total_loss / len(data_loader)
    if loss_dicts:
        avg_loss_dict = {k: np.mean([d[k] for d in loss_dicts if k in d]) for k in loss_dicts[0].keys()}
    else:
        avg_loss_dict = {}
    
    return avg_loss, avg_loss_dict


def validate(model, criterion, data_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    loss_dicts = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            # images is already a tensor from CocoCollate
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
            
            total_loss += losses.item()
            loss_dicts.append({k: v.item() for k, v in loss_dict.items()})
    
    avg_loss = total_loss / len(data_loader)
    if loss_dicts:
        avg_loss_dict = {k: np.mean([d[k] for d in loss_dicts if k in d]) for k in loss_dicts[0].keys()}
    else:
        avg_loss_dict = {}
    
    return avg_loss, avg_loss_dict


def main():
    parser = argparse.ArgumentParser(description='Train DETR on Utility Inventory Dataset')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / config['data']['dataset_path']
    checkpoint_dir = project_root / config['output']['checkpoint_dir']
    final_model_dir = project_root / config['output']['final_model_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Device - Auto-detect best available
    device_str = config['device'].lower()
    if device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon GPU)")
    elif device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    
    # Create datasets
    # Get image size (convert to int if needed)
    image_size = int(config['data']['image_size'])
    
    train_dataset = CocoDetection(
        img_folder=str(dataset_path / config['data']['train_split']),
        ann_file=str(dataset_path / config['data']['train_split'] / '_annotations.coco.json'),
        transforms=make_coco_transforms('train', image_size)
    )
    
    val_dataset = CocoDetection(
        img_folder=str(dataset_path / config['data']['val_split']),
        ann_file=str(dataset_path / config['data']['val_split'] / '_annotations.coco.json'),
        transforms=make_coco_transforms('val', image_size)
    )
    
    # Convert data config values to proper types
    batch_size = int(config['data']['batch_size'])
    num_workers = int(config['data']['num_workers'])
    image_size = int(config['data']['image_size'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CocoCollate()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CocoCollate()
    )
    
    # Create model (convert config values to proper types)
    model = DETRdemo(
        num_classes=int(config['model']['num_classes']),
        hidden_dim=int(config['model']['hidden_dim']),
        nheads=int(config['model']['nheads']),
        num_encoder_layers=int(config['model']['num_encoder_layers']),
        num_decoder_layers=int(config['model']['num_decoder_layers'])
    ).to(device)
    
    # Load pretrained weights if available
    if config['model']['pretrained']:
        load_pretrained_weights(model)
    
    # Create matcher and criterion
    # Convert loss weights to float
    class_weight = float(config['training']['loss']['class_weight'])
    bbox_weight = float(config['training']['loss']['bbox_weight'])
    giou_weight = float(config['training']['loss']['giou_weight'])
    
    matcher = HungarianMatcher(
        cost_class=class_weight,
        cost_bbox=bbox_weight,
        cost_giou=giou_weight
    )
    
    weight_dict = {
        'loss_ce': class_weight,
        'loss_bbox': bbox_weight,
        'loss_giou': giou_weight
    }
    
    criterion = SetCriterion(
        num_classes=config['model']['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict
    ).to(device)
    
    # Optimizer and scheduler
    # Convert YAML values to float (handles scientific notation strings like "1e-4")
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    lr_drop = int(config['training']['lr_drop'])
    lr_gamma = float(config['training']['lr_gamma'])
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=lr_drop,
        gamma=lr_gamma
    )
    
    # Initialize training logger
    log_dir = project_root / config['output']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(str(log_dir / "training_log.json"))
    
    # Add dataset info to logger
    logger.metrics['dataset_info'] = {
        'total_images': len(train_dataset) + len(val_dataset),
        'train_images': len(train_dataset),
        'val_images': len(val_dataset),
        'num_classes': config['model']['num_classes'],
        'classes': ['insulators', 'crossarm', 'utility-pole']
    }
    
    # Training loop
    best_val_loss = float('inf')
    num_epochs = int(config['training']['num_epochs'])
    save_every = int(config['output']['save_every'])
    training_start_time = time.time()
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        # Train
        train_loss, train_loss_dict = train_epoch(model, criterion, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_loss_dict = validate(model, criterion, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        for k, v in train_loss_dict.items():
            print(f"    {k}: {v:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        for k, v in val_loss_dict.items():
            print(f"    {k}: {v:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_loss_dict=train_loss_dict,
            val_loss_dict=val_loss_dict,
            learning_rate=optimizer.param_groups[0]['lr'],
            epoch_time=epoch_time
        )
        
        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = final_model_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  ✓ Saved best model: {best_model_path}")
    
    # Save final model
    final_model_path = final_model_dir / "final_model.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'val_loss': val_loss,
    }, final_model_path)
    
    # Finish logging
    total_training_time = time.time() - training_start_time
    logger.finish_training(total_training_time)
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"✓ Final model saved: {final_model_path}")
    print(f"✓ Training log saved: {logger.log_file}")
    print(f"✓ Total training time: {total_training_time/3600:.2f} hours")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

