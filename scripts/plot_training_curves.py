#!/usr/bin/env python3
"""
Plot training curves from training log JSON file.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_training_curves(log_file: str, output_file: str = None):
    """Plot training and validation loss curves."""
    
    # Load training log
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    epochs = [e['epoch'] for e in data['epochs']]
    train_loss = [e['train_loss'] for e in data['epochs']]
    val_loss = [e['val_loss'] for e in data['epochs']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DETR Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Classification Loss
    ax2 = axes[0, 1]
    train_ce = [e['train_loss_ce'] for e in data['epochs']]
    val_ce = [e['val_loss_ce'] for e in data['epochs']]
    ax2.plot(epochs, train_ce, 'b-', label='Train CE', linewidth=2)
    ax2.plot(epochs, val_ce, 'r-', label='Val CE', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Classification Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bounding Box Loss
    ax3 = axes[1, 0]
    train_bbox = [e['train_loss_bbox'] for e in data['epochs']]
    val_bbox = [e['val_loss_bbox'] for e in data['epochs']]
    ax3.plot(epochs, train_bbox, 'b-', label='Train Bbox', linewidth=2)
    ax3.plot(epochs, val_bbox, 'r-', label='Val Bbox', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Bounding Box Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. GIoU Loss
    ax4 = axes[1, 1]
    train_giou = [e['train_loss_giou'] for e in data['epochs']]
    val_giou = [e['val_loss_giou'] for e in data['epochs']]
    ax4.plot(epochs, train_giou, 'b-', label='Train GIoU', linewidth=2)
    ax4.plot(epochs, val_giou, 'r-', label='Val GIoU', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('GIoU Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_training_curves.py <training_log.json> [output.png]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    plot_training_curves(log_file, output_file)

