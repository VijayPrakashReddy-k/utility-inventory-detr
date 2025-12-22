"""
Log training metrics to JSON file for later visualization.
Add this to your training script to track metrics.
"""

import json
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    """Log training metrics to JSON file."""
    
    def __init__(self, log_file: str = "training/training_log.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'system_info': self._get_system_info(),
            'training_start': datetime.now().isoformat(),
            'epochs': []
        }
    
    def _get_system_info(self):
        """Get system configuration."""
        import platform
        import torch
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        # Try to get CPU info
        try:
            import subprocess
            if platform.system() == 'Darwin':  # macOS
                cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                info['cpu'] = cpu
        except:
            pass
        
        # PyTorch info
        try:
            info['pytorch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if hasattr(torch.backends, 'mps'):
                info['mps_available'] = torch.backends.mps.is_available()
            else:
                info['mps_available'] = False
        except:
            pass
        
        return info
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  train_loss_dict: dict, val_loss_dict: dict, 
                  learning_rate: float, epoch_time: float):
        """Log metrics for one epoch."""
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_loss_ce': train_loss_dict.get('loss_ce', 0),
            'train_loss_bbox': train_loss_dict.get('loss_bbox', 0),
            'train_loss_giou': train_loss_dict.get('loss_giou', 0),
            'val_loss_ce': val_loss_dict.get('loss_ce', 0),
            'val_loss_bbox': val_loss_dict.get('loss_bbox', 0),
            'val_loss_giou': val_loss_dict.get('loss_giou', 0),
            'learning_rate': learning_rate,
            'epoch_time_seconds': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['epochs'].append(epoch_data)
        self._save()
    
    def finish_training(self, total_time_seconds: float):
        """Mark training as complete."""
        self.metrics['training_end'] = datetime.now().isoformat()
        self.metrics['total_training_time_seconds'] = total_time_seconds
        self.metrics['total_training_time_hours'] = total_time_seconds / 3600
        self._save()
    
    def _save(self):
        """Save metrics to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

