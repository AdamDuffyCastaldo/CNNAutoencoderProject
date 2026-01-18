"""
I/O Utilities for SAR Autoencoder
"""
import numpy as np
from glob import glob
import torch
from pathlib import Path
from typing import Dict, Optional
import rasterio


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load to
    
    Returns:
        Checkpoint dict with epoch, config, etc.
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        path: Save path
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        config: Training configuration
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        **kwargs
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_sar_image(tiff_path):
    """Load Sentinel-1 GeoTIFF."""
    with rasterio.open(tiff_path) as src:
        image = src.read(1).astype(np.float32)
    return image


def find_all_sar_files(raw_dir):
    raw_dir = Path(raw_dir)
    safe_pattern_vv = str(raw_dir / '*.SAFE' / 'measurement' / '*-vv-*.tiff')
    safe_pattern_vh = str(raw_dir / "*.SAFE" / "measurement" / "*-vh-*.tiff")
    
    
    safe_pattern_vhtif = str(raw_dir / '*.SAFE' / 'measurement' / f'*-vh-*.tif')
    safe_pattern_vvtif = str(raw_dir / '*.SAFE' / 'measurement' / f'*-vv-*.tif')
    
    direct_pattern_vv = str(raw_dir / f'*-vv-*.tiff')
    direct_pattern_vh = str(raw_dir / f'*-vh-*.tiff')
    direct_pattern_tif = str(raw_dir / f'*-vv-*.tif')
    direct_pattern_tif_vh = str(raw_dir / f'*-vh-*.tif')

    files = []
    files.extend(glob(safe_pattern_vh))
    files.extend(glob(safe_pattern_vv))
    files.extend(glob(safe_pattern_vhtif))
    files.extend(glob(safe_pattern_vvtif))
    files.extend(glob(direct_pattern_vv))
    files.extend(glob(direct_pattern_vh))
    files.extend(glob(direct_pattern_tif))
    files.extend(glob(direct_pattern_tif_vh))

    files = sorted(set(files))
    
    return files


def get_info(sar_path):
    path = Path(sar_path)
    filename = path.stem  
    
    parts = filename.split('-')
    
    info = {
        'filename': filename,
        'satellite': parts[0] if len(parts) > 0 else 'unknown', 
        'mode': parts[1] if len(parts) > 1 else 'unknown',        
        'product': parts[2] if len(parts) > 2 else 'unknown',     
        'polarization': parts[3] if len(parts) > 3 else 'unknown', 
    }

    if len(parts) > 4:
        date_str = parts[4]  # e.g., 20260117t122220
        if len(date_str) >= 8:
            info['date'] = date_str[:8]  # 20260117
    
    return info