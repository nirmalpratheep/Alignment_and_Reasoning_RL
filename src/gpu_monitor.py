"""GPU monitoring utilities for memory and utilization tracking."""
import torch
import wandb
from typing import Dict


def get_gpu_memory_stats(device: str = "cuda:0") -> Dict[str, float]:
    """Get GPU memory statistics.
    
    Args:
        device: CUDA device string
        
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0, "utilization_pct": 0.0}
    
    try:
        # Extract device index (always use 0 after CUDA_VISIBLE_DEVICES remapping)
        if "cuda:" in device:
            device_idx = int(device.split(":")[-1])
        else:
            device_idx = 0
        
        # Ensure CUDA is initialized
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024**3)  # GB
        
        # Get total and free memory
        total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # GB
        free = total - reserved
        
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "free_gb": round(free, 2),
            "total_gb": round(total, 2),
            "utilization_pct": round((allocated / total) * 100, 1) if total > 0 else 0.0
        }
    except Exception as e:
        print(f"⚠ GPU stats error: {e}")
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0, "utilization_pct": 0.0}


def log_gpu_stats_to_wandb(
    device: str,
    prefix: str = "gpu",
    step: int = None
) -> Dict[str, float]:
    """Log GPU stats to W&B.
    
    Args:
        device: CUDA device string
        prefix: Prefix for metric names
        step: Optional step number
        
    Returns:
        GPU stats dictionary
    """
    stats = get_gpu_memory_stats(device)
    
    log_data = {
        f"{prefix}/memory_allocated_gb": stats["allocated_gb"],
        f"{prefix}/memory_reserved_gb": stats["reserved_gb"],
        f"{prefix}/memory_max_allocated_gb": stats["max_allocated_gb"],
        f"{prefix}/memory_free_gb": stats["free_gb"],
        f"{prefix}/memory_utilization_pct": stats["utilization_pct"],
    }
    
    if step is not None:
        log_data["train_step"] = step
    
    wandb.log(log_data)
    
    return stats


def reset_peak_memory_stats(device: str = "cuda:0"):
    """Reset peak memory tracking.
    
    Args:
        device: CUDA device string
    """
    if torch.cuda.is_available():
        if "cuda:" in device:
            device_idx = int(device.split(":")[-1])
        else:
            device_idx = 0
        torch.cuda.reset_peak_memory_stats(device_idx)


def print_gpu_stats(device: str, label: str = ""):
    """Print GPU stats to console.
    
    Args:
        device: CUDA device string
        label: Optional label for the stats
    """
    stats = get_gpu_memory_stats(device)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}GPU Memory: {stats['allocated_gb']:.2f}GB allocated, "
          f"{stats['max_allocated_gb']:.2f}GB peak, "
          f"{stats['utilization_pct']:.1f}% utilization")
