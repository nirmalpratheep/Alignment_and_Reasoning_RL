"""Fast checkpoint utilities for vLLM weight synchronization.

This module provides utilities for saving lightweight FSDP checkpoints
that can be loaded by vLLM. The goal is to minimize I/O overhead while
maintaining compatibility with vLLM's checkpoint loading.
"""
import torch
from pathlib import Path
from typing import Optional
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

from step3_GRPO.train.fsdp_utils import print_rank_0


def save_temp_checkpoint(
    model: FSDP,
    step: int,
    temp_dir: str,
    rank: int,
    tokenizer=None
) -> Optional[str]:
    """Save lightweight temporary checkpoint for vLLM loading (rank 0 only).
    
    This saves only the model weights (no optimizer state) to minimize I/O overhead.
    The checkpoint is compatible with vLLM's loading mechanism.
    
    Args:
        model: FSDP model
        step: Current training step
        temp_dir: Directory for temporary checkpoints
        rank: Current process rank
        tokenizer: Optional tokenizer to save alongside model
        
    Returns:
        Path to checkpoint directory (only on rank 0, None on other ranks)
    """
    if rank != 0:
        return None
    
    checkpoint_dir = Path(temp_dir) / f"temp_step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print_rank_0(f"  Saving temp checkpoint to {checkpoint_dir}...")
    
    # Save FSDP full state dict (consolidated on rank 0)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        
        if rank == 0:
            # Save model weights only (vLLM compatible)
            torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
            
            # Save tokenizer if provided (vLLM needs this)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)
            
            print_rank_0(f"  ✓ Temp checkpoint saved")
    
    return str(checkpoint_dir)


def cleanup_old_temp_checkpoints(
    temp_dir: str,
    keep_last_n: int = 2,
    rank: int = 0
):
    """Clean up old temporary checkpoints to save disk space.
    
    Keeps only the N most recent checkpoints and deletes older ones.
    Only executes on rank 0 to avoid race conditions.
    
    Args:
        temp_dir: Directory containing temp checkpoints
        keep_last_n: Number of recent checkpoints to keep
        rank: Current process rank
    """
    if rank != 0:
        return
    
    temp_dir_path = Path(temp_dir)
    if not temp_dir_path.exists():
        return
    
    # Find all temp checkpoint directories
    checkpoint_dirs = sorted(
        [d for d in temp_dir_path.glob("temp_step_*") if d.is_dir()],
        key=lambda p: int(p.name.split("_")[-1])  # Sort by step number
    )
    
    # Delete old checkpoints
    if len(checkpoint_dirs) > keep_last_n:
        for old_dir in checkpoint_dirs[:-keep_last_n]:
            print_rank_0(f"  Cleaning up old temp checkpoint: {old_dir}")
            import shutil
            shutil.rmtree(old_dir)
