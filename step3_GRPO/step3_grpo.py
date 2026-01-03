"""FSDP GRPO Training Launcher for torchrun.

This is the entry point for distributed training with FSDP.
Launch with torchrun for multi-GPU training:
    
    torchrun --nproc_per_node=2 step3_grpo.py

Or use the convenience scripts:
    bash launch.sh
    .\launch.ps1  (Windows)
"""
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import wandb

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import distributed utilities from fsdp_utils
from step3_GRPO.train.fsdp_utils import (
    setup_distributed,
    cleanup_distributed,
    print_rank_0,
    is_main_process,
    load_and_wrap_fsdp_model,
)

# Import training components
from src.config_loader import load_config, validate_config
from src.data_utils import load_datasets
from step3_GRPO.train.grpo_trainer import grpo_training_loop


def main():
    """Main FSDP GRPO training launcher.
    
    Steps:
    1. Initialize distributed process group (NCCL)
    2. Set CUDA device for each rank
    3. Load configuration
    4. Load datasets
    5. TODO: Create model with FSDP wrapping
    6. TODO: Run training loop
    """
    
    # ==============================================
    # Step 1: Distributed Setup (CRITICAL - DO FIRST!)
    # ==============================================
    # This handles:
    # - Reading LOCAL_RANK, RANK, WORLD_SIZE from environment
    # - torch.cuda.set_device(local_rank)
    # - dist.init_process_group(backend="nccl")
    # - dist.barrier()
    
    print_rank_0("="*70)
    print_rank_0("FSDP GRPO Training Pipeline")
    print_rank_0("="*70)
    
    rank, world_size, local_rank = setup_distributed()
    
    print_rank_0(f"✓ Distributed setup complete")
    print_rank_0(f"  Rank: {rank}/{world_size}")
    print_rank_0(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    print_rank_0("")
    
    # ==============================================
    # Step 2: Load Configuration
    # ==============================================
    if is_main_process():
        print("Loading configuration...")
    
    config = load_config("config/grpo_config.yaml")
    validate_config(config)
    
    if is_main_process():
        print(f"✓ Config loaded: {config.model.name}")
        print("")
    
    # ==============================================
    # Step 3: Initialize W&B (rank 0 only)
    # ==============================================
    if is_main_process():
        print("Initializing W&B...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"grpo-fsdp-{world_size}gpu-{timestamp}"
        
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            config=config.to_dict(),
            notes=f"FSDP GRPO training - {world_size} GPUs",
            tags=["grpo", "fsdp", "math", f"{world_size}gpu"]
        )
        
        print(f"✓ W&B initialized: {run_name}")
        print(f"  Run ID: {wandb.run.id}")
        print("")
    
    # Wait for rank 0 to finish W&B setup
    dist.barrier()
    
    # ==============================================
    # Step 4: Load Datasets with DistributedSampler
    # ==============================================
    if is_main_process():
        print("Loading datasets with distributed sampling...")
    
    train_loader, val_loader, tokenizer = load_datasets(
        config, 
        prepare_sft_format=False,
        rank=rank,
        world_size=world_size
    )
    
    # Synchronize all ranks after data loading
    dist.barrier()
    
    # ==============================================
    # Step 5: Load Model with FSDP
    # ==============================================
    if is_main_process():
        print("Loading and wrapping model with FSDP...")
    
    # Load model and wrap with FSDP
    # This handles:
    # - Loading model on CPU first
    # - Wrapping with FSDP (FULL_SHARD strategy)
    # - Mixed precision (bfloat16)
    # - Moving sharded model to GPU
    # - Enabling gradient checkpointing
    # - Optional torch.compile for H100
    model = load_and_wrap_fsdp_model(config, local_rank)
    
    if is_main_process():
        print(f"✓ Model ready for training on {world_size} GPUs")
        print("")
    
    # Synchronize all ranks after model loading
    dist.barrier()
    
    # ==============================================
    # Step 6: GRPO Training Loop
    # ==============================================
    print_rank_0("\n" + "="*70)
    print_rank_0("Starting GRPO Training...")
    print_rank_0("="*70)
    
    # Run GRPO training loop
    grpo_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )
    
    print_rank_0("\n" + "="*70)
    print_rank_0("GRPO Training Complete!")
    print_rank_0("="*70)
    
    # ==============================================
    # Step 7: Cleanup
    # ==============================================
    dist.barrier()
    cleanup_distributed()
    print_rank_0("✓ Training session completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] ERROR: {e}")
        if dist.is_initialized():
            cleanup_distributed()
        raise
