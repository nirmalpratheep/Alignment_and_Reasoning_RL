"""Distributed training utilities for FSDP.

This module provides utilities for:
- Process group initialization with NCCL backend
- GPU binding for distributed ranks
- Environment variable reading and validation
"""
import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional


def get_distributed_info() -> Tuple[int, int, int]:
    """Read and validate distributed environment variables set by torchrun.
    
    Environment variables set by torchrun:
    - LOCAL_RANK: GPU index on this node (0-indexed)
    - RANK: Global rank across all nodes (0-indexed)
    - WORLD_SIZE: Total number of processes
    
    Returns:
        Tuple of (rank, world_size, local_rank)
        
    Raises:
        ValueError: If required environment variables are not set
    """
    # Read environment variables
    local_rank = os.environ.get('LOCAL_RANK')
    rank = os.environ.get('RANK')
    world_size = os.environ.get('WORLD_SIZE')
    
    # Validate that all required variables are present
    if local_rank is None or rank is None or world_size is None:
        raise ValueError(
            "Required distributed environment variables not set. "
            "Please launch with torchrun: "
            "torchrun --nproc_per_node=2 script.py"
        )
    
    # Convert to integers
    local_rank = int(local_rank)
    rank = int(rank)
    world_size = int(world_size)
    
    return rank, world_size, local_rank


def init_process_group_nccl(timeout_minutes: int = 30) -> None:
    """Initialize NCCL process group for distributed GPU training.
    
    NCCL (NVIDIA Collective Communications Library) is optimized for
    multi-GPU communication and supports features like NVLink.
    
    Args:
        timeout_minutes: Timeout for initialization in minutes
        
    Raises:
        RuntimeError: If process group initialization fails
    """
    if dist.is_initialized():
        print_rank_0("Warning: Process group already initialized, skipping...")
        return
    
    # Initialize with NCCL backend
    timeout = torch.distributed.timedelta(minutes=timeout_minutes)
    
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timeout,
        )
        print_rank_0(f"✓ NCCL process group initialized (timeout: {timeout_minutes}m)")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize NCCL process group: {e}")


def setup_distributed() -> Tuple[int, int, int]:
    """Master function to set up distributed training environment.
    
    This function orchestrates the complete distributed setup:
    1. Reads environment variables (LOCAL_RANK, RANK, WORLD_SIZE)
    2. Sets CUDA device based on local_rank
    3. Initializes NCCL process group
    
    CRITICAL RULE: Call this BEFORE creating any models or CUDA tensors.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
        
    Example:
        >>> rank, world_size, local_rank = setup_distributed()
        >>> print(f"Rank {rank}/{world_size} on GPU {local_rank}")
    """
    # Step 1: Get distributed info from environment
    rank, world_size, local_rank = get_distributed_info()
    
    # Step 2: Set CUDA device BEFORE any model creation
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    if local_rank >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK {local_rank} >= available GPUs {torch.cuda.device_count()}"
        )
    
    torch.cuda.set_device(local_rank)
    
    # Step 3: Initialize process group
    init_process_group_nccl()
    
    # Print confirmation from rank 0 only
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Distributed Training Setup Complete")
        print(f"{'='*70}")
        print(f"Backend: NCCL")
        print(f"World Size: {world_size}")
        print(f"Devices: {torch.cuda.device_count()} GPUs")
        print(f"Device Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        print(f"{'='*70}\n")
    
    # Wait for all processes to reach this point
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Clean up distributed process group.
    
    Call this at the end of training to properly shut down the process group.
    """
    if dist.is_initialized():
        dist.barrier()  # Ensure all processes are done
        dist.destroy_process_group()
        print_rank_0("✓ Process group destroyed")


def print_rank_0(message: str, force: bool = False) -> None:
    """Print message only from rank 0 to avoid duplicate logs.
    
    Args:
        message: Message to print
        force: If True, print from all ranks (for debugging)
    """
    if force or not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


def get_rank() -> int:
    """Get current process rank, returns 0 if not in distributed mode."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get world size, returns 1 if not in distributed mode."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Check if current process is rank 0 (main process)."""
    return get_rank() == 0


def get_fsdp_wrap_policy(model):
    """Get transformer-based auto-wrap policy for FSDP.
    
    This policy wraps each transformer layer (decoder block) separately,
    which provides good balance between memory efficiency and communication overhead.
    
    Args:
        model: The model to wrap (used to detect layer type)
        
    Returns:
        FSDP auto-wrap policy function
    """
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial
    
    # Dynamically detect the transformer layer class from the model
    # This works for any transformer model (Qwen2, Llama, GPT, etc.)
    transformer_layer_cls = None
    
    # Try to find the decoder layer class from the model
    for module in model.modules():
        module_name = module.__class__.__name__
        if 'DecoderLayer' in module_name or 'Block' in module_name or 'Layer' in module_name:
            # Found a likely transformer layer
            if hasattr(module, 'self_attn') or hasattr(module, 'attn'):
                transformer_layer_cls = module.__class__
                print_rank_0(f"  Detected transformer layer: {transformer_layer_cls.__name__}")
                break
    
    if transformer_layer_cls is None:
        raise ValueError("Could not detect transformer layer class from model")
    
    # Create the auto-wrap policy using functools.partial
    # This returns a callable that FSDP can use
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )
    
    return auto_wrap_policy


def get_fsdp_mixed_precision_policy(dtype_str: str = "bfloat16"):
    """Get FSDP mixed precision policy.
    
    Args:
        dtype_str: Data type string ("bfloat16" or "float16")
        
    Returns:
        MixedPrecision policy for FSDP
    """
    from torch.distributed.fsdp import MixedPrecision
    
    # Map string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    # Mixed precision configuration
    # - param_dtype: Keep parameters in mixed precision
    # - reduce_dtype: Use mixed precision for gradient reduce
    # - buffer_dtype: Keep buffers in full precision for stability
    mixed_precision_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=torch.float32,  # Buffers stay in FP32 for numerical stability
    )
    
    return mixed_precision_policy


def load_model_for_fsdp(config, local_rank: int):
    """Load model on CPU or meta device for FSDP wrapping.
    
    IMPORTANT: Models should be loaded on CPU/meta device BEFORE FSDP wrapping.
    FSDP will handle moving shards to GPU.
    
    Args:
        config: Configuration object with model settings
        local_rank: Local rank for this process
        
    Returns:
        Loaded model (on CPU/meta device)
    """
    from transformers import AutoModelForCausalLM
    
    print_rank_0(f"Loading model: {config.model.name}")
    print_rank_0(f"  dtype: {config.model.dtype}")
    print_rank_0(f"  trust_remote_code: {config.model.trust_remote_code}")
    
    # Determine if we should use Flash Attention 2
    attn_implementation = None
    if hasattr(config.model, 'attn_implementation'):
        attn_implementation = config.model.attn_implementation
        print_rank_0(f"  attention: {attn_implementation}")
    
    # Load model on CPU first (FSDP will shard and move to GPU)
    # Using device_map="cpu" to avoid automatic GPU placement
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=attn_implementation,
        device_map=None,  # Don't auto-place on GPU
    )
    
    # Move to CPU explicitly
    model = model.cpu()
    
    print_rank_0(f"✓ Model loaded on CPU (ready for FSDP wrapping)")
    
    # Print model info
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def wrap_model_with_fsdp(model, config, local_rank: int):
    """Wrap model with FSDP for distributed training.
    
    Args:
        model: Pre-loaded model (on CPU)
        config: Configuration object
        local_rank: Local rank for GPU assignment
        
    Returns:
        FSDP-wrapped model
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    
    print_rank_0("\nWrapping model with FSDP...")
    print_rank_0(f"  Sharding strategy: {config.fsdp.sharding_strategy}")
    print_rank_0(f"  Mixed precision: {config.fsdp.mixed_precision}")
    
    # Get sharding strategy
    sharding_strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = sharding_strategy_map.get(
        config.fsdp.sharding_strategy, 
        ShardingStrategy.FULL_SHARD
    )
    
    # Get auto-wrap policy (pass model to detect layer class)
    auto_wrap_policy = get_fsdp_wrap_policy(model)
    
    # Get mixed precision policy
    mixed_precision = get_fsdp_mixed_precision_policy(config.fsdp.mixed_precision)
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=local_rank,  # This moves model to GPU
        sync_module_states=config.fsdp.get('sync_module_states', True),
        forward_prefetch=config.fsdp.get('forward_prefetch', True),
        limit_all_gathers=config.fsdp.get('limit_all_gathers', True),
        use_orig_params=True,  # Important for optimizer state dict
    )
    
    print_rank_0(f"✓ Model wrapped with FSDP on GPU {local_rank}")
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print_rank_0("✓ Gradient checkpointing enabled")
    
    # Compile model if requested (PyTorch 2.0+)
    if hasattr(config.model, 'torch_compile') and config.model.torch_compile:
        print_rank_0("Compiling model with torch.compile...")
        compile_mode = getattr(config.model, 'compile_mode', 'default')
        model = torch.compile(model, mode=compile_mode)
        print_rank_0(f"✓ Model compiled (mode: {compile_mode})")
    
    return model


def load_and_wrap_fsdp_model(config, local_rank: int):
    """Convenience function to load and wrap model with FSDP in one call.
    
    Args:
        config: Configuration object
        local_rank: Local rank for GPU assignment
        
    Returns:
        FSDP-wrapped model ready for training
    """
    # Load model on CPU
    model = load_model_for_fsdp(config, local_rank)
    
    # Wrap with FSDP
    model = wrap_model_with_fsdp(model, config, local_rank)
    
    # Synchronize all ranks
    dist.barrier()
    
    print_rank_0("\n" + "="*70)
    print_rank_0("FSDP Model Loading Complete")
    print_rank_0("="*70)
    
    return model
