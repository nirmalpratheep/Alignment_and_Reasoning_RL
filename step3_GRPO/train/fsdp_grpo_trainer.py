"""FSDP-based GRPO trainer for single-node 2× H100 GPUs."""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2DecoderLayer
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from step3_GRPO.train.rewards_math import batch_compute_rewards
from step3_GRPO.train.advantage_group import compute_group_advantages, compute_grpo_loss


@dataclass
class FSDPConfig:
    """FSDP configuration."""
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bfloat16"
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    sync_module_states: bool = True


def setup_distributed():
    """Initialize distributed training environment."""
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"=" * 80)
        print(f"DISTRIBUTED SETUP")
        print(f"=" * 80)
        print(f"World size: {world_size}")
        print(f"Backend: nccl")
        print(f"=" * 80)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def create_fsdp_model(
    model: torch.nn.Module,
    fsdp_config: FSDPConfig,
    device_id: int,
) -> FSDP:
    """Wrap model with FSDP.
    
    Args:
        model: Base model to wrap
        fsdp_config: FSDP configuration
        device_id: CUDA device ID
        
    Returns:
        FSDP-wrapped model
    """
    # Auto-wrap policy for transformer layers
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={Qwen2DecoderLayer},  # Qwen2 specific
    )
    
    # Mixed precision configuration
    if fsdp_config.mixed_precision == "bfloat16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision_policy = None
    
    # Sharding strategy
    if fsdp_config.sharding_strategy == "FULL_SHARD":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif fsdp_config.sharding_strategy == "SHARD_GRAD_OP":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.NO_SHARD
    
    # Wrap with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=device_id,
        forward_prefetch=fsdp_config.forward_prefetch,
        limit_all_gathers=fsdp_config.limit_all_gathers,
        sync_module_states=fsdp_config.sync_module_states,
    )
    
    return fsdp_model


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
    rank: int = 0,
) -> tuple:
    """Load model and tokenizer with H100 optimizations.
    
    Args:
        model_name: Model name or path
        dtype: Data type (bfloat16 recommended for H100)
        attn_implementation: Attention implementation (flash_attention_2 for H100)
        use_compile: Whether to use torch.compile
        compile_mode: Compilation mode
        rank: Process rank
        
    Returns:
        (model, tokenizer)
    """
    if rank == 0:
        print(f"\nLoading model: {model_name}")
        print(f"  dtype: {dtype}")
        print(f"  attention: {attn_implementation}")
        print(f"  compile: {use_compile} (mode={compile_mode})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        device_map=None,  # FSDP will handle device placement
    )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("  ✓ Gradient checkpointing enabled")
    
    # Compile model for H100 optimization
    if use_compile:
        if rank == 0:
            print(f"  Compiling model (mode={compile_mode})...")
        model = torch.compile(
            model,
            mode=compile_mode,
            fullgraph=False,  # More compatible
        )
        if rank == 0:
            print("  ✓ Model compiled")
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded: {num_params:,} parameters")
    
    return model, tokenizer


def generate_completions(
    model: FSDP,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    group_size: int,
    max_tokens: int,
    temperature: float,
    min_tokens: int = 4,
    rank: int = 0,
) -> tuple:
    """Generate multiple completions per prompt.
    
    Args:
        model: FSDP model
        tokenizer: Tokenizer
        prompts: List of prompts (batch_size,)
        group_size: Number of completions per prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        min_tokens: Minimum tokens to generate
        rank: Process rank
        
    Returns:
        (completions, log_probs): Lists of generated text and log probabilities
    """
    model.eval()
    
    batch_size = len(prompts)
    
    # Repeat each prompt group_size times
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * group_size)
    
    # Tokenize
    inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(f"cuda:{rank}")
    
    # Generate
    with torch.no_grad():
        outputs = model.module.generate(  # Access unwrapped model
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Decode completions
    generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
    completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Compute log probabilities (simplified - just for tracking)
    # In full implementation, would compute actual token-level log probs
    log_probs = torch.zeros(len(completions), device=f"cuda:{rank}")
    
    return completions, log_probs


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    rank: int,
):
    """Save FSDP checkpoint.
    
    Args:
        model: FSDP model
        optimizer: Optimizer
        step: Current training step
        output_dir: Output directory
        rank: Process rank
    """
    checkpoint_dir = Path(output_dir) / f"checkpoint_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FSDP state dict (sharded)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }
        
        if rank == 0:
            torch.save(state_dict, checkpoint_dir / f"rank{rank}_checkpoint.pt")
    
    # Synchronize
    dist.barrier()
    
    if rank == 0:
        print(f"✓ Checkpoint saved: {checkpoint_dir}")
    
    return str(checkpoint_dir)


def grpo_training_step(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    prompts: List[str],
    ground_truths: List[str],
    tokenizer: AutoTokenizer,
    config: dict,
    rank: int,
) -> dict:
    """Single GRPO training step.
    
    Args:
        model: FSDP model
        optimizer: Optimizer
        prompts: Batch of prompts
        ground_truths: Batch of ground truth solutions
        tokenizer: Tokenizer
        config: Configuration dictionary
        rank: Process rank
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # Generate completions (K per prompt)
    completions, log_probs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=config['grpo']['group_size'],
        max_tokens=config['grpo']['max_tokens'],
        temperature=config['grpo']['temperature'],
        min_tokens=config['grpo']['min_tokens'],
        rank=rank,
    )
    
    # Compute rewards
    rewards = batch_compute_rewards(
        prompts=prompts,
        completions=completions,
        ground_truths=ground_truths,
        group_size=config['grpo']['group_size'],
    )
    rewards_tensor = torch.tensor(rewards, device=f"cuda:{rank}")
    
    # Compute group-relative advantages
    advantages, adv_stats = compute_group_advantages(
        rewards=rewards_tensor,
        group_size=config['grpo']['group_size'],
        eps=config['grpo']['advantage_eps'],
        use_std_normalization=config['grpo']['use_std_normalization'],
    )
    
    # For now, use simplified loss (in full implementation, would do forward pass for log_probs)
    # This is a placeholder - real implementation needs token-level log prob computation
    loss = -((advantages * log_probs).mean())
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Collect metrics
    metrics = {
        'loss': loss.item(),
        'correct_rate': (rewards_tensor == 1.0).float().mean().item(),
        'generation_length_mean': sum(len(c.split()) for c in completions) / len(completions),
        **adv_stats,
    }
    
    return metrics
