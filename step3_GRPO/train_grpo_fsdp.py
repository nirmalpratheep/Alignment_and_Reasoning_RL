"""Main training script for FSDP GRPO on 2× H100 GPUs.

Launch with:
    torchrun --nproc_per_node=2 train_grpo_fsdp.py --config configs/fsdp_2gpu.yaml
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from step3_GRPO.train.fsdp_grpo_trainer import (
    setup_distributed,
    cleanup_distributed,
    load_model_and_tokenizer,
    create_fsdp_model,
    grpo_training_step,
    save_fsdp_checkpoint,
    FSDPConfig,
)
from step3_GRPO.eval.vllm_eval import run_vllm_evaluation, load_gsm8k_testset
from utils.dataset_loader import MathDatasetLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FSDP GRPO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="step3_GRPO/configs/fsdp_2gpu.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    # Resolve path: try multiple locations
    original_path = config_path
    if not Path(config_path).is_absolute():
        # Try in order:
        # 1. Current working directory
        # 2. step3_GRPO directory (where this script is)
        # 3. Project root
        script_dir = Path(__file__).parent
        possible_paths = [
            Path(config_path),  # Current working directory
            script_dir / config_path,  # Relative to script location
            project_root / config_path,  # Project root
            project_root / "step3_GRPO" / config_path,  # Explicit step3_GRPO path
        ]
        
        resolved_path = None
        for path in possible_paths:
            if path.exists():
                resolved_path = path
                break
        
        if resolved_path is None:
            # Use script directory as fallback
            resolved_path = script_dir / original_path
        config_path = resolved_path
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found. Tried: {original_path}")
    
    with open(str(config_path), 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: dict, rank: int) -> tuple:
    """Load and prepare training data.
    
    Returns:
        (train_data, test_data, prompt_template)
    """
    if rank == 0:
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)
    
    # Load MATH dataset
    loader = MathDatasetLoader()
    datasets, subsets, total_train, total_test = loader.load_all_subsets()
    
    if rank == 0:
        print(f"Loaded {len(subsets)} subsets")
        print(f"Total train: {total_train}, Total test: {total_test}")
    
    # Collect examples
    train_examples = loader.collect_train_examples(include_metadata=True)
    test_examples = loader.collect_test_examples(include_metadata=True)
    
    # Load prompt template
    prompt_file = config['data']['prompt_file']
    # Resolve path relative to project root if not absolute
    if not Path(prompt_file).is_absolute():
        prompt_file = project_root / prompt_file
    with open(str(prompt_file), 'r') as f:
        prompt_template = f.read()
    
    if rank == 0:
        print(f"✓ Train samples: {len(train_examples)}")
        print(f"✓ Test samples: {len(test_examples)}")
        print("="*80)
    
    return train_examples, test_examples, prompt_template


def initialize_wandb(config: dict, rank: int) -> tuple:
    """Initialize W&B run and share ID across ranks.
    
    Returns:
        (run_id, run_name)
    """
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"fsdp-grpo-{timestamp}"
        
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging'].get('wandb_entity'),
            name=run_name,
            config=config,
            tags=["fsdp", "h100", "grpo", "flash-attn-2", "2gpu"],
        )
        
        run_id = wandb.run.id
        print(f"\n✓ W&B initialized: {run_name} (ID: {run_id})")
    else:
        run_id = None
        run_name = None
    
    # Broadcast run_id to all ranks
    run_id_list = [run_id] if rank == 0 else [None]
    dist.broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]
    
    if rank != 0:
        print(f"Rank {rank}: Using W&B run ID: {run_id}")
    
    return run_id, run_name


def create_data_batches(
    train_data: List[Dict],
    batch_size_per_gpu: int,
    max_steps: int,
    rank: int,
) -> List[tuple]:
    """Create batches of training data.
    
    Returns:
        List of (prompts, ground_truths) tuples
    """
    import random
    
    batches = []
    
    for step in range(max_steps):
        # Sample batch
        batch_examples = random.sample(train_data, batch_size_per_gpu)
        
        prompts = [ex['problem'] for ex in batch_examples]
        ground_truths = [ex.get('solution', '') for ex in batch_examples]
        
        batches.append((prompts, ground_truths))
    
    return batches


def main():
    """Main training loop."""
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Load config
    config = load_config(args.config)
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    
    if rank == 0:
        print("\n" + "="*80)
        print("2× H100 FSDP GRPO TRAINING")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Model: {config['model']['name']}")
        print(f"FlashAttention-2: {config['model']['attn_implementation']}")
        print(f"torch.compile: {config['model']['torch_compile']}")
        print(f"Max steps: {config['training']['max_steps']}")
        print("="*80)
    
    # Initialize W&B
    run_id, run_name = initialize_wandb(config, rank)
    
    # Load datasets
    train_data, test_data, prompt_template = prepare_dataset(config, rank)
    
    # Load model and tokenizer
    model_name = config['model']['name']
    # Resolve model path relative to project root if it's a local path
    if not Path(model_name).is_absolute():
        model_path = project_root / model_name
        if model_path.exists():
            model_name = str(model_path)
    
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        dtype=config['model']['dtype'],
        attn_implementation=config['model']['attn_implementation'],
        use_compile=config['model']['torch_compile'],
        compile_mode=config['model']['compile_mode'],
        rank=rank,
    )
    
    # Wrap with FSDP
    if rank == 0:
        print("\nWrapping model with FSDP...")
    
    fsdp_config = FSDPConfig(
        sharding_strategy=config['fsdp']['sharding_strategy'],
        mixed_precision=config['fsdp']['mixed_precision'],
        forward_prefetch=config['fsdp']['forward_prefetch'],
        limit_all_gathers=config['fsdp']['limit_all_gathers'],
        sync_module_states=config['fsdp']['sync_module_states'],
    )
    
    fsdp_model = create_fsdp_model(model, fsdp_config, local_rank)
    
    if rank == 0:
        print("✓ FSDP model created")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas'],
    )
    
    # Create training batches
    batches = create_data_batches(
        train_data,
        config['training']['batch_size_per_gpu'],
        config['training']['max_steps'],
        rank,
    )
    
    # Training loop
    if rank == 0:
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
    
    global_step = 0
    
    for step, (prompts, ground_truths) in enumerate(tqdm(batches, disable=rank!=0)):
        global_step = step + 1
        
        # Training step
        metrics = grpo_training_step(
            model=fsdp_model,
            optimizer=optimizer,
            prompts=prompts,
            ground_truths=ground_truths,
            tokenizer=tokenizer,
            config=config,
            rank=rank,
        )
        
        # Log metrics (rank 0 only)
        if rank == 0:
            # Add GPU stats
            metrics.update({
                'train/gpu0_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                'train/gpu1_memory_allocated_gb': torch.cuda.memory_allocated(1) / 1e9,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/global_step': global_step,
            })
            
            # Rename keys for W&B
            wandb_metrics = {f"train/{k}": v for k, v in metrics.items() if not k.startswith('train/')}
            wandb_metrics.update({k: v for k, v in metrics.items() if k.startswith('train/')})
            
            wandb.log(wandb_metrics, step=global_step)
        
        # Evaluation (rank 1, time-sliced)
        eval_interval = config['training']['eval_every']
        if rank == 1 and global_step % eval_interval == 0:
            # Save checkpoint first
            output_dir = config['checkpointing']['output_dir']
            # Resolve path relative to project root if not absolute
            if not Path(output_dir).is_absolute():
                output_dir = project_root / output_dir
            checkpoint_path = save_fsdp_checkpoint(
                fsdp_model, optimizer, global_step,
                str(output_dir),
                rank,
            )
            
            # Join W&B run
            wandb.init(
                project=config['logging']['wandb_project'],
                id=run_id,
                resume="allow"
            )
            
            # Run evaluation
            try:
                results = run_vllm_evaluation(
                    checkpoint_path=checkpoint_path,
                    test_data=test_data[:config['evaluation']['num_samples']],
                    prompt_template=prompt_template,
                    batch_size=config['evaluation']['batch_size'],
                    max_tokens=config['evaluation']['max_tokens'],
                    temperature=config['evaluation']['temperature'],
                    gpu_id=1,
                )
                
                # Log eval metrics
                eval_metrics = {
                    'eval/gsm8k_accuracy': results['gsm8k_acc'],
                    'eval/overall_accuracy': results['overall_acc'],
                    'eval/avg_solution_length': results['avg_length'],
                    'eval/valid_format_rate': results['valid_format_rate'],
                    'eval/eval_time_seconds': results['eval_time'],
                    'eval/checkpoint_step': global_step,
                }
                wandb.log(eval_metrics, step=global_step)
                
                # Log sample outputs
                table = wandb.Table(
                    columns=["Problem", "Prediction", "Ground Truth", "Correct"],
                    data=[[s['problem'], s['pred'], s['gt'], s['correct']] 
                          for s in results['samples']]
                )
                wandb.log({"eval/sample_outputs": table}, step=global_step)
                
            except Exception as e:
                print(f"Evaluation failed: {e}")
    
    # Final checkpoint
    if rank == 0:
        output_dir = config['checkpointing']['output_dir']
        # Resolve path relative to project root if not absolute
        if not Path(output_dir).is_absolute():
            output_dir = project_root / output_dir
        save_fsdp_checkpoint(
            fsdp_model, optimizer, global_step,
            str(output_dir),
            rank,
        )
    
    # Cleanup
    if rank == 0:
        wandb.finish()
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
