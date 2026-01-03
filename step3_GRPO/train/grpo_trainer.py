"""GRPO Training Loop with FSDP support.

Implements Group Relative Policy Optimization (GRPO) for distributed training.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional

from step3_GRPO.train.fsdp_utils import print_rank_0, is_main_process
from step3_GRPO.train.rewards_math import compute_math_rewards
from step3_GRPO.train.advantage_group import compute_group_advantages


def create_optimizer_and_scheduler(model: FSDP, config, num_training_steps: int):
    """Create optimizer and learning rate scheduler for FSDP model.
    
    Args:
        model: FSDP-wrapped model
        config: Configuration object
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    print_rank_0("Creating optimizer and scheduler...")
    
    # Create AdamW optimizer
    # Note: FSDP with use_orig_params=True allows standard optimizer access
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )
    
    # Create linear warmup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # GRPO typically doesn't use warmup
        num_training_steps=num_training_steps
    )
    
    print_rank_0(f"✓ Optimizer: AdamW (lr={config.training.learning_rate:.2e})")
    print_rank_0(f"✓ Scheduler: Linear warmup")
    
    return optimizer, scheduler


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    config,
    rank: int
) -> Optional[str]:
    """Save FSDP checkpoint (only from rank 0).
    
    Args:
        model: FSDP model
        optimizer: Optimizer
        step: Current training step
        config: Configuration
        rank: Current process rank
        
    Returns:
        Path to checkpoint (only on rank 0, None on other ranks)
    """
    if rank != 0:
        return None
    
    checkpoint_dir = Path(config.checkpointing.output_dir) / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print_rank_0(f"\nSaving checkpoint to {checkpoint_dir}...")
    
    # Save FSDP full state dict (consolidated on rank 0)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        
        if rank == 0:
            # Save model
            torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
            
            # Save optimizer
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            
            # Save training state
            torch.save({
                'step': step,
            }, checkpoint_dir / "training_state.pt")
            
            print_rank_0(f"✓ Checkpoint saved")
    
    return str(checkpoint_dir)


def generate_completions_batch(
    model: FSDP,
    prompts: list,
    tokenizer,
    config,
    device
):
    """Generate K completions for each prompt using sampling.
    
    Args:
        model: FSDP model
        prompts: List of prompt strings
        tokenizer: Tokenizer
        config: Configuration with GRPO params
        device: Device to run on
        
    Returns:
        List of generated completion strings
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    model.eval()
    
    completions = []
    
    with torch.no_grad():
        # Tokenize prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Use FSDP's summon_full_params to temporarily gather all parameters for generation
        # This is necessary because .generate() doesn't work with sharded parameters
        with FSDP.summon_full_params(model, writeback=False):
            # Generate completions with sampling
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=config.grpo.max_tokens,
                min_new_tokens=config.grpo.get('min_tokens', 4),
                temperature=config.grpo.temperature,
                do_sample=True,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (exclude prompt)
        for i, output_ids in enumerate(outputs):
            prompt_length = inputs['input_ids'][i].shape[0]
            generated_ids = output_ids[prompt_length:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
            completions.append(completion)
    
    model.train()
    return completions


def compute_log_probs(
    model: FSDP,
    prompts: list,
    completions: list,
    tokenizer,
    device
) -> torch.Tensor:
    """Compute log probabilities of completions given prompts.
    
    Args:
        model: FSDP model
        prompts: List of prompt strings
        completions: List of completion strings
        tokenizer: Tokenizer
        device: Device
        
    Returns:
        Log probabilities tensor [batch_size]
    """
    model.eval()
    
    log_probs_list = []
    
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            # Tokenize prompt + completion
            full_text = prompt + completion
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)
            
            # Tokenize just prompt to get length
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            prompt_length = prompt_inputs['input_ids'].shape[1]
            
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )
            
            # Get logits for completion tokens
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Compute log probs for completion tokens
            completion_log_probs = []
            for i in range(prompt_length - 1, inputs['input_ids'].shape[1] - 1):
                token_id = inputs['input_ids'][0, i + 1]
                token_logits = logits[i]
                token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                completion_log_probs.append(token_log_prob.item())
            
            # Sum log probs
            total_log_prob = sum(completion_log_probs) if completion_log_probs else 0.0
            log_probs_list.append(total_log_prob)
    
    model.train()
    return torch.tensor(log_probs_list, dtype=torch.float32)


def grpo_training_step(
    model: FSDP,
    batch: dict,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    config,
    device,
    step: int
) -> dict:
    """Single GRPO training step.
    
    GRPO Algorithm:
    1. Sample K completions per prompt
    2. Compute rewards for each completion
    3. Compute advantages using group normalization
    4. Update policy to maximize advantage-weighted log-probs
    
    Args:
        model: FSDP model
        batch: Batch of examples with 'problem' and 'solution'
        tokenizer: Tokenizer
        optimizer: Optimizer
        config: Configuration
        device: Device
        step: Current step
        
    Returns:
        Dictionary of metrics
    """
    K = config.grpo.group_size
    
    # Extract data from batch
    problems = [item['problem'] for item in batch]
    solutions = [item['solution'] for item in batch]
    
    # Create prompts (use prompt template if available)
    if hasattr(config.data, 'prompt_file'):
        with open(config.data.prompt_file, 'r') as f:
            prompt_template = f.read()
        prompts = [prompt_template.replace('{question}', p) for p in problems]
    else:
        prompts = problems
    
    # === Phase 1: Sample K completions per prompt (BATCHED) ===
    # Instead of processing one prompt at a time, process all prompts together
    # This maximizes GPU utilization
    
    # Repeat each prompt K times
    all_prompts = []
    all_ground_truths = []
    for prompt, gt in zip(prompts, solutions):
        all_prompts.extend([prompt] * K)
        all_ground_truths.extend([gt] * K)
    
    # Generate all completions in one large batch
    # This is much more efficient than generating K at a time
    print_rank_0(f"[Step {step}] Generating {len(all_prompts)} completions ({len(prompts)} problems × {K} samples)...")
    import time
    gen_start = time.time()
    all_completions = generate_completions_batch(
        model, all_prompts, tokenizer, config, device
    )
    gen_time = time.time() - gen_start
    print_rank_0(f"[Step {step}] ✓ Generation complete ({gen_time:.1f}s, {len(all_completions)/gen_time:.1f} completions/s)")
    
    # === Phase 2: Compute rewards using existing math reward function ===
    print_rank_0(f"[Step {step}] Computing rewards...")
    reward_start = time.time()
    rewards_list = compute_math_rewards(all_completions, all_ground_truths)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    reward_time = time.time() - reward_start
    print_rank_0(f"[Step {step}] ✓ Rewards computed ({reward_time:.1f}s, mean={rewards.mean().item():.3f})")
    
    # === Phase 3: Compute advantages using existing group advantage function ===
    print_rank_0(f"[Step {step}] Computing advantages...")
    advantages, adv_stats = compute_group_advantages(
        rewards=rewards,
        group_size=K,
        eps=config.grpo.advantage_eps,
        use_std_normalization=config.grpo.use_std_normalization
    )
    print_rank_0(f"[Step {step}] ✓ Advantages computed (mean={advantages.mean().item():.3f})")
    
    # === Phase 4: Policy gradient update ===
    print_rank_0(f"[Step {step}] Computing policy gradients...")
    model.train()
    
    # Compute current log probs (requires gradient)
    policy_log_probs = []
    
    for idx, (prompt, completion) in enumerate(zip(all_prompts, all_completions)):
        # Tokenize
        full_text = prompt + completion
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)
        
        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        
        logits = outputs.logits[0]
        
        # Compute log probs for completion tokens
        completion_log_prob = 0.0
        for i in range(prompt_length - 1, inputs['input_ids'].shape[1] - 1):
            token_id = inputs['input_ids'][0, i + 1]
            token_logits = logits[i]
            token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
            completion_log_prob = completion_log_prob + token_log_prob
        
        policy_log_probs.append(completion_log_prob)
    
    policy_log_probs = torch.stack(policy_log_probs)
    print_rank_0(f"[Step {step}] ✓ Policy log probs computed")
    
    # GRPO loss: -E[A * log π(y|x)]
    loss = -(advantages.detach() * policy_log_probs).mean()
    
    # Scale loss by gradient accumulation steps for proper averaging
    grad_accum_steps = config.training.get('gradient_accumulation_steps', 1)
    loss = loss / grad_accum_steps
    
    # Backward pass (accumulate gradients)
    print_rank_0(f"[Step {step}] Computing gradients (loss={loss.item()*grad_accum_steps:.4f})...")
    loss.backward()
    print_rank_0(f"[Step {step}] ✓ Backward pass complete")

    
    # Return unscaled loss for logging
    metrics = {
        'loss': (loss.item() * grad_accum_steps),  # Unscale for logging
        **adv_stats,  # Includes reward stats and advantage stats
    }
    
    return metrics


def grpo_training_loop(
    model: FSDP,
    train_loader,
    val_loader,
    tokenizer,
    config,
    rank: int,
    world_size: int,
    local_rank: int
):
    """Main GRPO training loop with gradient accumulation and final validation evaluation.
    
    Args:
        model: FSDP-wrapped model
        train_loader: DataLoader with DistributedSampler for training
        val_loader: DataLoader with DistributedSampler for validation
        tokenizer: Tokenizer
        config: Configuration
        rank: Global rank
        world_size: Total number of processes
        local_rank: Local GPU index
    """
    print_rank_0("\n" + "="*70)
    print_rank_0("Starting GRPO Training Loop")
    print_rank_0("="*70)
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Create optimizer and scheduler
    max_steps = config.training.max_steps
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, max_steps)
    
    # Get gradient accumulation steps
    grad_accum_steps = config.training.get('gradient_accumulation_steps', 1)
    print_rank_0(f"Gradient accumulation steps: {grad_accum_steps}")
    print_rank_0(f"Total training steps: {max_steps}")
    
    # Training loop
    model.train()
    global_step = 0
    accum_step = 0  # Track gradient accumulation
    
    progress_bar = tqdm(
        total=max_steps,
        desc="Training",
        disable=not is_main_process()
    )
    
    for epoch in range(100):  # Large number, will break with max_steps
        # Set epoch for DistributedSampler (important for proper shuffling!)
        train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break
            
            # GRPO training step (computes loss and backward, but doesn't update yet)
            print_rank_0(f"\n{'='*70}")
            print_rank_0(f"Microbatch {accum_step + 1}/{grad_accum_steps} (GRPO step will be {global_step + 1}/{max_steps})")
            print_rank_0(f"{'='*70}")
            
            metrics = grpo_training_step(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                config=config,
                device=device,
                step=global_step
            )
            
            accum_step += 1
            print_rank_0(f"✓ Microbatch {accum_step}/{grad_accum_steps} complete")
            
            # Only update weights after accumulating enough gradients
            if accum_step >= grad_accum_steps:
                print_rank_0(f"\n{'='*70}")
                print_rank_0(f"UPDATING WEIGHTS (GRPO Step {global_step + 1}/{max_steps})")
                print_rank_0(f"{'='*70}")
                
                # Gradient clipping
                if hasattr(config.training, 'max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    print_rank_0(f"✓ Gradients clipped")
                
                # Update weights
                optimizer.step()
                print_rank_0(f"✓ Weights updated")
                optimizer.zero_grad()
                print_rank_0(f"✓ Gradients zeroed")
                scheduler.step()
                print_rank_0(f"✓ Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                
                # Reset accumulation counter
                accum_step = 0
                
                # Increment global step (only after actual weight update)
                global_step += 1
                
                # Logging
                if is_main_process() and global_step % config.logging.log_every_steps == 0:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/reward_mean': metrics['reward_mean'],
                        'train/reward_std': metrics['reward_std'],
                        'train/reward_max': metrics['reward_max'],
                        'train/reward_min': metrics['reward_min'],
                        'train/advantage_mean': metrics['advantage_mean'],
                        'train/advantage_std': metrics['advantage_std'],
                        'train/advantage_max': metrics['advantage_max'],
                        'train/advantage_min': metrics['advantage_min'],
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/step': global_step,
                    })
                
                # Update progress
                if is_main_process():
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'reward': f"{metrics['reward_mean']:.2f}"
                    })
                
                # Checkpointing
                if global_step % config.training.eval_every == 0 and global_step > 0:
                    dist.barrier()  # Sync before checkpoint
                    save_fsdp_checkpoint(model, optimizer, global_step, config, rank)
                    dist.barrier()  # Sync after checkpoint
        
        if global_step >= max_steps:
            break
    
    progress_bar.close()
    
    # Final checkpoint
    print_rank_0("\nSaving final checkpoint...")
    dist.barrier()
    final_checkpoint_path = save_fsdp_checkpoint(model, optimizer, global_step, config, rank)
    dist.barrier()
    
    print_rank_0("\n" + "="*70)
    print_rank_0("GRPO Training Complete!")
    print_rank_0("="*70)
    
    # ==============================================
    # Final Evaluation on Validation Set (rank 0 only)
    # ==============================================
    if is_main_process():
        print("\n" + "="*70)
        print("Running Final Evaluation on Validation Set")
        print("="*70)
        
        try:
            from step3_GRPO.eval.vllm_eval import run_vllm_evaluation
            
            # Load prompt template
            with open(config.data.prompt_file, 'r') as f:
                prompt_template = f.read()
            
            # Use validation loader to get samples
            import random
            val_samples = []
            for batch in val_loader:
                for item in batch:
                    val_samples.append({
                        'problem': item['problem'],
                        'solution': item['solution'],
                    })
                    if len(val_samples) >= config.evaluation.num_samples:
                        break
                if len(val_samples) >= config.evaluation.num_samples:
                    break
            
            print(f"Evaluating on {len(val_samples)} validation samples...")
            
            # Run evaluation with vLLM
            eval_results = run_vllm_evaluation(
                checkpoint_path=final_checkpoint_path,
                test_data=val_samples,
                prompt_template=prompt_template,
                batch_size=config.evaluation.batch_size,
                max_tokens=config.evaluation.max_tokens,
                temperature=config.evaluation.temperature,
                gpu_id=0,  # Use GPU 0 for evaluation
            )
            
            # Log to W&B
            wandb.log({
                'eval_final/accuracy': eval_results['accuracy'],
                'eval_final/avg_length': eval_results['avg_length'],
                'eval_final/valid_format_rate': eval_results['valid_format_rate'],
                'eval_final/eval_time': eval_results['eval_time'],
            })
            
            print("\n✓ Final evaluation complete!")
            print(f"  Final Validation Accuracy: {eval_results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"\n⚠ Final evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Wait for evaluation to complete
    dist.barrier()
