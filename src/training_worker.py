"""Training worker for GPU 0 - main training loop."""
import torch
import wandb
import os
from pathlib import Path
from typing import Optional
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config_loader import Config


def prepare_optimizer_and_scheduler(
    model: AutoModelForCausalLM,
    config: Config,
    num_training_steps: int
):
    """Prepare optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        config: Configuration object
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    step: int,
    config: Config
) -> str:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        step: Current training step
        config: Configuration object
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpointing.temp_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"step_{step}"
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    return str(checkpoint_path)


def training_loop(
    config: Config,
    train_data: list,
    tokenizer: AutoTokenizer,
    eval_queue: Optional[mp.Queue] = None,
    stop_signal: Optional[mp.Event] = None
) -> None:
    """Main training loop on GPU 0 with periodic training set evaluation.
    
    Args:
        config: Configuration object
        train_data: Training dataset
        tokenizer: Tokenizer
        eval_queue: Queue for sending checkpoint paths to eval worker
    """
    from src.logging_utils import DetailedEvaluationLogger, compute_token_entropy, compute_response_length
    from src.analysis_utils import (
        categorize_results,
        analyze_format_failures,
        generate_summary_report,
        save_analysis_report,
        print_analysis_summary
    )
    
    print("="*80)
    print("STARTING TRAINING WORKER (GPU 0)")
    print("="*80)
    
    # Set device - MUST set CUDA_VISIBLE_DEVICES for GPU isolation
    device = config.training.device
    
    import os
    if "cuda:" in device:
        gpu_idx = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        print(f"✓ Set CUDA_VISIBLE_DEVICES={gpu_idx} for training")
    
    torch.cuda.set_device(0)  # Now GPU 0 is the only visible device
    
    # Load model
    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
    ).to(device)
    
    model.train()
    print("✓ Model loaded and in training mode")
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # Calculate total training steps
    num_training_steps = min(len(train_data), config.training.max_batches)
    
    # Prepare optimizer and scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        model, config, num_training_steps
    )
    
    print(f"Training for {num_training_steps} steps")
    print(f"Eval every {config.training.eval_every} steps")
    print("="*80)
    
    # Training loop
    train_step = 0
    progress_bar = tqdm(total=num_training_steps, desc="Training")
    
    for example in train_data[:num_training_steps]:
        # Check for early stop signal
        if stop_signal and stop_signal.is_set():
            print(f"\n⚠ Early stop signal received at step {train_step}")
            break
        # Tokenize
        from training_utils import prepare_sft_batch
        batch = prepare_sft_batch([example], tokenizer)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        if train_step % config.logging.log_every == 0:
            # Log GPU stats (use cuda:0 since CUDA_VISIBLE_DEVICES remaps)
            from src.gpu_monitor import log_gpu_stats_to_wandb
            gpu_stats = log_gpu_stats_to_wandb("cuda:0", prefix="train_gpu", step=train_step)
            
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train_step": train_step,
            })
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Checkpoint and evaluation
        if train_step % config.training.eval_every == 0 and train_step > 0:
            print(f"\n[Step {train_step}] Saving checkpoint...")
            checkpoint_path = save_checkpoint(model, tokenizer, train_step, config)
            
            # Push to eval queue (non-blocking)
            if eval_queue is not None:
                try:
                    if not eval_queue.full():
                        eval_queue.put(checkpoint_path)
                        print(f"✓ Checkpoint queued for evaluation")
                    else:
                        print(f"⚠ Eval queue full, skipping...")
                except Exception as e:
                    print(f"⚠ Failed to queue checkpoint: {e}")
        
        train_step += 1
    
    progress_bar.close()
    
    # Final checkpoint
    print(f"\n[Final] Saving final checkpoint...")
    final_checkpoint_path = save_checkpoint(model, tokenizer, train_step, config)
    
    # Signal eval worker to stop
    if eval_queue is not None:
        eval_queue.put(None)
        print("✓ Sent shutdown signal to eval worker")
    
    # Save final model to output directory
    output_dir = Path(config.checkpointing.output_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Final model saved to {output_dir}")
    
    print("="*80)
    print("TRAINING COMPLETED")
    print("="*80)


def evaluate_train_subset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_data: list,
    config: Config,
    train_step: int,
    device: str
) -> dict:
    """Evaluate model on training subset with comprehensive analysis.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        train_data: Training dataset
        config: Configuration
        train_step: Current training step
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    from transformers import StoppingCriteria, StoppingCriteriaList
    from src.logging_utils import DetailedEvaluationLogger, compute_token_entropy, compute_response_length
    from src.analysis_utils import (
        categorize_results,
        analyze_format_failures,
        generate_summary_report,
        save_analysis_report,
        print_analysis_summary
    )
    
    # Sample subset for evaluation (e.g., 100 examples)
    num_eval = min(100, len(train_data))
    eval_samples = train_data[:num_eval]
    
    # Initialize logger
    logger = DetailedEvaluationLogger(
        log_dir="results/train_eval_logs",
        eval_step=train_step
    )
    
    # Custom stopping criteria
    class StopOnToken(StoppingCriteria):
        def __init__(self, stop_ids):
            self.stop_ids = stop_ids
        
        def __call__(self, input_ids, scores, **kwargs):
            for stop_id_seq in self.stop_ids:
                if len(input_ids[0]) >= len(stop_id_seq):
                    if input_ids[0][-len(stop_id_seq):].tolist() == stop_id_seq:
                        return True
            return False
    
    stop_str = "</answer>"
    stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([StopOnToken([stop_ids])])
    
    model.eval()
    all_results = []
    correct = 0
    format_correct = 0
    
    print(f"  Generating {num_eval} training responses...")
    with torch.no_grad():
        for ex in tqdm(eval_samples, desc="  Train eval"):
            # Generate response
            encoded = tokenizer(ex['prompt'], return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.generation.max_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
            
            generated_ids = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            
            # Grade response
            from utils.drgrpo_grader import r1_zero_reward_fn
            ground_truth = ex.get('solution', '')
            problem = ex.get('problem', '')
            solution = ex.get('solution', '')
            reward_dict = r1_zero_reward_fn(response, ground_truth, fast=True)
            
            # Compute metrics
            token_entropy = compute_token_entropy(response, tokenizer)
            response_length = compute_response_length(response, tokenizer)
            
            # Create result
            result = {
                'prompt': ex['prompt'],
                'response': response,
                'ground_truth': ground_truth,
                'rewards': {
                    'format_reward': reward_dict['format_reward'],
                    'answer_reward': reward_dict['answer_reward'],
                    'total_reward': reward_dict['reward']
                },
                'metrics': {
                    'token_entropy': token_entropy,
                    'response_length': response_length
                }
            }
            all_results.append(result)
            
            # Log
            logger.log_test_case(
                prompt=ex['prompt'],
                response=response,
                ground_truth=ground_truth,
                format_reward=reward_dict['format_reward'],
                answer_reward=reward_dict['answer_reward'],
                total_reward=reward_dict['reward'],
                token_entropy=token_entropy,
                response_length=response_length,
                problem=problem,
                solution=solution
            )
            
            if reward_dict['format_reward'] == 1.0:
                format_correct += 1
            if reward_dict['reward'] == 1.0:
                correct += 1
    
    model.train()  # Back to training mode
    
    # Save logs
    summary_stats = logger.save()
    
    # Categorize and analyze
    print("  Categorizing training results...")
    category_1, category_2, category_3 = categorize_results(all_results)
    format_issues = analyze_format_failures(category_3)
    
    # Compute metrics
    metrics = {
        'accuracy': correct / num_eval,
        'format_accuracy': format_correct / num_eval,
        'num_evaluated': num_eval,
        'correct': correct,
        'format_correct': format_correct,
        'avg_response_length': summary_stats['avg_response_length'],
        'avg_response_length_correct': summary_stats['avg_response_length_correct'],
        'avg_response_length_incorrect': summary_stats['avg_response_length_incorrect'],
        'avg_token_entropy': summary_stats['avg_token_entropy']
    }
    
    # Generate and save summary
    summary_report = generate_summary_report(
        category_1, category_2, category_3,
        format_issues, metrics, train_step
    )
    
    save_analysis_report(
        summary_report, category_1, category_2, category_3,
        "results/train_analysis", train_step
    )
    
    print_analysis_summary(summary_report, category_3, category_2)
    
    return metrics
