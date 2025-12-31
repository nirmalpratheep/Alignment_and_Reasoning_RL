"""GRPO training worker for GPU 0 - reinforcement learning training loop."""
import torch
import wandb
import os
from pathlib import Path
from typing import Optional
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm

from src.config_loader import Config
from utils.drgrpo_grader import r1_zero_reward_fn


def prepare_grpo_dataset(train_data: list, prompt_template: str):
    """Prepare dataset for GRPO training (prompts only).
    
    Args:
        train_data: Training dataset with prompts and solutions
        prompt_template: Prompt template string
        
    Returns:
        List of dictionaries with 'prompt' and 'ground_truth' keys
    """
    grpo_data = []
    
    for example in train_data:
        # Extract just the prompt part (before the solution)
        # The train_data from SFT contains full conversations
        # We need to extract just the problem statement
        if 'problem' in example:
            problem = example['problem']
            solution = example.get('solution', '')
            
            # Format the prompt using the template
            prompt = prompt_template.replace("{problem}", problem)
            
            grpo_data.append({
                'prompt': prompt,
                'ground_truth': solution,
                'problem': problem  # Keep for reward computation
            })
    
    return grpo_data


def compute_rewards(prompts, completions, ground_truths):
    """Compute rewards for generated completions using drgrpo_grader.
    
    Args:
        prompts: List of prompt strings
        completions: List of completion strings  
        ground_truths: List of ground truth solutions
        
    Returns:
        List of reward values
    """
    rewards = []
    
    for completion, ground_truth in zip(completions, ground_truths):
        # Prepare problem dict for grader
        problem = {'solution': ground_truth}
        
        # Compute reward using existing grader
        # r1_zero_reward_fn expects a list of problems and responses
        reward_value = r1_zero_reward_fn([problem], [completion])
        
        # Extract scalar reward (r1_zero_reward_fn returns a list)
        if isinstance(reward_value, list):
            reward_value = reward_value[0]
        
        rewards.append(reward_value)
    
    return rewards


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    step: int,
    config: Config
):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        step: Current training step
        config: Configuration object
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(config.checkpointing.output_dir) / f"checkpoint_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    print(f"✓ Checkpoint saved: {checkpoint_dir}")
    return str(checkpoint_dir)


def grpo_training_loop(
    config: Config,
    train_data: list,
    tokenizer: AutoTokenizer,
    eval_queue: Optional[mp.Queue] = None
):
    """Main GRPO training loop on GPU 0.
    
    Args:
        config: Configuration object
        train_data: Training dataset (prompts only)
        tokenizer: Tokenizer
        eval_queue: Queue for sending checkpoint paths to eval worker
    """
    print("\n" + "="*80)
    print("GRPO TRAINING WORKER (GPU 0)")
    print("="*80)
    
    # Set device
    device = config.training.device
    print(f"Device: {device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    
    # Load model from SFT checkpoint
    print(f"\nLoading model from: {config.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.dtype),
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    print(f"✓ Model loaded: {model.config.hidden_size}d, {model.num_parameters():,} params")
    
    # Prepare GRPO dataset
    print("\nPreparing GRPO dataset...")
    with open(config.data.prompt_file, 'r') as f:
        prompt_template = f.read()
    
    grpo_data = prepare_grpo_dataset(train_data, prompt_template)
    print(f"✓ Prepared {len(grpo_data)} training prompts")
    
    # Setup GRPO configuration with custom parameters mapped to TRL's GRPOConfig
    print("\nConfiguring GRPO training...")
    training_args = GRPOConfig(
        # Output and logging
        output_dir=config.checkpointing.output_dir,
        logging_steps=config.logging.log_every,
        save_strategy="steps",
        save_steps=config.training.eval_every,
        report_to="wandb",
        
        # Training configuration (inherits from TrainingArguments)
        max_steps=config.training.max_steps,  # n_grpo_steps = 200
        per_device_train_batch_size=config.grpo.rollout_batch_size,  # 256
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,  # 128 (microbatch=2)
        learning_rate=config.training.learning_rate,  # 1e-5
        weight_decay=config.training.weight_decay,  # 0.0
        adam_beta1=config.training.betas[0],  # 0.9
        adam_beta2=config.training.betas[1],  # 0.95
        bf16=True,
        max_grad_norm=1.0,
        
        # GRPO-specific parameters
        num_generation=config.grpo.group_size,  # 8 (completions per prompt)
        temperature=config.grpo.sampling_temperature,  # 1.0
        max_new_tokens=config.grpo.sampling_max_tokens,  # 1024
        min_new_tokens=config.grpo.sampling_min_tokens,  # 4 (disallow empty)
        
        # vLLM configuration for efficient generation
        use_vllm=True,  # Enable vLLM for fast generation
        vllm_mode="colocate",  # Run vLLM within trainer process
        vllm_gpu_memory_utilization=config.grpo.gpu_memory_utilization,  # 0.85
    )
    
    # Create custom reward function that uses drgrpo_grader
    def reward_fn(samples):
        """Reward function for GRPO trainer.
        
        Args:
            samples: List of dicts with 'prompt' and 'completion' keys
            
        Returns:
            List of reward values
        """
        prompts = [s['prompt'] for s in samples]
        completions = [s['completion'] for s in samples]
        
        # Get ground truths from the dataset
        # This assumes samples maintain order with grpo_data
        ground_truths = [grpo_data[i % len(grpo_data)]['ground_truth'] 
                        for i in range(len(samples))]
        
        return compute_rewards(prompts, completions, ground_truths)
    
    # Initialize GRPO trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=grpo_data,
        reward_fn=reward_fn,
    )
    
    print("✓ GRPO trainer initialized")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Rollout batch size: {config.grpo.rollout_batch_size}")
    print(f"  - Group size (generations/prompt): {config.grpo.group_size}")
    print(f"  - Gradient accumulation steps: {config.grpo.gradient_accumulation_steps}")
    print(f"  - Max steps: {config.training.max_steps}")
    print(f"  - GPU memory utilization: {config.grpo.gpu_memory_utilization}")
    
    # Training loop with checkpoint saving
    print("\n" + "="*80)
    print("STARTING GRPO TRAINING")
    print("="*80)
    
    try:
        # Train the model
        trainer.train()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n⚠ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Save final checkpoint
        final_checkpoint = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            step=trainer.state.global_step,
            config=config
        )
        
        # Send to eval queue
        if eval_queue is not None and not eval_queue.full():
            eval_queue.put(final_checkpoint)
            print(f"✓ Final checkpoint sent to evaluation: {final_checkpoint}")
