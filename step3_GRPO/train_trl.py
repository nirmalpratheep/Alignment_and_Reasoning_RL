"""TRL-based GRPO Training with vLLM for Math Reasoning.

This is a simplified implementation using HuggingFace TRL's GRPOTrainer
with vLLM colocate mode for high-throughput generation.

Usage:
    accelerate launch step3_GRPO/train_trl.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Use the rewards_math module for reward computation
from step3_GRPO.train.rewards_math import compute_math_rewards
from utils.drgrpo_grader import r1_zero_reward_fn  # For evaluation


def load_config(config_path: str = "config/grpo_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def math_reward_function(completions: list, solution: list, **kwargs) -> list:
    """TRL-compatible reward function for math problems.
    
    Uses the compute_math_rewards function from rewards_math.py
    
    Args:
        completions: List of model-generated completions
        solution: List of ground truth solutions (from dataset column)
        **kwargs: Additional columns from dataset
        
    Returns:
        List of reward floats
    """
    return compute_math_rewards(completions, solution)


def format_reward_function(completions: list, **kwargs) -> list:
    """Bonus reward for correct format (thinking + answer tags)."""
    rewards = []
    for completion in completions:
        result = r1_zero_reward_fn(completion, "dummy", fast=True)
        rewards.append(result['format_reward'])
    return rewards


def prepare_dataset(prompt_template: str, split: str = "train", num_samples: int = None):
    """Load and prepare NuminaMath dataset with prompt formatting.
    
    Args:
        prompt_template: Template with {question} placeholder
        split: Dataset split to load
        num_samples: Optional limit on number of samples
        
    Returns:
        HuggingFace Dataset ready for GRPOTrainer
    """
    # Load NuminaMath dataset
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split=split)
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format dataset for TRL GRPOTrainer
    def format_example(example):
        # Create prompt from template
        prompt = prompt_template.replace("{question}", example["problem"])
        return {
            "prompt": prompt,
            "solution": example["solution"],  # Ground truth for reward function
        }
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return formatted_dataset


def main():
    # Load configuration
    config = load_config("config/grpo_config.yaml")
    
    # Load prompt template
    prompt_template = load_prompt_template(config['data']['prompt_file'])
    
    print("=" * 70)
    print("TRL GRPO Training with vLLM")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Group Size (K): {config['grpo']['group_size']}")
    print(f"Temperature: {config['grpo']['temperature']}")
    print(f"Max Steps: {config['training']['max_steps']}")
    print("=" * 70)
    
    # Load tokenizer and model
    model_name = config['model']['name']
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if config['model']['dtype'] == 'bfloat16' else torch.float32,
        trust_remote_code=config['model'].get('trust_remote_code', True),
    )
    
    # Prepare datasets
    print("\nLoading datasets...")
    train_dataset = prepare_dataset(
        prompt_template,
        split="train",
        num_samples=config['data'].get('num_train_samples')
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    
    # Create TRL GRPOConfig with memory optimizations
    # GRPO needs: model + vLLM + logprobs computation = ~3x model memory
    training_args = GRPOConfig(
        output_dir=config['checkpointing']['output_dir'],
        
        # Training hyperparameters
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        max_steps=config['training']['max_steps'],
        
        # Batch sizes - OPTIMIZED (23GB/80GB used, plenty of headroom)
        per_device_train_batch_size=4,  # Increased from 2
        gradient_accumulation_steps=16,  # Adjusted to keep effective batch similar
        
        # GRPO specific
        num_generations=config['grpo']['group_size'],  # K completions per prompt
        temperature=config['grpo']['temperature'],
        max_completion_length=config['grpo']['max_tokens'],
        
        # vLLM settings - OPTIMIZED for more throughput
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.55,  # Increased (more KV cache = faster generation)
        vllm_enable_sleep_mode=True,  # Offload vLLM weights during training
        
        # Logging
        logging_steps=config['logging']['log_every_steps'],
        report_to="wandb",
        run_name="grpo-trl-vllm",
        
        # Checkpointing
        save_steps=config['training']['eval_every'],
        save_total_limit=config['checkpointing'].get('save_total_limit', 3),
        
        # Precision
        bf16=config['model']['dtype'] == 'bfloat16',
        
        # Speed optimizations (we have plenty of memory)
        gradient_checkpointing=False,  # Faster (no recompute)
        max_prompt_length=512,
    )
    
    # Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[math_reward_function],  # Custom reward function
    )
    
    print("\n" + "=" * 70)
    print("Starting GRPO Training")
    print("=" * 70)
    
    # Train
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config['checkpointing']['output_dir'])
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    
    # =========================================
    # Final Evaluation on Validation Dataset
    # =========================================
    print("\n" + "=" * 70)
    print("Running Final Evaluation on Validation Set")
    print("=" * 70)
    
    run_final_evaluation(
        model_path=config['checkpointing']['output_dir'],
        prompt_template=prompt_template,
        config=config
    )


def run_final_evaluation(model_path: str, prompt_template: str, config: dict):
    """Run comprehensive evaluation on FULL validation dataset matching SFT format.
    
    Uses same metrics and output format as SFT evaluation:
    - Overall accuracy
    - Format accuracy  
    - 3-category breakdown: correct / wrong answer / format failure
    - Source-based category breakdown
    """
    import wandb
    import json
    from datetime import datetime
    from collections import defaultdict
    from vllm import LLM, SamplingParams
    
    print(f"\nLoading model from: {model_path}")
    
    # Load FULL validation dataset (matching SFT eval)
    print("Loading full validation dataset...")
    val_dataset = load_dataset("AI-MO/NuminaMath-CoT", split="test")
    # If no test split, use a portion of train
    if len(val_dataset) == 0:
        val_dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train[-5000:]")
    
    num_eval = config['evaluation'].get('num_samples', len(val_dataset))
    if num_eval:
        val_dataset = val_dataset.select(range(min(num_eval, len(val_dataset))))
    
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Prepare prompts
    prompts = []
    ground_truths = []
    sources = []
    problems = []
    
    for example in val_dataset:
        prompt = prompt_template.replace("{question}", example["problem"])
        prompts.append(prompt)
        ground_truths.append(example["solution"])
        sources.append(example.get("source", "unknown"))
        problems.append(example["problem"])
    
    # Initialize vLLM for fast inference
    print("\nInitializing vLLM for evaluation...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=config['grpo'].get('max_model_len', 2048),
    )
    
    sampling_params = SamplingParams(
        temperature=config['evaluation'].get('temperature', 0.0),  # Greedy for evaluation
        max_tokens=config['evaluation'].get('max_tokens', 1024),
        top_p=config['evaluation'].get('top_p', 1.0),
    )
    
    # Generate completions
    print(f"Generating {len(prompts)} completions...")
    outputs = llm.generate(prompts, sampling_params)
    completions = [output.outputs[0].text for output in outputs]
    
    # Compute rewards/accuracy with detailed breakdown (matching SFT format)
    print("Computing accuracy...")
    all_results = []
    source_results = defaultdict(lambda: {"correct": 0, "format_ok": 0, "total": 0})
    
    # 3-category breakdown (same as SFT)
    category_1_correct = []  # format=1, answer=1
    category_2_wrong_answer = []  # format=1, answer=0
    category_3_format_failure = []  # format=0
    
    for i, (prompt, completion, gt, source, problem) in enumerate(
        zip(prompts, completions, ground_truths, sources, problems)
    ):
        result = r1_zero_reward_fn(completion, gt, fast=True)
        
        format_ok = result['format_reward'] > 0
        answer_ok = result['answer_reward'] > 0
        is_correct = result['reward'] > 0
        
        result_entry = {
            "idx": i,
            "prompt": prompt,
            "response": completion,
            "ground_truth": gt,
            "problem": problem,
            "source": source,
            "format_reward": result['format_reward'],
            "answer_reward": result['answer_reward'],
            "total_reward": result['reward'],
        }
        all_results.append(result_entry)
        
        # Source-based breakdown
        source_results[source]["total"] += 1
        if format_ok:
            source_results[source]["format_ok"] += 1
        if is_correct:
            source_results[source]["correct"] += 1
        
        # 3-category breakdown (matching SFT)
        if format_ok and answer_ok:
            category_1_correct.append(result_entry)
        elif format_ok and not answer_ok:
            category_2_wrong_answer.append(result_entry)
        else:
            category_3_format_failure.append(result_entry)
    
    # Compute metrics
    total = len(all_results)
    correct = len(category_1_correct)
    format_ok_count = len(category_1_correct) + len(category_2_wrong_answer)
    
    accuracy = correct / total if total > 0 else 0
    format_accuracy = format_ok_count / total if total > 0 else 0
    
    # Print results (matching SFT format)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS (GRPO)")
    print("=" * 70)
    print(f"Total samples evaluated: {total}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Format Accuracy:  {format_accuracy:.4f} ({format_ok_count}/{total})")
    
    print("\n" + "-" * 50)
    print("3-CATEGORY BREAKDOWN (same as SFT)")
    print("-" * 50)
    print(f"Category 1 (Correct):        {len(category_1_correct):5d}  ({len(category_1_correct)/total*100:.1f}%)")
    print(f"Category 2 (Wrong Answer):   {len(category_2_wrong_answer):5d}  ({len(category_2_wrong_answer)/total*100:.1f}%)")
    print(f"Category 3 (Format Failure): {len(category_3_format_failure):5d}  ({len(category_3_format_failure)/total*100:.1f}%)")
    
    print("\n" + "-" * 50)
    print("SOURCE BREAKDOWN")
    print("-" * 50)
    print(f"{'Source':<25} {'Accuracy':<12} {'Format':<12} {'Count':<10}")
    print("-" * 50)
    
    for source, stats in sorted(source_results.items(), key=lambda x: -x[1]["total"]):
        src_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        src_fmt = stats["format_ok"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{source:<25} {src_acc:.4f}       {src_fmt:.4f}       {stats['total']:<10}")
    
    print("=" * 70)
    
    # Save detailed results to file (matching SFT format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['checkpointing']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "model_path": model_path,
        "total_samples": total,
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "category_1_correct": len(category_1_correct),
        "category_2_wrong_answer": len(category_2_wrong_answer),
        "category_3_format_failure": len(category_3_format_failure),
        "source_breakdown": {k: dict(v) for k, v in source_results.items()},
    }
    
    summary_path = output_dir / f"eval_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Log to wandb
    try:
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/format_accuracy": format_accuracy,
            "eval/total_samples": total,
            "eval/num_correct": correct,
            "eval/num_format_correct": format_ok_count,
            "eval/category_1_correct": len(category_1_correct),
            "eval/category_2_wrong_answer": len(category_2_wrong_answer),
            "eval/category_3_format_failure": len(category_3_format_failure),
        })
        
        # Log source breakdown as table
        source_data = [
            [src, stats["correct"], stats["format_ok"], stats["total"],
             stats["correct"]/stats["total"] if stats["total"] > 0 else 0]
            for src, stats in source_results.items()
        ]
        wandb.log({
            "eval/source_breakdown": wandb.Table(
                columns=["Source", "Correct", "Format OK", "Total", "Accuracy"],
                data=source_data
            )
        })
    except Exception as e:
        print(f"Warning: Could not log to wandb: {e}")
    
    # Cleanup vLLM
    del llm
    torch.cuda.empty_cache()
    
    print("\n✓ Evaluation complete!")
    return {
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "category_1": len(category_1_correct),
        "category_2": len(category_2_wrong_answer),
        "category_3": len(category_3_format_failure),
        "source_results": dict(source_results)
    }


if __name__ == "__main__":
    main()

