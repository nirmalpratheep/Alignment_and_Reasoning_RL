"""vLLM-based evaluation for GRPO training."""
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

import torch
from vllm import LLM, SamplingParams

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from step3_GRPO.train.rewards_math import compute_math_rewards


def load_gsm8k_testset(num_samples: int = 1000) -> List[Dict]:
    """Load GSM8K test set.
    
    Args:
        num_samples: Number of test samples
        
    Returns:
        List of test examples with 'problem' and 'solution'
    """
    from datasets import load_dataset
    
    dataset = load_dataset("gsm8k", "main", split="test")
    
    examples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        examples.append({
            'problem': item['question'],
            'solution': item['answer'],
        })
    
    return examples


def run_vllm_evaluation(
    checkpoint_path: str,
    test_data: List[Dict],
    prompt_template: str,
    batch_size: int = 320,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    gpu_id: int = 1,
) -> Dict[str, Any]:
    """Run evaluation using vLLM.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data: List of test examples
        prompt_template: Prompt template string
        batch_size: Batch size for vLLM
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        gpu_id: GPU ID to use
        
    Returns:
        Dictionary with evaluation results
    """
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"EVALUATION (GPU {gpu_id})")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize vLLM
    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )
    
    # Prepare prompts
    prompts = []
    for example in test_data:
        prompt = prompt_template.replace("{problem}", example['problem'])
        prompts.append(prompt)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    
    # Generate
    print("Generating predictions...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract completions
    completions = [output.outputs[0].text for output in outputs]
    
    # Compute rewards (accuracy)
    ground_truths = [ex['solution'] for ex in test_data]
    rewards = compute_math_rewards(completions, ground_truths)
    
    # Compute metrics
    accuracy = sum(rewards) / len(rewards)
    avg_length = sum(len(c.split()) for c in completions) / len(completions)
    valid_format_rate = sum(1 for c in completions if len(c.strip()) > 0) / len(completions)
    
    eval_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg length: {avg_length:.1f} words")
    print(f"  Valid format: {valid_format_rate:.2%}")
    print(f"  Time: {eval_time:.1f}s")
    print(f"{'='*80}\n")
    
    # Collect sample outputs
    samples = []
    for i in range(min(10, len(test_data))):
        samples.append({
            'problem': test_data[i]['problem'],
            'pred': completions[i][:200],  # Truncate for display
            'gt': ground_truths[i][:200],
            'correct': bool(rewards[i] == 1.0),
        })
    
    results = {
        'accuracy': accuracy,
        'gsm8k_acc': accuracy,  # For now, assuming GSM8K
        'math_acc': 0.0,  # Placeholder
        'overall_acc': accuracy,
        'level_accs': [0.0] * 5,  # Placeholder for MATH levels
        'avg_length': avg_length,
        'valid_format_rate': valid_format_rate,
        'eval_time': eval_time,
        'samples': samples,
    }
    
    return results
