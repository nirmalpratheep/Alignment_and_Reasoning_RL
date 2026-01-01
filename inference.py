#!/usr/bin/env python3
"""
Standalone inference script for evaluating a checkpoint on the test dataset.
Usage:
    python inference.py --checkpoint <checkpoint_path> [--num_samples N] [--output_dir <dir>]
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.dataset_loader import MathDatasetLoader
from utils.drgrpo_grader import r1_zero_reward_fn
from src.logging_utils import DetailedEvaluationLogger, compute_token_entropy, compute_response_length
from src.analysis_utils import (
    categorize_results,
    analyze_format_failures,
    generate_summary_report,
    save_analysis_report,
    print_analysis_summary
)
from src.vllm_utils import generate_with_vllm


def main():
    parser = argparse.ArgumentParser(description="Run inference on checkpoint using test dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of test samples to evaluate (default: 5000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/inference",
        help="Output directory for results (default: results/inference)"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/rl_zero.prompt",
        help="Path to prompt template file (default: prompts/rl_zero.prompt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("INFERENCE ON CHECKPOINT")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print("="*80)
    
    # Load prompt template
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        print(f"Error: Prompt file does not exist: {prompt_file}")
        sys.exit(1)
    
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    # Load test dataset
    print("\nLoading test dataset...")
    loader = MathDatasetLoader()
    loader.load_all_subsets()
    test_examples = loader.collect_test_examples()
    print(f"✓ Loaded {len(test_examples)} test examples")
    
    # Limit to num_samples
    eval_samples = test_examples[:args.num_samples]
    print(f"✓ Evaluating on {len(eval_samples)} samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    # Try to load from checkpoint, fallback to base model name if needed
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from checkpoint, trying base model...")
        # Try to infer base model from config.json if exists
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                base_model = config.get("_name_or_path", "Qwen/Qwen2.5-Math-1.5B")
        else:
            base_model = "Qwen/Qwen2.5-Math-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Load model with vLLM
    print(f"\nLoading model from checkpoint with vLLM...")
    print(f"Device: {args.device}, GPU memory utilization: {args.gpu_memory_utilization}")
    
    llm = LLM(
        model=str(checkpoint_path),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    print("✓ Model loaded")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # Initialize evaluation logger
    eval_step = 0  # Use 0 for standalone inference
    logger = DetailedEvaluationLogger(
        log_dir=str(output_dir / "eval_logs"),
        eval_step=eval_step
    )
    
    # Prepare prompts
    print(f"\nPreparing prompts for {len(eval_samples)} examples...")
    prompts = []
    for example in eval_samples:
        formatted_prompt = prompt_template.replace('{question}', example['problem'])
        prompts.append(formatted_prompt)
    
    # Generate responses
    print(f"Generating responses...")
    responses = generate_with_vllm(llm, prompts, sampling_params)
    print(f"✓ Generated {len(responses)} responses")
    
    # Grade responses and collect results
    print("\nGrading responses...")
    correct = 0
    format_correct = 0
    all_results = []
    
    for i, (example, prompt, response) in enumerate(zip(eval_samples, prompts, responses)):
        ground_truth = example.get('solution', '')
        problem = example.get('problem', '')
        
        # Grade response
        reward_dict = r1_zero_reward_fn(
            response=response,
            ground_truth=ground_truth,
            fast=True
        )
        
        # Compute additional metrics
        token_entropy = compute_token_entropy(response, tokenizer)
        response_length = compute_response_length(response, tokenizer)
        
        # Create result entry
        result = {
            'prompt': prompt,
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
            },
            'problem': problem,
            'solution': ground_truth
        }
        all_results.append(result)
        
        # Log detailed information
        logger.log_test_case(
            prompt=prompt,
            response=response,
            ground_truth=ground_truth,
            format_reward=reward_dict['format_reward'],
            answer_reward=reward_dict['answer_reward'],
            total_reward=reward_dict['reward'],
            token_entropy=token_entropy,
            response_length=response_length,
            problem=problem,
            solution=ground_truth
        )
        
        if reward_dict['format_reward'] == 1.0:
            format_correct += 1
        if reward_dict['reward'] == 1.0:
            correct += 1
    
    # Save detailed logs
    summary_stats = logger.save()
    
    # Categorize results
    print("\nCategorizing results...")
    category_1, category_2, category_3 = categorize_results(all_results)
    
    # Analyze format failures
    format_issues = analyze_format_failures(category_3)
    
    # Compute metrics
    num_eval = len(eval_samples)
    accuracy = correct / num_eval
    format_accuracy = format_correct / num_eval
    
    metrics = {
        'accuracy': accuracy,
        'format_accuracy': format_accuracy,
        'num_evaluated': num_eval,
        'correct': correct,
        'format_correct': format_correct,
        'avg_response_length': summary_stats['avg_response_length'],
        'avg_response_length_correct': summary_stats['avg_response_length_correct'],
        'avg_response_length_incorrect': summary_stats['avg_response_length_incorrect'],
        'avg_token_entropy': summary_stats['avg_token_entropy']
    }
    
    # Generate summary report
    summary_report = generate_summary_report(
        category_1, category_2, category_3,
        format_issues, metrics, eval_step
    )
    
    # Save analysis report
    analysis_dir = output_dir / "analysis"
    save_analysis_report(
        summary_report, category_1, category_2, category_3,
        str(analysis_dir), eval_step
    )
    
    # Print analysis summary
    print_analysis_summary(summary_report, category_3, category_2)
    
    # Save results JSON
    results_file = output_dir / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print final summary
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Format Accuracy: {format_accuracy:.4f} ({format_accuracy*100:.2f}%)")
    print(f"Correct: {correct}/{num_eval}")
    print(f"Format Correct: {format_correct}/{num_eval}")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

