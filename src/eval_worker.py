"""Evaluation worker for GPU 1 - continuous validation using vLLM."""
import torch
import wandb
import multiprocessing as mp
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from src.config_loader import Config
from src.vllm_utils import init_vllm, create_sampling_params, load_policy_into_vllm_instance, generate_with_vllm
from utils.drgrpo_grader import r1_zero_reward_fn


def evaluate_checkpoint(
    llm,
    val_data: list,
    sampling_params,
    config: Config,
    tokenizer,
    eval_step: int
) -> dict:
    """Evaluate current checkpoint on validation set with detailed logging and analysis.
    
    Args:
        llm: vLLM instance with loaded checkpoint
        val_data: Validation dataset
        sampling_params: Sampling parameters for generation
        config: Configuration object
        tokenizer: Tokenizer for computing metrics
        eval_step: Current evaluation step
        
    Returns:
        Dictionary with evaluation metrics
    """
    from src.logging_utils import DetailedEvaluationLogger, compute_token_entropy, compute_response_length
    from src.analysis_utils import (
        categorize_results, 
        analyze_format_failures,
        generate_summary_report,
        save_analysis_report,
        print_analysis_summary
    )
    
    num_eval = min(config.evaluation.num_eval_samples, len(val_data))
    eval_samples = val_data[:num_eval]
    
    # Initialize detailed logger
    logger = DetailedEvaluationLogger(
        log_dir="results/eval_logs",
        eval_step=eval_step
    )
    
    # Generate responses
    print(f"  Generating {num_eval} responses...")
    prompts = [ex['prompt'] for ex in eval_samples]
    responses = generate_with_vllm(llm, prompts, sampling_params)
    
    # Grade responses and log details
    correct = 0
    format_correct = 0
    all_results = []
    
    for i, (ex, response) in enumerate(zip(eval_samples, responses)):
        ground_truth = ex.get('solution', '')
        
        # Use grader to evaluate
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
        
        # Log detailed information
        logger.log_test_case(
            prompt=ex['prompt'],
            response=response,
            ground_truth=ground_truth,
            format_reward=reward_dict['format_reward'],
            answer_reward=reward_dict['answer_reward'],
            total_reward=reward_dict['reward'],
            token_entropy=token_entropy,
            response_length=response_length
        )
        
        if reward_dict['format_reward'] == 1.0:
            format_correct += 1
        
        if reward_dict['reward'] == 1.0:
            correct += 1
    
    # Save detailed logs and get summary
    summary_stats = logger.save()
    
    # Categorize results
    print("  Categorizing results...")
    category_1, category_2, category_3 = categorize_results(all_results)
    
    # Analyze format failures
    format_issues = analyze_format_failures(category_3)
    
    # Compute metrics
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
    
    # Generate comprehensive summary report
    summary_report = generate_summary_report(
        category_1, category_2, category_3,
        format_issues, metrics, eval_step
    )
    
    # Save analysis report
    save_analysis_report(
        summary_report, category_1, category_2, category_3,
        "results/analysis", eval_step
    )
    
    # Print analysis summary
    print_analysis_summary(summary_report, category_3, category_2)
    
    return metrics


def eval_worker(
    queue: mp.Queue,
    config: Config,
    val_data: list,
    seed: int = 42
) -> None:
    """Evaluation worker process on GPU 1.
    
    Continuously watches queue for checkpoint paths and reloads vLLM
    from each checkpoint to run validation.
    
    Args:
        queue: Multiprocessing queue for receiving checkpoint paths
        config: Configuration object
        val_data: Validation dataset
        seed: Random seed
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    
    print("="*80)
    print("STARTING EVALUATION WORKER (GPU 1)")
    print("="*80)
    
    # Set device
    device = config.evaluation.device
    torch.cuda.set_device(device)
    
    # Load tokenizer for metric computation
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=config.evaluation.temperature,
        top_p=config.evaluation.top_p,
        max_tokens=config.evaluation.max_tokens,
        stop_sequences=config.generation.stop_sequences,
        include_stop_str=config.generation.include_stop_str
    )
    
    print("✓ Evaluation worker ready")
    print("="*80)
    
    eval_step = 0
    llm = None  # vLLM instance will be created per checkpoint
    
    # Main evaluation loop
    while True:
        # Wait for checkpoint path from queue
        print(f"\n[Eval Step {eval_step}] Waiting for checkpoint...")
        checkpoint_path = queue.get()
        
        # Check for shutdown signal
        if checkpoint_path is None:
            print("Received shutdown signal. Exiting eval worker.")
            break
        
        print(f"[Eval Step {eval_step}] Loading vLLM from checkpoint: {checkpoint_path}")
        
        try:
            # Clean up previous vLLM instance to free memory
            if llm is not None:
                del llm
                torch.cuda.empty_cache()
            
            # Reload vLLM directly from checkpoint (correct approach for vLLM 0.4+)
            llm = LLM(
                model=checkpoint_path,  # Load directly from checkpoint
                dtype="float16",
                seed=seed,
                gpu_memory_utilization=0.7,
                tensor_parallel_size=1,
                enforce_eager=True,
            )
            
            print(f"✓ vLLM loaded from checkpoint")
            
            # Run evaluation with detailed logging
            print(f"Running evaluation on {config.evaluation.num_eval_samples} samples...")
            metrics = evaluate_checkpoint(llm, val_data, sampling_params, config, tokenizer, eval_step)
            
            # Log to W&B
            wandb.log({
                "eval/accuracy": metrics['accuracy'],
                "eval/format_accuracy": metrics['format_accuracy'],
                "eval/num_correct": metrics['correct'],
                "eval/num_format_correct": metrics['format_correct'],
                "eval/avg_response_length": metrics['avg_response_length'],
                "eval/avg_response_length_correct": metrics['avg_response_length_correct'],
                "eval/avg_response_length_incorrect": metrics['avg_response_length_incorrect'],
                "eval/avg_token_entropy": metrics['avg_token_entropy'],
                "eval_step": eval_step,
            })
            
            print(f"✓ Evaluation complete:")
            print(f"  - Accuracy: {metrics['accuracy']:.3f}")
            print(f"  - Format Accuracy: {metrics['format_accuracy']:.3f}")
            print(f"  - Correct: {metrics['correct']}/{metrics['num_evaluated']}")
            print(f"  - Avg Response Length: {metrics['avg_response_length']:.1f}")
            print(f"  - Avg Token Entropy: {metrics['avg_token_entropy']:.3f}")
            
        except Exception as e:
            print(f"⚠ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        eval_step += 1
    
    # Cleanup
    if llm is not None:
        del llm
        torch.cuda.empty_cache()
    
    print("="*80)
    print("EVALUATION WORKER STOPPED")
    print("="*80)
