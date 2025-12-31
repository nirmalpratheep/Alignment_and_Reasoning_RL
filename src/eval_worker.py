"""Evaluation worker for GPU 1 - continuous validation using vLLM."""
import torch
import wandb
import multiprocessing as mp
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from src.config_loader import Config
from src.vllm_utils import init_vllm, create_sampling_params, load_policy_into_vllm_instance, generate_with_vllm
from utils.drgrpo_grader import r1_zero_reward_fn


def compute_eval_loss(checkpoint_path: str, val_data: list, tokenizer, num_eval_samples: int, model=None) -> tuple:
    """Compute actual cross-entropy eval loss using transformers.
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_data: Validation dataset
        tokenizer: Tokenizer
        num_eval_samples: Number of samples to evaluate
        model: Optional existing model to reuse (will reload weights)
        
    Returns:
        Tuple of (avg_loss, model) - returns model for reuse
    """
    import torch
    from transformers import AutoModelForCausalLM
    
    device = "cuda:0"  # Use current GPU for loss computation
    
    # Load or reload weights into model
    if model is None:
        # First time: create model
        print(f"  Loading model structure for loss computation...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Subsequent times: just reload weights
        print(f"  Reloading checkpoint weights...")
        from pathlib import Path
        checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)  # strict=False for tied weights
        else:
            # Try safetensors
            from safetensors.torch import load_file
            checkpoint_file = Path(checkpoint_path) / "model.safetensors"
            state_dict = load_file(str(checkpoint_file))
            model.load_state_dict(state_dict, strict=False)  # strict=False for tied weights
    
    model.eval()
    
    # Prepare data for loss computation
    total_loss = 0.0
    num_samples = 0
    batch_size = 4  # Small batch to fit in memory alongside vLLM
    
    # Limit to num_eval_samples
    eval_subset = val_data[:num_eval_samples]
    
    with torch.no_grad():
        for i in range(0, len(eval_subset), batch_size):
            batch = eval_subset[i:i+batch_size]
            
            # Tokenize batch - combine prompt and response
            # SFT data has 'prompt' and 'response' keys
            texts = [item['prompt'] + item['response'] for item in batch]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            
            # Compute loss
            # For causal LM, labels are the same as input_ids (shifted internally)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * len(batch)
            num_samples += len(batch)
    
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss, model  # Return model for reuse


def evaluate_checkpoint(
    llm,
    val_data: list,
    sampling_params,
    config: Config,
    tokenizer,
    eval_step: int,
    prompt_template: str
) -> dict:
    """Evaluate current checkpoint on validation set with detailed logging and analysis.
    
    Args:
        llm: vLLM instance with loaded checkpoint
        val_data: Validation dataset
        sampling_params: Sampling parameters for generation
        config: Configuration object
        tokenizer: Tokenizer for computing metrics
        eval_step: Current evaluation step
        prompt_template: Prompt template for formatting prompts
        
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
    
    # Prepare prompts from problem (same as evaluate_model.py)
    print(f"  Preparing prompts for {num_eval} examples...")
    prompts = []
    for example in eval_samples:
        # Format prompt on-the-fly from problem using template (same as evaluate_model.py)
        formatted_prompt = prompt_template.replace('{question}', example['problem'])
        prompts.append(formatted_prompt)
    
    # Generate responses (same as evaluate_model.py - using vLLM directly)
    print(f"  Generating {num_eval} responses...")
    responses = generate_with_vllm(llm, prompts, sampling_params)
    
    # Grade responses and log details
    correct = 0
    format_correct = 0
    all_results = []
    
    for i, (ex, prompt, response) in enumerate(zip(eval_samples, prompts, responses)):
        ground_truth = ex.get('solution', '')
        problem = ex.get('problem', '')
        solution = ex.get('solution', '')
        
        # Use grader to evaluate (same as evaluate_model.py)
        reward_dict = r1_zero_reward_fn(
            response=response,
            ground_truth=ground_truth,
            fast=True
        )
        
        # Compute additional metrics
        token_entropy = compute_token_entropy(response, tokenizer)
        response_length = compute_response_length(response, tokenizer)
        
        # Create result entry (same structure as evaluate_model.py)
        result = {
            'prompt': prompt,  # Use the formatted prompt from prompts list
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
            prompt=prompt,  # Use the formatted prompt from prompts list
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
    seed: int = 42,
    result_queue: mp.Queue = None,
    wandb_run_info: dict = None
) -> None:
    """Evaluation worker process on GPU 1.
    
    Continuously watches queue for checkpoint paths and reloads vLLM
    from each checkpoint to run validation.
    
    Args:
        queue: Multiprocessing queue for receiving checkpoint paths
        config: Configuration object
        val_data: Validation dataset
        seed: Random seed
        result_queue: Optional queue to send final metrics back to main process
        wandb_run_info: Optional dict with wandb run info (name, id, project, entity) to log to same run
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    
    print("="*80)
    print("STARTING EVALUATION WORKER (GPU 1)")
    print("="*80)
    
    # Set device - MUST set CUDA_VISIBLE_DEVICES for vLLM
    device = config.evaluation.device
    
    # Extract GPU index and set environment variable
    import os
    if "cuda:" in device:
        gpu_idx = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        print(f"✓ Set CUDA_VISIBLE_DEVICES={gpu_idx} for vLLM")
    
    torch.cuda.set_device(0)  # Now GPU 1 appears as GPU 0 to this process
    
    # Load tokenizer for metric computation
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompt template (same as evaluate_model.py)
    import os
    prompt_template_path = config.data.prompt_file
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    print(f"✓ Loaded prompt template from {prompt_template_path}")
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=config.evaluation.temperature,
        top_p=config.evaluation.top_p,
        max_tokens=config.evaluation.max_tokens,
        stop_sequences=config.generation.stop_sequences,
        include_stop_str=config.generation.include_stop_str
    )
    
    # Initialize W&B in this process (required for multiprocessing)
    # If wandb_run_info is provided, try to resume the same run; if that fails, create a new run for eval
    wandb_initialized = False
    if wandb_run_info:
        print(f"Initializing wandb to resume run: {wandb_run_info.get('name')} (ID: {wandb_run_info.get('id')})")
        # Try to resume the training run
        try:
            wandb.init(
                project=wandb_run_info.get("project", config.logging.wandb_project),
                entity=wandb_run_info.get("entity", config.logging.wandb_entity),
                name=wandb_run_info.get("name"),
                id=wandb_run_info.get("id"),  # Use the stored run ID
                resume="allow",  # Resume the existing run
                reinit=True,
                settings=wandb.Settings(init_timeout=120)  # 2 minute timeout
            )
            # Verify wandb is actually initialized
            if wandb.run is not None:
                wandb_initialized = True
                print(f"✓ Wandb initialized successfully - resuming training run")
                print(f"  Run ID: {wandb.run.id}, Run name: {wandb.run.name}")
            else:
                print(f"⚠ Wandb.init() returned but wandb.run is None")
                raise Exception("wandb.run is None after init")
        except Exception as e:
            print(f"⚠ Warning: Failed to resume training run: {e}")
            print("  Creating new wandb run for evaluation...")
            # Fallback: create a new run for evaluation
            try:
                eval_run_name = f"{wandb_run_info.get('name')}-eval"
                wandb.init(
                    project=wandb_run_info.get("project", config.logging.wandb_project),
                    entity=wandb_run_info.get("entity", config.logging.wandb_entity),
                    name=eval_run_name,
                    group="dual-gpu-training",
                    job_type="evaluation",
                    reinit=True,
                    settings=wandb.Settings(init_timeout=120)
                )
                if wandb.run is not None:
                    wandb_initialized = True
                    print(f"✓ Wandb initialized with new eval run: {eval_run_name}")
                    print(f"  Run ID: {wandb.run.id}")
                else:
                    print(f"⚠ Failed to create new eval run - continuing without wandb")
            except Exception as e2:
                print(f"⚠ Warning: Failed to create new eval run: {e2}")
                print("  Continuing without wandb logging")
    else:
        # Fallback: create separate run if info not provided
        try:
            wandb.init(
                project=config.logging.wandb_project,
                name=f"eval-worker-{wandb.util.generate_id()}",
                group="dual-gpu-training",
                job_type="evaluation",
                reinit=True,
                settings=wandb.Settings(init_timeout=3600)  # 1 hour timeout (effectively no timeout)
            )
            wandb_initialized = True
            print("✓ Wandb initialized (separate run)")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize wandb: {e}")
            print("  Continuing without wandb logging...")
    
    if wandb_initialized:
        print("✓ Evaluation worker ready (wandb initialized)")
    else:
        print("⚠ Evaluation worker ready (wandb NOT initialized - will continue without logging)")
    print("="*80)
    
    eval_step = 0
    llm = None  # vLLM instance will be created per checkpoint
    loss_model = None  # Transformers model for loss computation (reused across checkpoints)
    
    # Main evaluation loop
    while True:
        # Wait for checkpoint path from queue (blocking - will wait until checkpoint arrives)
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
            metrics = evaluate_checkpoint(llm, val_data, sampling_params, config, tokenizer, eval_step, prompt_template)
            
            # Clean up vLLM before computing loss to avoid OOM
            print(f"Unloading vLLM to free memory for loss computation...")
            del llm
            llm = None
            torch.cuda.empty_cache()
            
            # Compute eval loss using transformers (reuse model)
            print(f"Computing eval loss...")
            eval_loss, loss_model = compute_eval_loss(
                checkpoint_path, val_data, tokenizer,
                config.evaluation.num_eval_samples, loss_model
            )
            print(f"✓ Eval loss computed: {eval_loss:.4f}")
            
            # Reload vLLM for next checkpoint
            print(f"Reloading vLLM for next checkpoint...")
            llm = LLM(
                model=checkpoint_path,
                dtype="float16",
                seed=seed,
                gpu_memory_utilization=0.7,
                tensor_parallel_size=1,
                enforce_eager=True,
            )
            print(f"✓ vLLM reloaded")
            
            # Compute categorization breakdown
            total = metrics['num_evaluated']
            correct_count = metrics['correct']
            format_correct_count = metrics['format_correct']
            
            # Categorization breakdown:
            # category_1 (correct): format=1, answer=1 = correct_count
            # category_2 (wrong answer): format=1, answer=0 = format_correct - correct
            # category_3 (format failure): format=0 = total - format_correct
            category_1_count = correct_count
            category_2_count = format_correct_count - correct_count
            category_3_count = total - format_correct_count
            
            # Log to W&B (same run as training) - only if wandb is initialized
            try:
                if wandb_initialized and wandb.run is not None:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/accuracy": metrics['accuracy'],
                        "eval/format_accuracy": metrics['format_accuracy'],
                        "eval/num_correct": metrics['correct'],
                        "eval/num_format_correct": metrics['format_correct'],
                        "eval/num_evaluated": metrics['num_evaluated'],
                        "eval/avg_response_length": metrics['avg_response_length'],
                        "eval/avg_response_length_correct": metrics['avg_response_length_correct'],
                        "eval/avg_response_length_incorrect": metrics['avg_response_length_incorrect'],
                        "eval/avg_token_entropy": metrics['avg_token_entropy'],
                        # Categorization metrics
                        "eval/category_1_correct_count": category_1_count,
                        "eval/category_1_correct_pct": (category_1_count / total * 100) if total > 0 else 0.0,
                        "eval/category_2_wrong_answer_count": category_2_count,
                        "eval/category_2_wrong_answer_pct": (category_2_count / total * 100) if total > 0 else 0.0,
                        "eval/category_3_format_failure_count": category_3_count,
                        "eval/category_3_format_failure_pct": (category_3_count / total * 100) if total > 0 else 0.0,
                        "eval_step": eval_step,
                    })
                else:
                    print("  (Skipping wandb logging - not initialized)")
                    print(f"    Debug: wandb_initialized={wandb_initialized}, wandb.run={wandb.run}")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to log to wandb: {e}")
                print("  Continuing without wandb logging...")
                import traceback
                traceback.print_exc()
            
            print(f"✓ Evaluation complete:")
            print(f"  - Accuracy: {metrics['accuracy']:.3f}")
            print(f"  - Format Accuracy: {metrics['format_accuracy']:.3f}")
            print(f"  - Correct: {metrics['correct']}/{metrics['num_evaluated']}")
            print(f"  - Avg Response Length: {metrics['avg_response_length']:.1f}")
            print(f"  - Avg Token Entropy: {metrics['avg_token_entropy']:.3f}")
            
            # Send metrics to result queue if provided (for hyperparameter optimization)
            if result_queue is not None:
                result_queue.put(metrics)
            
        except Exception as e:
            print(f"⚠ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Send error signal to result queue
            if result_queue is not None:
                result_queue.put({"accuracy": 0.0, "error": str(e)})
        
        eval_step += 1
    
    # Cleanup
    if llm is not None:
        del llm
        torch.cuda.empty_cache()
    
    print("="*80)
    print("EVALUATION WORKER STOPPED")
    print("="*80)
