"""
Hyperparameter Optimization for SFT Training using Optuna
Uses Bayesian Optimization (TPE) instead of grid search for efficient search.

Architecture:
- Persistent evaluation worker runs continuously, processing checkpoints from all trials
- Trials train models and send checkpoints to a shared queue
- Evaluation worker processes checkpoints and sends results back via result manager
"""
import os
import sys
import yaml
import wandb
import optuna
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
import threading
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set spawn method for CUDA compatibility (must be before any CUDA operations)
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Search space bounds
LR_MIN, LR_MAX = 5e-6, 1e-4
BATCH_SIZE_CHOICES = [ 32, 64, 128, 256]
WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX = 0.0, 0.1

# Fixed parameters
FIXED_CONFIG = {
    "max_batches": 50000,  # Very high limit - allow full dataset training
    "eval_every": 1000,     # Evaluate every 1000 batches
    "warmup_steps": 50,
    "num_eval_samples": 500,  # 500 samples for reliable evaluation
}

# Number of optimization trials
N_TRIALS = 30  # More trials for ASHA

# Early stopping config
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 evaluations
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement threshold

# Global shared queues for persistent eval worker
_shared_checkpoint_queue: Optional[mp.Queue] = None
_shared_result_manager: Optional[mp.Manager] = None
_shared_result_dict: Optional[dict] = None
_persistent_eval_process: Optional[mp.Process] = None


def create_config(lr: float, batch_size: int, weight_decay: float, trial_id: int) -> Dict[str, Any]:
    """Create config dict for a trial.
    
    GPU Assignment:
    - Training: cuda:0 (dedicated for training trials)
    - Evaluation: cuda:1 (dedicated for persistent eval worker)
    """
    return {
        "model": {
            "name": "Qwen/Qwen2.5-Math-1.5B",
            "dtype": "bfloat16"
        },
        "training": {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": 1,
            "gradient_accumulation_steps": max(1, 64 // batch_size),
            "warmup_steps": FIXED_CONFIG["warmup_steps"],
            "max_batches": FIXED_CONFIG["max_batches"],
            "eval_every": FIXED_CONFIG["eval_every"],
            "device": "cuda:0"  # GPU 0: Dedicated for training
        },
        "evaluation": {
            "device": "cuda:1",  # GPU 1: Dedicated for evaluation
            "batch_size": 240,
            "num_eval_samples": FIXED_CONFIG["num_eval_samples"],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024
        },
        "generation": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "stop_sequences": ["</answer>"],
            "include_stop_str": True
        },
        "data": {
            "prompt_file": "prompts/rl_zero.prompt",
            "train_val_split": 0.8
        },
        "checkpointing": {
            "output_dir": f"results/optuna/trial_{trial_id}",
            "queue_maxsize": 10,  # Increased for periodic checkpoints
            "temp_dir": f"/tmp/qwen_optuna_{trial_id}"
        },
        "logging": {
            "wandb_project": "math-sft-optuna-asha",
            "wandb_entity": None,
            "log_every": 10
        }
    }


def compute_eval_loss(checkpoint_path: str, val_data: list, tokenizer, eval_config: dict, model=None) -> tuple:
    """Compute actual cross-entropy eval loss using transformers.
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_data: Validation dataset
        tokenizer: Tokenizer
        eval_config: Evaluation config
        model: Optional existing model to reuse (will reload weights)
        
    Returns:
        Tuple of (avg_loss, model) - returns model for reuse
    """
    import torch
    from transformers import AutoModelForCausalLM
    from torch.utils.data import DataLoader
    
    device = "cuda:0"  # Use first available GPU for loss computation
    
    # Load or reload weights into model
    if model is None:
        # First time: create model
        print(f"  Loading model structure...")
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
    num_eval_samples = eval_config["evaluation"]["num_eval_samples"]
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



def persistent_eval_worker(
    checkpoint_queue: mp.Queue,
    result_dict: dict,
    eval_config: Dict[str, Any],
    val_data: list,
    shutdown_event: mp.Event
):
    """Persistent evaluation worker that processes checkpoints from all trials.
    
    Simple approach: recreates vLLM instance from each checkpoint.
    This is more reliable than trying to load weights into existing instances.
    
    Args:
        checkpoint_queue: Queue receiving (trial_id, checkpoint_path) tuples
        result_dict: Shared dictionary to store results: {trial_id: metrics}
        eval_config: Evaluation configuration (same for all trials)
        val_data: Validation dataset
        shutdown_event: Event to signal when to shutdown
    """
    import os
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    
    print("="*80)
    print("STARTING PERSISTENT EVALUATION WORKER (GPU 1)")
    print("="*80)
    
    # Set device for vLLM
    device = eval_config["evaluation"]["device"]
    if "cuda:" in device:
        gpu_idx = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        print(f"✓ Set CUDA_VISIBLE_DEVICES={gpu_idx} for vLLM")
    
    torch.cuda.set_device(0)  # GPU 1 appears as GPU 0 after CUDA_VISIBLE_DEVICES
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(eval_config["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=eval_config["evaluation"]["temperature"],
        top_p=eval_config["evaluation"]["top_p"],
        max_tokens=eval_config["evaluation"]["max_tokens"],
        stop=eval_config["generation"]["stop_sequences"],
        include_stop_str_in_output=eval_config["generation"]["include_stop_str"]
    )
    
    num_eval_samples = min(eval_config["evaluation"]["num_eval_samples"], len(val_data))
    
    # Pre-load vLLM with base model so it's ready for first checkpoint
    print("\nPre-loading vLLM with base model (ready for first checkpoint at step 1000)...")
    llm = LLM(
        model=eval_config["model"]["name"],  # Load base model
        dtype="bfloat16" if eval_config["model"]["dtype"] == "bfloat16" else "float16",
        gpu_memory_utilization=0.7,  # Reduced to fit available memory
    )
    print("✓ vLLM pre-loaded and ready")
    
    loss_model = None  # Will be initialized on first use and reused
    
    print("✓ Evaluation worker ready, waiting for checkpoints...")
    print("="*80)
    
    while not shutdown_event.is_set():
        try:
            # Get checkpoint with timeout to check shutdown event
            try:
                item = checkpoint_queue.get(timeout=1.0)
            except:
                continue  # Timeout, check shutdown event
            
            if item is None:  # Shutdown signal
                break
            
            trial_id, checkpoint_path = item
            print(f"\n[Eval] Processing checkpoint for Trial {trial_id}: {checkpoint_path}")
            
            try:
                # Clean up previous vLLM instance
                if llm is not None:
                    del llm
                    torch.cuda.empty_cache()
                
                # Load vLLM from checkpoint (simple approach - recreate each time)
                load_start = time.time()
                print(f"Loading vLLM from checkpoint...")
                # Note: device is controlled via CUDA_VISIBLE_DEVICES (already set above)
                llm = LLM(
                    model=checkpoint_path,
                    dtype="bfloat16" if eval_config["model"]["dtype"] == "bfloat16" else "float16",
                    gpu_memory_utilization=0.7,  # Reduced to fit available memory
                )
                load_time = time.time() - load_start
                print(f"✓ vLLM loaded from checkpoint (took {load_time:.2f}s)")
                
                # Run evaluation
                print(f"Running evaluation on {num_eval_samples} samples...")
                
                # Use the evaluate_checkpoint function for consistency
                from src.eval_worker import evaluate_checkpoint
                from src.config_loader import Config
                from src.analysis_utils import categorize_results, analyze_format_failures
                import wandb
                
                config = Config(eval_config)
                with open(eval_config["data"]["prompt_file"], 'r') as f:
                    prompt_template = f.read()
                
                metrics = evaluate_checkpoint(
                    llm, val_data, sampling_params, config, tokenizer, trial_id, prompt_template
                )
                
                # Clean up vLLM to free memory for loss computation
                print(f"Unloading vLLM to free memory for loss computation...")
                del llm
                llm = None
                torch.cuda.empty_cache()
                
                # Compute actual eval loss using transformers (reuse model)
                print(f"Computing eval loss...")
                eval_loss, loss_model = compute_eval_loss(checkpoint_path, val_data, tokenizer, eval_config, loss_model)
                print(f"✓ Eval loss computed: {eval_loss:.4f}")
                
                # Reload vLLM for next evaluation
                print(f"Reloading vLLM for next checkpoint...")
                llm = LLM(
                    model=checkpoint_path,
                    dtype="bfloat16" if eval_config["model"]["dtype"] == "bfloat16" else "float16",
                    gpu_memory_utilization=0.7,  # Reduced to fit available memory
                )
                print(f"✓ vLLM reloaded")
                
                # Get categorization data from the saved analysis report
                # (evaluate_checkpoint saves it but doesn't return it, so we need to recompute or read it)
                # For now, we'll compute categorization from the metrics we have
                # Note: This is a simplified version - full categorization is in the saved report
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
                
                # Log to the same wandb run as the training trial
                # Get the run info for this trial from shared dict
                run_info_key = "__trial_run_info__"
                run_info = None
                if run_info_key in result_dict:
                    run_info = result_dict[run_info_key].get(trial_id)
                
                if run_info:
                    # Resume/join the existing run using the run ID
                    wandb.init(
                        project=run_info.get("project", eval_config.get("logging", {}).get("wandb_project", "math-sft-optuna-asha")),
                        name=run_info.get("name"),
                        id=run_info.get("id"),  # Use the stored run ID
                        resume="allow",  # Resume the existing run
                        reinit=True
                    )
                else:
                    # Fallback: create new run if info not available yet
                    # This might happen if eval runs before training sets the info
                    wandb.init(
                        project=eval_config.get("logging", {}).get("wandb_project", "math-sft-optuna-asha"),
                        name=f"trial_{trial_id}",
                        group="optuna_search",
                        reinit=True
                    )
                
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
                    "trial_id": trial_id,
                })
                
                wandb.finish()
                
                # Store result (include categorization and eval_loss for reference)
                metrics_with_categorization = {
                    **metrics,
                    "eval_loss": eval_loss,
                    "categorization": {
                        "category_1_correct": category_1_count,
                        "category_2_wrong_answer": category_2_count,
                        "category_3_format_failure": category_3_count,
                    }
                }
                result_dict[trial_id] = metrics_with_categorization
                
                print(f"✓ Evaluation complete for Trial {trial_id}:")
                print(f"  - Eval Loss: {eval_loss:.4f}")
                print(f"  - Accuracy: {metrics['accuracy']:.3f}")
                print(f"  - Format Accuracy: {metrics['format_accuracy']:.3f}")
                print(f"  - Correct: {metrics['correct']}/{metrics['num_evaluated']}")
                print(f"  - Categorization:")
                print(f"    * Correct (format+answer): {category_1_count} ({category_1_count/total*100:.1f}%)")
                print(f"    * Wrong answer (format ok): {category_2_count} ({category_2_count/total*100:.1f}%)")
                print(f"    * Format failure: {category_3_count} ({category_3_count/total*100:.1f}%)")
                
            except Exception as e:
                print(f"⚠ Error evaluating Trial {trial_id}: {e}")
                import traceback
                traceback.print_exc()
                result_dict[trial_id] = {"accuracy": 0.0, "error": str(e)}
        
        except Exception as e:
            print(f"⚠ Error in eval worker loop: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    if llm is not None:
        del llm
        torch.cuda.empty_cache()
    
    print("="*80)
    print("PERSISTENT EVALUATION WORKER STOPPED")
    print("="*80)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function - returns eval loss to minimize."""
    
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
    weight_decay = trial.suggest_float("weight_decay", WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX)
    
    trial_id = trial.number
    print(f"\n{'='*80}")
    print(f"OPTUNA TRIAL {trial_id}: LR={lr:.2e}, BS={batch_size}, WD={weight_decay:.4f}")
    print(f"{'='*80}\n")
    
    # Create config
    config_dict = create_config(lr, batch_size, weight_decay, trial_id)
    config_path = f"config/optuna_trial_{trial_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    try:
        # Import here to avoid multiprocessing issues
        from src.config_loader import load_config, validate_config, Config
        from src.training_worker import training_loop
        from utils.dataset_loader import MathDatasetLoader
        from src.data_utils import prepare_sft_dataset
        from transformers import AutoTokenizer
        import torch
        
        # Load config
        config = load_config(config_path)
        validate_config(config)
        
        # Initialize W&B and store run info for eval worker
        run_name = f"trial_{trial_id}_lr{lr:.2e}_bs{batch_size}"
        wandb.init(
            project=config_dict["logging"]["wandb_project"],
            name=run_name,
            config={
                "trial_id": trial_id,
                "learning_rate": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "effective_batch_size": batch_size * config_dict["training"]["gradient_accumulation_steps"],
            },
            group="optuna_search",
            reinit=True
        )
        
        # Store run name and ID so eval worker can log to the same run
        if _shared_result_dict is not None:
            # Use a separate key for run info
            run_info_key = "__trial_run_info__"
            if run_info_key not in _shared_result_dict:
                _shared_result_dict[run_info_key] = {}
            _shared_result_dict[run_info_key][trial_id] = {
                "name": run_name,
                "id": wandb.run.id,  # Store the actual run ID
                "project": config_dict["logging"]["wandb_project"]
            }
        
        # Load datasets
        loader = MathDatasetLoader()
        loader.load_all_subsets()  # Load all subsets
        train_examples = loader.collect_train_examples()
        test_examples = loader.collect_test_examples()
        
        with open(config_dict["data"]["prompt_file"], 'r') as f:
            prompt_template = f.read()
        
        tokenizer = AutoTokenizer.from_pretrained(config_dict["model"]["name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_data = prepare_sft_dataset(train_examples, prompt_template)
        
        # Create a wrapper queue that intercepts checkpoints and adds trial_id
        # This allows training_loop to work unchanged while sending to shared queue
        class TrialCheckpointQueue:
            """Wrapper that adds trial_id to checkpoints before sending to shared queue."""
            def __init__(self, shared_queue, trial_id):
                self.shared_queue = shared_queue
                self.trial_id = trial_id
            
            def put(self, checkpoint_path, block=True, timeout=None):
                """Put checkpoint with trial_id into shared queue."""
                if checkpoint_path is None:
                    # Shutdown signal - don't forward to eval
                    return
                print(f"  Queuing checkpoint for eval: {checkpoint_path}")
                self.shared_queue.put((self.trial_id, str(checkpoint_path)), block=block, timeout=timeout)
            
            def put_nowait(self, checkpoint_path):
                """Non-blocking put."""
                self.put(checkpoint_path, block=False)
            
            def full(self):
                """Check if shared queue is full."""
                return self.shared_queue.full()
        
        # Use wrapper queue - routes checkpoints to shared queue with trial_id
        eval_queue_wrapper = TrialCheckpointQueue(_shared_checkpoint_queue, trial_id)
        
        # Create stop signal for early termination
        stop_signal = mp.Event()
        
        # Store stop signal in shared dict for eval worker access
        trial_stop_key = f"__stop_signal_{trial_id}__"
        _shared_result_dict[trial_stop_key] = stop_signal
        
        # Run training in background (async on GPU0)
        print(f"\nStarting async training on GPU0...")
        training_thread = threading.Thread(
            target=training_loop,
            args=(config, train_data, tokenizer, eval_queue_wrapper, stop_signal),
            daemon=True
        )
        training_thread.start()
        
        # Wait for training to complete or be stopped by eval
        training_thread.join(timeout=None)  # Wait indefinitely
        
        # Get final eval loss (last reported)
        best_loss = _shared_result_dict.get(f"__best_loss_{trial_id}__", float('inf'))
        
        # Cleanup
        if trial_stop_key in _shared_result_dict:
            del _shared_result_dict[trial_stop_key]
        if f"__best_loss_{trial_id}__" in _shared_result_dict:
            del _shared_result_dict[f"__best_loss_{trial_id}__"]
        
        wandb.finish()
        
        print(f"✓ Trial {trial_id} complete: best_eval_loss={best_loss:.4f}")
        return best_loss
        
    except Exception as e:
        print(f"⚠ Trial {trial_id} failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        # Mark as failed in result dict
        if _shared_result_dict is not None:
            _shared_result_dict[trial_id] = {"eval_loss": float('inf'), "error": str(e)}
        raise optuna.TrialPruned()
    
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)


def main():
    """Run Optuna hyperparameter optimization with persistent eval worker."""
    global _shared_checkpoint_queue, _shared_result_manager, _shared_result_dict, _persistent_eval_process
    
    print("="*80)
    print("SFT HYPERPARAMETER OPTIMIZATION (Optuna TPE)")
    print("="*80)
    print(f"\nArchitecture: Persistent evaluation worker + training trials")
    print(f"\nSearch space:")
    print(f"  - Learning rate: [{LR_MIN:.0e}, {LR_MAX:.0e}] (log scale)")
    print(f"  - Batch size: {BATCH_SIZE_CHOICES}")
    print(f"\nOptimization:")
    print(f"  - Algorithm: TPE (Tree-structured Parzen Estimator)")
    print(f"  - Trials: {N_TRIALS}")
    print(f"  - Objective: Minimize eval/loss")
    print(f"\nFixed settings:")
    for k, v in FIXED_CONFIG.items():
        print(f"  - {k}: {v}")
    print("="*80)
    
    # Create directories
    Path("results/optuna").mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(parents=True, exist_ok=True)
    
    # Load validation data once (shared across all trials)
    print("\n" + "="*80)
    print("LOADING VALIDATION DATA (shared across all trials)")
    print("="*80)
    from utils.dataset_loader import MathDatasetLoader
    from src.data_utils import prepare_sft_dataset
    
    loader = MathDatasetLoader()
    loader.load_all_subsets()
    test_examples = loader.collect_test_examples()
    
    with open("prompts/rl_zero.prompt", 'r') as f:
        prompt_template = f.read()
    
    val_data = prepare_sft_dataset(test_examples, prompt_template)
    print(f"✓ Loaded {len(val_data)} validation samples")
    
    # Create evaluation config (same for all trials)
    # GPU 1 is dedicated for the persistent evaluation worker
    eval_config = {
        "model": {
            "name": "Qwen/Qwen2.5-Math-1.5B",
            "dtype": "bfloat16"
        },
        "logging": {
            "wandb_project": "math-sft-optuna-asha",
            "wandb_entity": None,
        },
        "evaluation": {
            "device": "cuda:1",  # GPU 1: Dedicated for persistent eval worker
            "batch_size": 320,
            "num_eval_samples": FIXED_CONFIG["num_eval_samples"],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024
        },
        "generation": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "stop_sequences": ["</answer>"],
            "include_stop_str": True
        },
        "data": {
            "prompt_file": "prompts/rl_zero.prompt",
            "train_val_split": 0.8
        }
    }
    
    # Create shared queues and manager
    print("\n" + "="*80)
    print("SETTING UP PERSISTENT EVALUATION WORKER")
    print("="*80)
    manager = mp.Manager()
    _shared_checkpoint_queue = manager.Queue(maxsize=100)  # Increased for periodic checkpoints (eval every 1000 batches)
    _shared_result_dict = manager.dict()  # Shared dict: {trial_id: metrics}
    _shared_result_manager = manager
    
    shutdown_event = mp.Event()
    
    # Start persistent eval worker
    _persistent_eval_process = mp.Process(
        target=persistent_eval_worker,
        args=(_shared_checkpoint_queue, _shared_result_dict, eval_config, val_data, shutdown_event),
        name="PersistentEvalWorker"
    )
    _persistent_eval_process.start()
    print("✓ Persistent evaluation worker started")
    
    # Give eval worker time to initialize
    time.sleep(3)
    
    # Create Optuna study with ASHA pruner
    study = optuna.create_study(
        study_name="sft_hyperparam_search_asha",
        direction="minimize",  # Minimize eval loss
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,  # Minimum number of checkpoints to evaluate
            reduction_factor=3,  # Prune bottom 1/3 at each rung
            min_early_stopping_rate=0  # Start pruning from first rung
        )
    )
    
    try:
        # Run optimization
        print("\n" + "="*80)
        print("STARTING OPTUNA OPTIMIZATION")
        print("="*80)
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
            catch=(Exception,)
        )
    finally:
        # Shutdown eval worker
        print("\n" + "="*80)
        print("SHUTTING DOWN EVALUATION WORKER")
        print("="*80)
        shutdown_event.set()
        # Send None to queue to wake up worker
        _shared_checkpoint_queue.put(None)
        
        if _persistent_eval_process.is_alive():
            _persistent_eval_process.join(timeout=120)
            if _persistent_eval_process.is_alive():
                print("⚠ Eval worker did not terminate, forcing termination...")
                _persistent_eval_process.terminate()
                _persistent_eval_process.join(timeout=10)
        print("✓ Evaluation worker stopped")
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # Check if any trials completed successfully
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        print("\n⚠ WARNING: No trials completed successfully!")
        print("\nTrial states:")
        for trial in study.trials:
            print(f"  Trial {trial.number}: {trial.state}")
            if trial.state == optuna.trial.TrialState.FAIL:
                print(f"    Error: {trial.system_attrs.get('error', 'Unknown error')}")
        print("\nPlease check the error messages above and fix the issues.")
        return
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    
    # Save study results
    results_path = "results/optuna/study_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump({
            "best_trial": study.best_trial.number,
            "best_accuracy": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "completed_trials": len(completed_trials)
        }, f)
    
    print(f"\nResults saved to: {results_path}")
    print(f"\nCheck W&B project 'math-sft-optuna-asha' for detailed metrics.")
    
    # Show trial history
    print("\nTrial history:")
    print("-" * 50)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"  Trial {trial.number}: LR={trial.params['learning_rate']:.2e}, "
                  f"BS={trial.params['batch_size']}, Acc={trial.value:.4f}")


if __name__ == "__main__":
    main()
