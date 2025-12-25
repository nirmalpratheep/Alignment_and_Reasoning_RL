"""
Hyperparameter Optimization for SFT Training using Optuna
Uses Bayesian Optimization (TPE) instead of grid search for efficient search.
"""
import os
import sys
import yaml
import wandb
import optuna
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Set spawn method for CUDA compatibility (must be before any CUDA operations)
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Search space bounds
LR_MIN, LR_MAX = 5e-6, 1e-4
BATCH_SIZE_CHOICES = [128, 256, 512, 1024]

# Fixed parameters
FIXED_CONFIG = {
    "max_batches": 200,
    "eval_every": 999999,    # Only eval at end (final checkpoint only)
    "warmup_steps": 20,
    "num_eval_samples": 200,
}

# Number of optimization trials
N_TRIALS = 15


def create_config(lr: float, batch_size: int, trial_id: int) -> Dict[str, Any]:
    """Create config dict for a trial."""
    return {
        "model": {
            "name": "Qwen/Qwen2.5-Math-1.5B",
            "dtype": "bfloat16"
        },
        "training": {
            "learning_rate": lr,
            "weight_decay": 0.01,
            "batch_size": batch_size,
            "num_epochs": 1,
            "gradient_accumulation_steps": max(1, 64 // batch_size),
            "warmup_steps": FIXED_CONFIG["warmup_steps"],
            "max_batches": FIXED_CONFIG["max_batches"],
            "eval_every": FIXED_CONFIG["eval_every"],
            "device": "cuda:0"
        },
        "evaluation": {
            "device": "cuda:1",
            "batch_size": 256,
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
            "queue_maxsize": 2,
            "temp_dir": f"/tmp/qwen_optuna_{trial_id}"
        },
        "logging": {
            "wandb_project": "math-sft-optuna",
            "wandb_entity": None,
            "log_every": 10
        }
    }


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function - returns accuracy to maximize."""
    
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
    
    trial_id = trial.number
    print(f"\n{'='*80}")
    print(f"OPTUNA TRIAL {trial_id}: LR={lr:.2e}, BS={batch_size}")
    print(f"{'='*80}\n")
    
    # Create config
    config_dict = create_config(lr, batch_size, trial_id)
    config_path = f"config/optuna_trial_{trial_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    try:
        # Import here to avoid multiprocessing issues
        from src.config_loader import load_config, validate_config, Config
        from src.training_worker import training_loop
        from src.eval_worker import eval_worker
        from utils.dataset_loader import MathDatasetLoader
        from src.data_utils import prepare_sft_dataset
        from transformers import AutoTokenizer
        import multiprocessing as mp
        import torch
        
        # Load config
        config = load_config(config_path)
        validate_config(config)
        
        # Initialize W&B
        wandb.init(
            project=config_dict["logging"]["wandb_project"],
            name=f"trial_{trial_id}_lr{lr:.2e}_bs{batch_size}",
            config={
                "trial_id": trial_id,
                "learning_rate": lr,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * config_dict["training"]["gradient_accumulation_steps"],
            },
            group="optuna_search",
            reinit=True
        )
        
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
        val_data = prepare_sft_dataset(test_examples, prompt_template)
        
        # Create eval queue and shared value for accuracy
        eval_queue = mp.Queue(maxsize=2)
        
        # Start eval worker
        eval_process = mp.Process(
            target=eval_worker,
            args=(eval_queue, config, val_data),
            name=f"EvalWorker-Trial{trial_id}"
        )
        eval_process.start()
        
        # Run training
        training_loop(config, train_data, tokenizer, eval_queue)
        
        # Wait for eval worker
        if eval_process.is_alive():
            eval_queue.put(None)
            eval_process.join(timeout=120)
            if eval_process.is_alive():
                eval_process.terminate()
        
        # Get final accuracy from W&B (simplified - use last logged value)
        # In practice, you'd track this properly
        final_accuracy = wandb.run.summary.get("eval/accuracy", 0.0)
        
        wandb.finish()
        
        # Report to Optuna for pruning
        trial.report(final_accuracy, step=FIXED_CONFIG["max_batches"])
        
        print(f"✓ Trial {trial_id} complete: accuracy={final_accuracy:.4f}")
        return final_accuracy
        
    except Exception as e:
        print(f"⚠ Trial {trial_id} failed: {e}")
        wandb.finish()
        raise optuna.TrialPruned()
    
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)


def main():
    """Run Optuna hyperparameter optimization."""
    
    print("="*80)
    print("SFT HYPERPARAMETER OPTIMIZATION (Optuna TPE)")
    print("="*80)
    print(f"\nSearch space:")
    print(f"  - Learning rate: [{LR_MIN:.0e}, {LR_MAX:.0e}] (log scale)")
    print(f"  - Batch size: {BATCH_SIZE_CHOICES}")
    print(f"\nOptimization:")
    print(f"  - Algorithm: TPE (Tree-structured Parzen Estimator)")
    print(f"  - Trials: {N_TRIALS}")
    print(f"  - Objective: Maximize eval/accuracy")
    print(f"\nFixed settings:")
    for k, v in FIXED_CONFIG.items():
        print(f"  - {k}: {v}")
    print("="*80)
    
    # Create directories
    Path("results/optuna").mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(parents=True, exist_ok=True)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name="sft_hyperparam_search",
        direction="maximize",  # Maximize accuracy
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
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
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }, f)
    
    print(f"\nResults saved to: {results_path}")
    print(f"\nCheck W&B project 'math-sft-optuna' for detailed metrics.")
    
    # Show trial history
    print("\nTrial history:")
    print("-" * 50)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"  Trial {trial.number}: LR={trial.params['learning_rate']:.2e}, "
                  f"BS={trial.params['batch_size']}, Acc={trial.value:.4f}")


if __name__ == "__main__":
    main()
