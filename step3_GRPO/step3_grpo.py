"""Main orchestrator for dual-GPU GRPO training pipeline."""
import multiprocessing as mp
import wandb
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config, validate_config
from src.grpo_training_worker import grpo_training_loop
from src.eval_worker import eval_worker
from utils.dataset_loader import MathDatasetLoader
from transformers import AutoTokenizer


def load_datasets(config):
    """Load and prepare datasets.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_data, val_data, tokenizer)
    """
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    # Load MATH dataset
    loader = MathDatasetLoader()
    datasets, subsets, total_train, total_test = loader.load_all_subsets()
    
    print(f"Loaded {len(subsets)} subsets")
    print(f"Total train: {total_train}, Total test: {total_test}")
    
    # Collect examples
    train_examples = loader.collect_train_examples(include_metadata=True)
    test_examples = loader.collect_test_examples(include_metadata=True)
    
    # Load tokenizer from SFT checkpoint
    print(f"\nLoading tokenizer from: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Train samples: {len(train_examples)}")
    print(f"✓ Val samples: {len(test_examples)}")
    print("="*80)
    
    return train_examples, test_examples, tokenizer


def main():
    """Main orchestrator for dual-GPU GRPO training."""
    # Load configuration
    config = load_config("config/grpo_config.yaml")
    validate_config(config)
    
    print("\n" + "="*80)
    print("DUAL-GPU GRPO TRAINING PIPELINE")
    print("="*80)
    print(f"Training Device: {config.training.device}")
    print(f"Evaluation Device: {config.evaluation.device}")
    print(f"Base Model: {config.model.name}")
    print("="*80)
    
    # Initialize W&B
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"math-grpo-dual-gpu-{timestamp}"
    
    wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=run_name,
        config=config.to_dict(),
        notes=f"Dual-GPU GRPO training - {timestamp}",
        tags=["grpo", "math", "dual-gpu", "bfloat16", "rl"]
    )
    
    # Store wandb run info to pass to eval worker
    wandb_run_info = {
        "name": run_name,
        "id": wandb.run.id,
        "project": config.logging.wandb_project,
        "entity": config.logging.wandb_entity
    }
    
    print(f"✓ W&B initialized: {run_name} (ID: {wandb.run.id})\n")
    
    # Load datasets
    train_data, val_data, tokenizer = load_datasets(config)
    
    # Create multiprocessing queue for checkpoint paths
    eval_queue = mp.Queue(maxsize=config.checkpointing.queue_maxsize)
    
    # Start evaluation worker process on GPU 1
    print("\n" + "="*80)
    print("STARTING PROCESSES")
    print("="*80)
    
    eval_process = mp.Process(
        target=eval_worker,
        args=(eval_queue, config, val_data, 42, None, wandb_run_info),
        name="EvalWorker-GPU1"
    )
    eval_process.start()
    print(f"✓ Evaluation worker started (PID: {eval_process.pid})")
    
    # Run GRPO training loop on GPU 0 (main process)
    try:
        grpo_training_loop(
            config=config,
            train_data=train_data,
            tokenizer=tokenizer,
            eval_queue=eval_queue
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n⚠ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n" + "="*80)
        print("CLEANUP")
        print("="*80)
        
        # Send shutdown signal to eval worker
        if not eval_queue.full():
            eval_queue.put(None)
        
        # Wait for eval worker to finish
        print("Waiting for evaluation worker to finish...")
        print("  (This may take a while as it processes all checkpoints)")
        eval_process.join(timeout=600)  # 10 minutes timeout
        
        if eval_process.is_alive():
            print("⚠ Evaluation worker didn't stop within timeout, terminating...")
            eval_process.terminate()
            eval_process.join(timeout=30)
            if eval_process.is_alive():
                print("⚠ Force killing eval worker...")
                eval_process.kill()
                eval_process.join()
        
        print("✓ All processes stopped")
        
        # Finish W&B
        wandb.finish()
        print("✓ W&B run finished")
    
    print("\n" + "="*80)
    print("GRPO TRAINING PIPELINE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    main()
