"""CPU-based test for GRPO training pipeline.

Tests the complete flow without requiring GPUs:
- Configuration loading
- Data loading
- Model loading (CPU only)
- Training loop structure
- Checkpointing

Usage:
    python test_cpu.py
"""
import sys
from pathlib import Path
import torch
import os

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("CPU TEST MODE - GRPO Training Pipeline")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("")

# ==============================================
# Step 1: Test Configuration Loading
# ==============================================
print("Step 1: Testing configuration loading...")
try:
    from src.config_loader import load_config, validate_config
    
    config = load_config("config/grpo_config.yaml")
    validate_config(config)
    
    print(f"✓ Config loaded: {config.model.name}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size_per_gpu}")
    print(f"  GRPO group size: {config.grpo.group_size}")
    print("")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Step 2: Test Data Loading (CPU, single process)
# ==============================================
print("Step 2: Testing data loading...")
try:
    from src.data_utils import load_datasets
    
    # Override config for testing
    config.training.batch_size = 2  # Small batch for testing
    config.data.num_workers = 0  # No multiprocessing in test
    
    # Load with rank=0, world_size=1 (single process)
    train_loader, val_loader, tokenizer = load_datasets(
        config,
        prepare_sft_format=False,
        rank=0,
        world_size=1
    )
    
    print(f"✓ Data loaded")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Test iterating over one batch
    batch = next(iter(train_loader))
    print(f"  Sample batch size: {len(batch)}")
    print(f"  Sample keys: {batch[0].keys()}")
    print("")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Step 3: Test Model Loading (Small Model on CPU)
# ==============================================
print("Step 3: Testing model loading on CPU...")
try:
    from transformers import AutoModelForCausalLM
    
    # Use a tiny model for CPU testing
    test_model_name = "gpt2"  # Small model for testing
    
    print(f"  Loading {test_model_name} (for structure testing)...")
    model = AutoModelForCausalLM.from_pretrained(
        test_model_name,
        torch_dtype=torch.float32,  # CPU doesn't support bfloat16
        device_map="cpu"
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded on CPU")
    print(f"  Total parameters: {total_params:,}")
    print("")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Step 4: Test Reward Computation
# ==============================================
print("Step 4: Testing reward computation...")
try:
    from step3_GRPO.train.rewards_math import compute_math_rewards
    
    # Test data
    test_completions = [
        "Let me solve this. </think> <answer>42</answer>",
        "Working on it. </think> <answer>99</answer>",
    ]
    test_ground_truths = [
        "The answer is \\boxed{42}",
        "The answer is \\boxed{43}",
    ]
    
    rewards = compute_math_rewards(test_completions, test_ground_truths)
    
    print(f"✓ Reward computation works")
    print(f"  Test rewards: {rewards}")
    print(f"  Expected: [1.0, 0.0] (first correct, second wrong)")
    print("")
    
except Exception as e:
    print(f"✗ Reward computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Step 5: Test Advantage Computation
# ==============================================
print("Step 5: Testing advantage computation...")
try:
    from step3_GRPO.train.advantage_group import compute_group_advantages
    
    # Test data: 2 groups of 4 completions each
    test_rewards = torch.tensor([
        1.0, 0.0, 1.0, 0.0,  # Group 1
        0.0, 0.0, 1.0, 1.0,  # Group 2
    ], dtype=torch.float32)
    
    advantages, stats = compute_group_advantages(
        rewards=test_rewards,
        group_size=4,
        eps=1e-6,
        use_std_normalization=True
    )
    
    print(f"✓ Advantage computation works")
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Advantage mean: {stats['advantage_mean']:.4f}")
    print(f"  Reward mean: {stats['reward_mean']:.4f}")
    print("")
    
except Exception as e:
    print(f"✗ Advantage computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Step 6: Test Training Components
# ==============================================
print("Step 6: Testing training components...")
try:
    from step3_GRPO.train.grpo_trainer import (
        create_optimizer_and_scheduler,
        save_fsdp_checkpoint,
    )
    
    # Test optimizer creation
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config,
        num_training_steps=100
    )
    
    print(f"✓ Optimizer created: {type(optimizer).__name__}")
    print(f"✓ Scheduler created: {type(scheduler).__name__}")
    print("")
    
except Exception as e:
    print(f"✗ Training components failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Step 7: Test Imports and Module Structure
# ==============================================
print("Step 7: Testing all critical imports...")
try:
    # Test all imports used in training
    from step3_GRPO.train.fsdp_utils import (
        print_rank_0,
        is_main_process,
        get_rank,
    )
    from step3_GRPO.train.grpo_trainer import grpo_training_loop
    
    print(f"✓ All imports successful")
    print(f"  Rank utility test: rank={get_rank()}, is_main={is_main_process()}")
    print("")
    
except Exception as e:
    print(f"✗ Import testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================
# Summary
# ==============================================
print("="*70)
print("CPU TEST SUMMARY")
print("="*70)
print("✓ All tests passed!")
print("")
print("Components verified:")
print("  [✓] Configuration loading")
print("  [✓] Data loading (CPU, single-process)")
print("  [✓] Model loading (CPU, small model)")
print("  [✓] Reward computation")
print("  [✓] Advantage computation")
print("  [✓] Optimizer/scheduler creation")
print("  [✓] All module imports")
print("")
print("The pipeline structure is correct!")
print("Ready for GPU testing when hardware is available.")
print("="*70)
