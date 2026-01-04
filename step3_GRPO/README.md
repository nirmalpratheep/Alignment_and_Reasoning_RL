# Step 3: GRPO Training with TRL + vLLM

This module implements Group Relative Policy Optimization (GRPO) for math reasoning using HuggingFace TRL with vLLM colocate mode for efficient generation.

## Quick Start

```bash
# Single GPU
./step3_GRPO/launch_trl.sh

# Multi-GPU (e.g., 2 GPUs)
./step3_GRPO/launch_trl.sh 2
```

## Benchmark Results

**Model**: Qwen2.5-Nirmal-Math-1.5B-Instruct + GRPO (checkpoint-200)  
**Validation Set**: NuminaMath-CoT (5000 samples)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **40.46%** (2023/5000) |
| **Format Accuracy** | **96.72%** (4836/5000) |

### 3-Category Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| Category 1 (Correct) | 2023 | 40.46% |
| Category 2 (Wrong Answer) | 2813 | 56.26% |
| Category 3 (Format Failure) | 164 | 3.28% |

**Key Insights**:
- High format compliance (96.72%) indicates strong instruction following
- 40.46% accuracy on challenging math problems
- Format failures primarily due to incomplete generation (token limit)

## Files

| File | Description |
|------|-------------|
| `train_trl.py` | Main TRL GRPOTrainer with vLLM colocate mode |
| `launch_trl.sh` | Launch script with accelerate |
| `train/rewards_math.py` | Math reward function using `r1_zero_reward_fn` |

## Configuration

Uses `config/grpo_config.yaml` for:
- Model settings (name, dtype)
- Training hyperparameters (lr, batch size, steps)
- GRPO settings (group_size, temperature, max_tokens)
- Evaluation settings

## Key Features

- **TRL GRPOTrainer**: HuggingFace's optimized GRPO implementation
- **vLLM Colocate Mode**: Shares GPU memory between training and generation
- **Custom Math Rewards**: Uses `r1_zero_reward_fn` for answer verification
- **Multi-GPU Support**: Uses accelerate for distributed training
- **Final Evaluation**: Full validation with 3-category breakdown (correct/wrong/format failure)

## Output

After training:
- Model saved to `config.checkpointing.output_dir`
- Evaluation summary JSON with accuracy and source breakdown
- Wandb logging with metrics and tables
