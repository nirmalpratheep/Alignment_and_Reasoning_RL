# Step 3: GRPO Training

Group Relative Policy Optimization (GRPO) training for mathematical reasoning, building on the SFT checkpoint from Step 1.

## Overview

This step implements GRPO training using the TRL (Transformer Reinforcement Learning) library. GRPO is a more efficient alternative to PPO that:
- Eliminates the need for a separate critic/value model
- Generates multiple completions per prompt and uses relative rewards
- Reduces memory usage and computational complexity

## Architecture

**Dual-GPU Pipeline** (same as Step 1):
- **GPU 0**: GRPO training using TRL's GRPOTrainer
- **GPU 1**: Evaluation using vLLM (same eval worker as SFT)

**Key Components**:
- `step3_grpo.py`: Main orchestrator (mirrors `step2_sft.py`)
- `src/grpo_training_worker.py`: GRPO training loop with TRL integration
- `config/grpo_config.yaml`: GRPO-specific configuration
- **Reward Function**: Reuses `utils/drgrpo_grader.py` from the existing codebase

## Configuration

See [`config/grpo_config.yaml`](../config/grpo_config.yaml) for all parameters.

**Key GRPO Parameters**:
- `num_generations`: 4 (number of completions per prompt)
- `kl_coef`: 0.05 (KL divergence penalty)
- `learning_rate`: 5e-7 (lower than SFT for RL fine-tuning)
- `batch_size`: 16 (smaller due to multiple generations)

**Model Source**:
- Loads from `results/checkpoints/checkpoint_6000` (SFT final checkpoint)

## Running the Training

### Prerequisites

Install TRL dependency:
```bash
cd z:\home\nirmalp\Alignment_and_Reasoning_RL
uv pip install trl>=0.12.0
```

### Start Training

```bash
python step3_GRPO/step3_grpo.py
```

The training will:
1. Load the SFT checkpoint from Step 1
2. Initialize dual-GPU setup (training on GPU 0, eval on GPU 1)
3. Start GRPO training with continual evaluation
4. Log all metrics to W&B project `math-grpo`

## Monitoring

- **W&B Project**: `math-grpo`
- **Training Metrics**: `train/loss`, `train/reward`, `train/kl_divergence`
- **Evaluation Metrics**: `eval/accuracy`, `eval/format_accuracy`

## Differences from Step 1 (SFT)

| Aspect | Step 1 (SFT) | Step 3 (GRPO) |
|--------|--------------|---------------|
| Algorithm | Supervised Fine-Tuning | Reinforcement Learning (GRPO) |
| Training Library | Manual PyTorch loop | TRL GRPOTrainer |
| Dataset Format | Prompt + completion pairs | Prompts only |
| Optimization | AdamW on cross-entropy loss | Policy gradient with rewards |
| Learning Rate | 2e-5 | 5e-7 (much lower) |
| Batch Size | 32 | 16 (needs multiple generations) |
| Model Source | Base Qwen model | SFT checkpoint |

## Expected Improvements

GRPO training should improve:
- **Mathematical Reasoning**: Better problem-solving through reward-based training
- **Format Compliance**: Stronger adherence to required output format
- **Answer Accuracy**: Higher percentage of correct solutions

The reward function from `drgrpo_grader.py` provides:
- Binary rewards for correct/incorrect answers
- Format validation
- Mathematical equivalence checking

## Notes

- GRPO is memory-intensive due to generating multiple completions per prompt
- Training may be slower than SFT due to the RL algorithm overhead
- The evaluation worker is shared with Step 1 (same `eval_worker.py`)
- Checkpoints are saved to `results/grpo_checkpoints/`
