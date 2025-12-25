# SFT Training Results - Qwen 2.5 Math 1.5B

## Performance: Baseline vs Final SFT

| Metric | Baseline | Final SFT (Step 6) | Improvement |
|--------|----------|-------------------|-------------|
| **Overall Accuracy** | 2.84% | 7.80% | +4.96% |
| **Format Accuracy** | 16.0% | 27.76% | +11.76% |
| **Wrong Answer Rate** | 13.14% | 19.96% | +6.82% |
| **Format Failure Rate** | 84.02% | 72.24% | -11.78% |
| **Correct Count** | 142/5000 | 390/5000 | +248 |
| **Format Correct Count** | 799/5000 | 1388/5000 | +589 |

## Training Progression

Evaluation performed every 1000 training steps on 5000 test samples:

| Eval Step | Training Steps | Accuracy | Format Accuracy | Wrong Answer | Format Failure |
|-----------|----------------|----------|-----------------|--------------|----------------|
| Baseline | 0 | 2.84% | 16.0% | 13.14% | 84.02% |
| 0 | 0 | 5.48% | 17.74% | 12.26% | 82.26% |
| 1 | 1000 | 7.34% | 21.00% | 13.66% | 79.00% |
| 2 | 2000 | 6.90% | 23.80% | 16.90% | 76.20% |
| 3 | 3000 | 7.52% | 25.38% | 17.86% | 74.62% |
| 4 | 4000 | 7.92% | 26.40% | 18.48% | 73.60% |
| 5 | 5000 | 7.82% | 27.50% | 19.68% | 72.50% |
| 6 (Final) | 6000 | 7.80% | 27.76% | 19.96% | 72.24% |

## Architecture

**Dual-GPU Pipeline**:
- **GPU 0**: Training (BF16, gradient checkpointing)
- **GPU 1**: Evaluation (vLLM, continuous checkpoint loading)

**Key Features**:
- **Unified W&B Logging**: Training and evaluation metrics logged to the same wandb run
- **Persistent Eval Worker**: Single eval worker processes all checkpoints efficiently
- **Gradient checkpointing** for memory efficiency
- **AdamW optimizer** with linear warmup
- **vLLM** for fast batch inference
- **CUDA_VISIBLE_DEVICES** isolation per worker

**W&B Integration**:
- Both training and evaluation metrics appear in the same wandb run
- Training metrics: `train/loss`, `train/learning_rate`, etc.
- Evaluation metrics: `eval/accuracy`, `eval/format_accuracy`, categorization breakdowns
- View results: [wandb project](https://wandb.ai/nirmalpratheep-self/math-sft)

## Memory Profile

| GPU | Component | Memory |
|-----|-----------|--------|
| GPU 0 | Training (BS=32) | ~45GB |
| GPU 1 | vLLM Inference | ~35GB |

## Configuration

### Current Settings (from `config/sft_config.yaml`)
- **Batch size**: 32 (effective: 64 with grad accum = 2)
- **Learning rate**: 2.0e-5
- **Total training steps**: 10000 (max_batches)
- **Eval frequency**: Every 1000 steps
- **Evaluation samples**: 10000 (full test set)

### Optimal Hyperparameters (from Step 2 Hyperparameter Optimization)

Based on hyperparameter optimization results, the optimal settings are:
- **Learning rate**: **7.98e-06** (optimal from Trial 1)
- **Batch size**: **256** (optimal from Trial 1)
- **Best accuracy achieved**: 3.0% (on 200 eval samples, 200 training steps)

**Note**: The current config uses LR=2e-5 and BS=32. Consider updating to the optimal values for better performance.

See [step2_hyper/README.md](../step2_hyper/README.md) for full hyperparameter optimization results.

## Detailed Analysis

### Response Characteristics

| Metric | Baseline | Final SFT | Change |
|--------|----------|-----------|--------|
| **Avg Response Length** | 279.0 tokens | 274.4 tokens | -4.6 tokens |
| **Avg Length (Correct)** | 120.6 tokens | 145.0 tokens | +24.4 tokens |
| **Avg Length (Incorrect)** | 288.2 tokens | 285.4 tokens | -2.8 tokens |
| **Avg Token Entropy** | 6.13 | 6.17 | +0.04 |

### Format Failure Analysis (Final Step)

The primary issue remains **incomplete generation** (87.0% of format failures):
- **Missing `</think>` tag**: 77.0% of format failures
- **Missing `<answer>` tag**: 61.8% of format failures  
- **Missing `</answer>` tag**: 87.0% of format failures
- **Wrong order**: 4.2% of format failures
- **Incomplete generation**: 87.0% of format failures (model runs out of tokens)

### Key Findings

1. **Significant improvement in accuracy**: 2.84% → 7.80% (+4.96% absolute, +175% relative)
2. **Format accuracy improved**: 16.0% → 27.76% (+11.76% absolute, +74% relative)
3. **Format failures still dominant**: 72.24% of responses fail format requirements (down from 84.02%)
4. **Wrong answers increased**: From 13.14% to 19.96% (more responses pass format but are incorrect)
5. **Incomplete generation is the main bottleneck**: 87.0% of format failures are due to incomplete responses (likely hitting max_tokens=1024 limit)
6. **Correct answers are shorter**: Average 145 tokens vs 285 tokens for incorrect answers

### Conclusions

1. **Format learning is moderate**: 27.76% format accuracy achieved, but still far from target
2. **Math reasoning improves slowly**: Only 7.80% correct answers after 6000 steps
3. **Bottleneck analysis**: 
   - Format failures: 72.24% (down from 82.26%)
   - Wrong answers: 19.96% (up from 12.26%)
   - The shift indicates format learning is happening, but reasoning quality needs improvement
4. **Generation length constraint**: Most format failures are incomplete generations (87.0%), suggesting max_tokens=1024 may be limiting
5. **Training progression**: 
   - Best accuracy achieved at step 4000 (7.92%)
   - Format accuracy continues improving through step 6 (27.76%)
   - Slight accuracy drop from step 4 to step 6 suggests potential overfitting
6. **Next steps**: 
   - Consider increasing max_tokens for evaluation (currently 1024)
   - RL (GRPO/PPO) for mathematical reasoning improvement
   - Continue SFT with more data or longer training
   - Try optimal hyperparameters from step2_hyper (LR=7.98e-06, BS=256)
