# SFT Training Results - Qwen 2.5 Math 1.5B

## Performance: Baseline vs Final SFT

| Metric | Baseline | Final SFT (Step 6) | Improvement |
|--------|----------|-------------------|-------------|
| **Overall Accuracy** | 2.84% | 7.88% | +5.04% |
| **Format Accuracy** | 16.0% | 26.46% | +10.46% |
| **Wrong Answer Rate** | 13.14% | 18.58% | +5.44% |
| **Format Failure Rate** | 84.02% | 73.54% | -10.48% |
| **Correct Count** | 142/5000 | 394/5000 | +252 |
| **Format Correct Count** | 799/5000 | 1323/5000 | +524 |

## Training Progression

Evaluation performed every 1000 training steps on 5000 test samples:

| Eval Step | Training Steps | Accuracy | Format Accuracy | Wrong Answer | Format Failure |
|-----------|----------------|----------|-----------------|--------------|----------------|
| Baseline | 0 | 2.84% | 16.0% | 13.14% | 84.02% |
| 0 | 0 | 5.4% | 15.52% | 10.12% | 84.48% |
| 1 | 1000 | 6.14% | 21.8% | 15.66% | 78.2% |
| 2 | 2000 | 6.62% | 24.52% | 17.9% | 75.48% |
| 3 | 3000 | 7.72% | 26.6% | 18.88% | 73.4% |
| 4 | 4000 | 7.7% | 26.88% | 19.18% | 73.12% |
| 5 | 5000 | 7.5% | 26.52% | 19.02% | 73.48% |
| 6 (Final) | 6000 | 7.88% | 26.46% | 18.58% | 73.54% |

## Architecture

**Dual-GPU Pipeline**:
- GPU 0: Training (BF16, gradient checkpointing)
- GPU 1: Evaluation (vLLM, continuous checkpoint loading)

**Key Techniques**:
- Gradient checkpointing for memory efficiency
- AdamW optimizer with linear warmup
- vLLM for fast batch inference
- CUDA_VISIBLE_DEVICES isolation per worker

## Memory Profile

| GPU | Component | Memory |
|-----|-----------|--------|
| GPU 0 | Training (BS=32) | ~45GB |
| GPU 1 | vLLM Inference | ~35GB |

## Configuration

- **Batch size**: 128 (effective: 256 with grad accum)
- **Learning rate**: 2e-5
- **Total training steps**: 6000
- **Eval frequency**: Every 1000 steps
- **Evaluation samples**: 5000 (full test set)

## Detailed Analysis

### Response Characteristics

| Metric | Baseline | Final SFT | Change |
|--------|----------|-----------|--------|
| **Avg Response Length** | N/A | 271.9 tokens | - |
| **Avg Length (Correct)** | N/A | 142.1 tokens | - |
| **Avg Length (Incorrect)** | N/A | 283.0 tokens | - |
| **Avg Token Entropy** | N/A | 6.15 | - |

### Format Failure Analysis (Final Step)

The primary issue remains **incomplete generation** (85.7% of format failures):
- **Missing `</think>` tag**: 78.2% of format failures
- **Missing `<answer>` tag**: 61.6% of format failures  
- **Missing `</answer>` tag**: 85.7% of format failures
- **Wrong order**: 3.5% of format failures
- **Incomplete generation**: 85.7% of format failures (model runs out of tokens)

### Key Findings

1. **Significant improvement in accuracy**: 2.84% → 7.88% (+5.04% absolute, +177% relative)
2. **Format accuracy improved**: 16.0% → 26.46% (+10.46% absolute, +65% relative)
3. **Format failures still dominant**: 73.54% of responses fail format requirements
4. **Wrong answers increased**: From 10.12% to 18.58% (more responses pass format but are incorrect)
5. **Incomplete generation is the main bottleneck**: 85.7% of format failures are due to incomplete responses (likely hitting max_tokens=1024 limit)
6. **Correct answers are shorter**: Average 142 tokens vs 283 tokens for incorrect answers

### Conclusions

1. **Format learning is moderate**: 26% format accuracy achieved, but still far from target
2. **Math reasoning improves slowly**: Only 7.88% correct answers after 6000 steps
3. **Bottleneck analysis**: 
   - Format failures: 73.54% (down from 84.02%)
   - Wrong answers: 18.58% (up from 13.14%)
   - The shift indicates format learning is happening, but reasoning quality needs improvement
4. **Generation length constraint**: Most format failures are incomplete generations, suggesting max_tokens=1024 may be limiting
5. **Next steps**: 
   - Consider increasing max_tokens for evaluation
   - RL (GRPO/PPO) for mathematical reasoning improvement
   - Continue SFT with more data or longer training
