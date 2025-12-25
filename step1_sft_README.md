# SFT Training Results - Qwen 2.5 Math 1.5B

## Performance: Baseline vs SFT

| Metric | Baseline | SFT (500 steps) | Improvement |
|--------|----------|-----------------|-------------|
| **Overall Accuracy** | 2.84% | 15.2% | +12.36% |
| **Format Accuracy** | 16.0% | 89.4% | +73.4% |
| **Wrong Answer Rate** | 13.1% | 74.2% | - |
| **Format Failure Rate** | 84.0% | 10.6% | -73.4% |

## Training Progression

```
Step   Loss    Accuracy   Format Acc
─────────────────────────────────────
   0   2.891     2.8%       16.0%
  50   1.245     5.6%       42.3%
 100   0.892     8.2%       61.5%
 200   0.654    11.4%       78.2%
 300   0.521    13.1%       84.6%
 400   0.448    14.3%       87.1%
 500   0.412    15.2%       89.4%
```

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

- **Batch size**: 32 (effective: 128 with grad accum)
- **Learning rate**: 2e-5
- **Steps**: 500
- **Eval frequency**: Every 10 steps

## Key Findings

1. **Format learning is fast**: 89% format accuracy by step 500
2. **Math reasoning improves slowly**: Only 15% correct answers
3. **Bottleneck shifted**: From format failure (84%→11%) to wrong answers (13%→74%)
4. **Next step**: RL (GRPO/PPO) for mathematical reasoning improvement
