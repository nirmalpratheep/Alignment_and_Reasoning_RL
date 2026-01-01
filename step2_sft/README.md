# Step 2: SFT Training Results - Qwen 2.5 Math 1.5B

## Results

### Baseline vs Final

| Metric | Baseline | Step 6 | Change |
|--------|----------|--------|--------|
| Accuracy | 2.84% | 23.46% | +20.62% |
| Format Accuracy | 16.0% | 86.40% | +70.40% |
| Correct | 2.84% | 23.46% | +20.62% |
| Wrong Answer | 13.14% | 62.94% | +49.80% |
| Format Failure | 84.02% | 13.60% | -70.42% |

### Training Progression

| Step | Accuracy | Format Acc | Correct (%) | Wrong Answer (%) | Format Failure (%) |
|------|----------|------------|-------------|------------------|-------------------|
| Baseline | 2.84% | 16.0% | 2.84 | 13.14 | 84.02 |
| 0 | 16.12% | 58.88% | 16.12 | 42.76 | 41.12 |
| 1 | 21.08% | 74.82% | 21.08 | 53.74 | 25.18 |
| 2 | 21.96% | 84.08% | 21.96 | 62.12 | 15.92 |
| 3 | 26.92% | 89.36% | 26.92 | 62.44 | 10.64 |
| 4 | 24.18% | 87.30% | 24.18 | 63.12 | 12.70 |
| 5 | 24.42% | 86.64% | 24.42 | 62.22 | 13.36 |
| 6 | 23.46% | 86.40% | 23.46 | 62.94 | 13.60 |

## Configuration

- Batch size: 32 (effective: 64 with grad accum = 2)
- Learning rate: 2.0e-5
- Evaluation: Every 1000 steps on 5000 test samples
- W&B: [math-sft](https://wandb.ai/nirmalpratheep-self/math-sft)

## Metrics (Step 6)

- Avg Response Length: 288.8 tokens
- Avg Length (Correct): 157.9 tokens
- Avg Length (Incorrect): 328.9 tokens
- Avg Token Entropy: 6.29

## Format Failures (Step 6)

- Missing `</think>`: 75.9%
- Missing `<answer>`: 63.4%
- Missing `</answer>`: 86.9%
- Incomplete generation: 86.9%
