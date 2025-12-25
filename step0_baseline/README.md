# Step 0: Baseline Model Evaluation

Evaluates Qwen 2.5 Math 1.5B base model on MATH dataset without any fine-tuning.

## Results: Qwen 2.5 Math 1.5B (Base)

| Category | Count | Percentage |
|----------|-------|------------|
| **Correct** | 142 | 2.84% |
| **Wrong Answer** | 657 | 13.14% |
| **Format Failure** | 4,201 | 84.02% |

## Key Findings

- **84% format failures**: Base model doesn't follow `<think>` / `<answer>` format
- **2.84% accuracy**: Expected for non-instruction-tuned model
- **Parser verified working**: Issues are model outputs, not grading

## Run

```bash
cd step0_baseline
python step0_baseLineModelEval.py
```

## Files

- `step0_baseLineModelEval.py` - Main evaluation script
- `evaluate_model.py` - Detailed evaluation with logging
