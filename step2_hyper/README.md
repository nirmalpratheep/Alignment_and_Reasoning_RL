# Step 2: Hyperparameter Optimization

Uses Optuna (TPE/Bayesian Optimization) to find optimal learning rate and batch size for SFT.

## Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| Learning Rate | [5e-6, 1e-4] | Log scale |
| Batch Size | [128, 256, 512, 1024] | Categorical |

## Configuration

- **Algorithm**: TPE (Tree-structured Parzen Estimator)
- **Trials**: 15
- **Training steps**: 200 per trial
- **Eval samples**: 200 per trial

## Run

```bash
cd step2_hyper
python step2_sft_hyper.py
```

## Output

- W&B project: `math-sft-optuna`
- Best params saved to: `results/optuna/study_results.yaml`

## Expected Time

- ~4 min per trial
- ~1 hour total for 15 trials
