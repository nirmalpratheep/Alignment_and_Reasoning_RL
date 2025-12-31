# Step 1: Hyperparameter Optimization

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
cd step1_sft_hyper
python step1_sft_hyper.py
```

## Output

- **W&B Project**: [math-sft-optuna](https://wandb.ai/nirmalpratheep-self/math-sft-optuna)
  - View all trial runs, training metrics, and evaluation results
  - Compare hyperparameters across trials
  - Analyze categorization breakdowns and accuracy trends
- **Best params saved to**: `results/optuna/study_results.yaml`

## Expected Time

- ~4 min per trial
- ~1 hour total for 15 trials

## Results

### Best Trial

- **Trial**: 1
- **Accuracy**: 0.0300 (3.0%)
- **Learning Rate**: 7.98e-06
- **Batch Size**: 256

### Training & Evaluation Metrics

View detailed metrics and visualizations on the [wandb project page](https://wandb.ai/nirmalpratheep-self/math-sft-optuna).

#### Training Loss Progression
![Training Loss](images/training_loss.png)
*Training loss curves across all trials. Lower learning rates show more stable convergence.*

#### Evaluation Accuracy
![Evaluation Accuracy](images/eval_accuracy.png)
*Evaluation accuracy across trials. Best performance achieved in Trial 1 with 3.0% accuracy.*

#### Hyperparameter Comparison
![Hyperparameter Comparison](images/hyperparam_comparison.png)
*Learning rate vs batch size comparison showing optimal region around LR=7-8e-06 and BS=256.*

#### Categorization Breakdown
![Categorization](images/categorization.png)
*Breakdown of results into correct answers, wrong answers, and format failures across trials.*

### Trial History

| Trial | Learning Rate | Batch Size | Accuracy |
|-------|---------------|------------|----------|
| 0 | 1.54e-05 | 128 | 0.0100 (1.0%) |
| **1** | **7.98e-06** | **256** | **0.0300 (3.0%)** ⭐ |
| 2 | 5.32e-06 | 128 | 0.0200 (2.0%) |
| 3 | 8.66e-06 | 256 | 0.0100 (1.0%) |
| 4 | 3.13e-05 | 1024 | 0.0000 (0.0%) |
| 5 | 5.25e-05 | 512 | 0.0000 (0.0%) |
| 6 | 3.09e-05 | 1024 | 0.0000 (0.0%) |
| 7 | 5.63e-05 | 512 | 0.0000 (0.0%) |
| 8 | 7.21e-06 | 512 | 0.0200 (2.0%) |
| 9 | 3.64e-05 | 512 | 0.0050 (0.5%) |
| 10 | 1.46e-05 | 256 | 0.0100 (1.0%) |
| 11 | 5.11e-06 | 128 | 0.0050 (0.5%) |
| 12 | 1.04e-05 | 256 | 0.0200 (2.0%) |
| 13 | 5.24e-06 | 128 | 0.0250 (2.5%) |
| 14 | 1.36e-05 | 128 | 0.0100 (1.0%) |

### Key Observations

1. **Best Performance**: Trial 1 achieved 3.0% accuracy with LR=7.98e-06 and BS=256
2. **Learning Rate Range**: Lower learning rates (5e-6 to 1e-5) generally performed better than higher rates (>3e-5)
3. **Batch Size**: Medium batch sizes (256, 512) showed better results than very large (1024) or very small (128) batches
4. **High LR Issues**: Trials with learning rates >3e-5 (Trials 4-7) resulted in 0% accuracy, suggesting training instability
5. **Top 3 Trials**:
   - Trial 1: 3.0% (LR=7.98e-06, BS=256)
   - Trial 13: 2.5% (LR=5.24e-06, BS=128)
   - Trials 2, 8, 12: 2.0% (various LR/BS combinations)

### Recommendations

Based on the results, the optimal hyperparameters for this model and dataset are:
- **Learning Rate**: ~7-8e-06 (sweet spot around 7.98e-06)
- **Batch Size**: 256 (good balance between stability and performance)

For future experiments, consider:
- Narrowing the learning rate search around 7-8e-06
- Exploring batch sizes between 256-512
- Avoiding learning rates >3e-5 which cause training instability