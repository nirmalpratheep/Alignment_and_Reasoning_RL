# Step 1: Hyperparameter Optimization

Uses Optuna (TPE/Bayesian Optimization) to find optimal learning rate and batch size for SFT.

## Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| Learning Rate | [5e-6, 1e-4] | Log scale |
| Batch Size | [32, 64, 128, 256] | Categorical (optimized for 1.5B model) |
| Weight Decay | [0.0, 0.1] | Linear scale |

## Configuration

- **Algorithm**: TPE (Tree-structured Parzen Estimator)
- **Pruner**: ASHA (Asynchronous Successive Halving) - prunes worst ~60-70% of trials early
- **Trials**: 20
- **Objective**: Minimize eval loss (cross-entropy)
- **Training**: Up to 50,000 batches per trial (allow full dataset)
- **Evaluation**: Every 1000 batches on 500 samples
- **Early Stopping**: patience=3, min_delta=0.001
- **Model Reuse**: Transformers model loaded once, weights reloaded per trial (~27s saved/trial)

## Run

```bash
cd step1_sft_hyper
python step1_sft_hyper.py
```

## Output

- **W&B Project**: [math-sft-optuna-asha](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha)
  - View all trial runs, training metrics, and evaluation results
  - Compare hyperparameters across trials (LR, batch size, weight decay)
  - Analyze eval loss, accuracy trends, and categorization breakdowns
- **Best params**: `results/optuna/study_results.yaml` (best_trial, eval_loss, LR, batch_size, weight_decay)
- **Evaluation summaries**: `results/analysis/summary_step_*.json` (accuracy, format_accuracy, categorization)

## Expected Time

- ~10-20 min per trial (ASHA prunes many early)
- ~3-5 hours total for 20 trials (depends on early stopping)


## Technical Architecture

**Dual-GPU Pipeline:**
- **GPU 0**: Sequential trial training (one trial at a time)
- **GPU 1**: Persistent eval worker (processes all trials)

**Evaluation Optimization:**
- **vLLM**: Fast accuracy metrics (unloaded before loss computation)
- **Transformers**: Actual cross-entropy loss (model loaded once, weights reloaded)
- **Memory Management**: vLLM ↔ transformers switching to avoid OOM
- **Time Savings**: ~27 seconds per trial from model weight reuse

**Early Stopping (Configured):**
- Evaluate every 1000 batches
- Track eval loss history
- Stop trial if no improvement for 3 consecutive evaluations
- Min improvement threshold: 0.001
- Allow up to 50,000 batches (full dataset) if loss keeps improving

**ASHA Pruning:**
- Prunes underperforming trials aggressively (~60-70%)
- Works in tandem with early stopping
- Focuses compute on promising hyperparameter regions

**W&B Logging:**
- Project: `math-sft-optuna-asha`
- Metrics: `eval/loss`, `eval/accuracy`, `train/loss`
- Each trial logged individually with hyperparameters

## Results

### Best Result

**Evaluation Step 2** (Trial 4) achieved the best accuracy:

- Accuracy: 24.00%
- Format Accuracy: 87.78%
- Correct: 24.00%
- Wrong Answer: 63.78%
- Format Failure: 12.22%

Hyperparameters (LR, batch size, weight decay) for Trial 4: See [W&B project](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha) or `results/optuna/study_results.yaml` when study completes.

### Training & Evaluation Metrics

View detailed metrics and visualizations on the [wandb project page](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha).

#### Training Loss Progression
![Training Loss](images/training_loss.png)
*Training loss curves across all trials. Lower learning rates show more stable convergence.*

#### Evaluation Accuracy
![Evaluation Accuracy](images/eval_accuracy.png)
*Evaluation accuracy across trials. Best performance achieved in the trial with lowest eval loss.*

#### Hyperparameter Comparison
![Hyperparameter Comparison](images/hyperparam_comparison.png)
*Learning rate vs batch size comparison showing optimal region.*

#### Categorization Breakdown
![Categorization](images/categorization.png)
*Breakdown of results into correct answers, wrong answers, and format failures across trials.*

### Evaluation Results

| Eval Step | Accuracy | Format Acc | Correct (%) | Wrong Answer (%) | Format Failure (%) |
|-----------|----------|------------|-------------|------------------|--------------------|
| 0 | 6.88% | 22.22% | 6.88 | 15.34 | 77.78 |
| 1 | 2.80% | 9.12% | 2.80 | 6.32 | 90.88 |
| **2** | **24.00%** ⭐ | **87.78%** | **24.00** | **63.78** | **12.22** |
| 3 | 7.04% | 25.52% | 7.04 | 18.48 | 74.48 |
| 4 | 8.00% | 27.46% | 8.00 | 19.46 | 72.54 |
| 5 | 7.88% | 26.92% | 7.88 | 19.04 | 73.08 |
| 6 | 8.32% | 26.82% | 8.32 | 18.50 | 73.18 |

Eval loss and hyperparameters (LR, BS, WD) for each trial: `results/optuna/study_results.yaml` or [W&B project](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha).

### Observations

- Step 2 achieved best accuracy (24.00%) with strong format compliance (87.78% format accuracy)
- Format accuracy improved significantly from steps 0-1 (9-22%) to steps 2-6 (25-88%)
- Most steps show 72-91% format failures; step 2 reduced this to 12.22%
- Step 2: 24% correct, 64% wrong answer (format correct but answer incorrect), 12% format failure
- Accuracy variation: 2.80% (step 1) to 24.00% (step 2)