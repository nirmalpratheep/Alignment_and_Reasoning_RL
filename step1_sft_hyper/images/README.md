# Screenshot Guide

This directory contains screenshots from the wandb project page for documentation.

## Screenshots to Capture

1. **training_loss.png**: 
   - Go to the wandb project: https://wandb.ai/nirmalpratheep-self/math-sft-optuna
   - Navigate to the "Charts" or "Metrics" tab
   - Select the `train/loss` metric
   - Show all trials overlaid or in a comparison view
   - Take screenshot

2. **eval_accuracy.png**:
   - In the same wandb project
   - Select the `eval/accuracy` metric
   - Show accuracy across all trials
   - Highlight the best trial (Trial 1 with 3.0%)
   - Take screenshot

3. **hyperparam_comparison.png**:
   - Go to the "Parallel Coordinates" or "Hyperparameter Importance" view
   - Show learning rate vs batch size vs accuracy
   - Or use the scatter plot view showing LR vs BS colored by accuracy
   - Take screenshot

4. **categorization.png**:
   - Select metrics: `eval/category_1_correct_pct`, `eval/category_2_wrong_answer_pct`, `eval/category_3_format_failure_pct`
   - Show the breakdown across trials
   - Or show a bar chart comparing the three categories
   - Take screenshot

## Tips

- Use browser zoom (Ctrl/Cmd + 0) to ensure good resolution
- Hide unnecessary UI elements if possible
- Include legend and axis labels in screenshots
- Save as PNG format for best quality

