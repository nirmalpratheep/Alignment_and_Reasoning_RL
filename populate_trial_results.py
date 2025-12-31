#!/usr/bin/env python3
"""Extract trial results and generate README table"""
import json
import yaml
from pathlib import Path

# Read all summary files
summaries = {}
for f in sorted(Path("results/analysis").glob("summary_step_*.json")):
    step = int(f.stem.split("_")[-1])
    with open(f) as file:
        summaries[step] = json.load(file)

# Read trial configs
configs = {}
for f in Path("config").glob("optuna_trial_*.yaml"):
    trial = int(f.stem.split("_")[-1])
    with open(f) as file:
        cfg = yaml.safe_load(file)
        configs[trial] = {
            "lr": cfg["training"]["learning_rate"],
            "bs": cfg["training"]["batch_size"],
            "wd": cfg["training"]["weight_decay"]
        }

# Check which trials have checkpoints
trial_dirs = {}
for d in Path("results/optuna").iterdir():
    if d.is_dir() and d.name.startswith("trial_"):
        trial = int(d.name.split("_")[1])
        has_checkpoint = (d / "final" / "model.safetensors").exists()
        trial_dirs[trial] = has_checkpoint

print("="*80)
print("TRIAL RESULTS EXTRACTION")
print("="*80)

# Since eval_step doesn't directly map to trial_id, we'll use what we have
# The summary files show evaluation results, but we need to map them to trials
# For now, let's assume summary_step N might correspond to trial N-1 (step 0 might be baseline)

# Generate table rows
print("\nGenerating table data...")
print(f"\nAvailable trials with checkpoints: {sorted(trial_dirs.keys())}")
print(f"Available summary steps: {sorted(summaries.keys())}")

# Try to match: if we have 6 trials (0-5) and 7 summaries (0-6), 
# step 0 might be a baseline, steps 1-6 might be trials 0-5
# Or step 0 might be trial 0, step 1 trial 1, etc.

# Let's create a mapping - we'll use the summary data we have
# Since we only have 1 config file (trial 6), we can't map perfectly
# But we can show what data exists

results = []
for step in sorted(summaries.keys()):
    s = summaries[step]
    results.append({
        "step": step,
        "accuracy": s["metrics"]["accuracy"],
        "format_accuracy": s["metrics"]["format_accuracy"],
        "correct_pct": s["categorization"]["correct"]["percentage"],
        "wrong_pct": s["categorization"]["wrong_answer"]["percentage"],
        "format_fail_pct": s["categorization"]["format_failure"]["percentage"]
    })

print("\n" + "="*80)
print("EXTRACTED RESULTS (from summary_step files)")
print("="*80)
print(f"\n{'Step':<6} {'Accuracy':<12} {'Format Acc':<12} {'Correct %':<12} {'Wrong %':<12} {'Format Fail %':<12}")
print("-" * 80)
for r in results:
    print(f"{r['step']:<6} {r['accuracy']:<12.4f} {r['format_accuracy']:<12.4f} "
          f"{r['correct_pct']:<12.2f} {r['wrong_pct']:<12.2f} {r['format_fail_pct']:<12.2f}")

# Find best performing step by accuracy
if results:
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest step by accuracy: Step {best['step']} with {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")

print(f"\nNote: eval_loss values are not in summary files - they come from study_results.yaml or W&B")
print(f"Trial hyperparameters (LR, BS, WD) are in config files, but only trial 6 config was found")

