#!/usr/bin/env python3
"""
Extract trial results from available data files.
This script tries to gather trial information from various sources.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

def extract_trial_configs() -> Dict[int, Dict]:
    """Extract trial configurations from config files."""
    configs = {}
    config_dir = Path("config")
    
    for config_file in config_dir.glob("optuna_trial_*.yaml"):
        try:
            trial_num = int(config_file.stem.split("_")[-1])
            with open(config_file) as f:
                config = yaml.safe_load(f)
                configs[trial_num] = {
                    "learning_rate": config.get("training", {}).get("learning_rate"),
                    "batch_size": config.get("training", {}).get("batch_size"),
                    "weight_decay": config.get("training", {}).get("weight_decay"),
                }
        except Exception as e:
            print(f"Warning: Could not parse {config_file}: {e}")
    
    return configs

def extract_summary_results() -> Dict[int, Dict]:
    """Extract summary results from analysis files."""
    summaries = {}
    analysis_dir = Path("results/analysis")
    
    # Note: summary_step files may not directly correspond to trials
    # But we can extract the data structure
    for summary_file in sorted(analysis_dir.glob("summary_step_*.json")):
        try:
            step_num = int(summary_file.stem.split("_")[-1])
            with open(summary_file) as f:
                summary = json.load(f)
                summaries[step_num] = {
                    "accuracy": summary.get("metrics", {}).get("accuracy"),
                    "format_accuracy": summary.get("metrics", {}).get("format_accuracy"),
                    "categorization": summary.get("categorization", {}),
                }
        except Exception as e:
            print(f"Warning: Could not parse {summary_file}: {e}")
    
    return summaries

def main():
    """Extract and display trial results."""
    print("="*80)
    print("EXTRACTING TRIAL RESULTS")
    print("="*80)
    
    # Check if study_results.yaml exists
    study_results_path = Path("results/optuna/study_results.yaml")
    if study_results_path.exists():
        print("\n✓ Found study_results.yaml")
        with open(study_results_path) as f:
            study_results = yaml.safe_load(f)
        print(f"  Best trial: {study_results.get('best_trial')}")
        print(f"  Best accuracy (eval loss): {study_results.get('best_accuracy')}")
        print(f"  Best params: {study_results.get('best_params')}")
    else:
        print("\n✗ study_results.yaml not found")
        print("  Results should be generated when the study completes.")
    
    # Extract configs
    configs = extract_trial_configs()
    print(f"\n✓ Found {len(configs)} trial configuration files")
    for trial_num in sorted(configs.keys()):
        cfg = configs[trial_num]
        print(f"  Trial {trial_num}: LR={cfg['learning_rate']:.2e}, BS={cfg['batch_size']}, WD={cfg['weight_decay']:.4f}")
    
    # Extract summaries
    summaries = extract_summary_results()
    print(f"\n✓ Found {len(summaries)} summary files")
    
    # Check trial directories
    trial_dirs = sorted([d for d in Path("results/optuna").iterdir() 
                        if d.is_dir() and d.name.startswith("trial_")])
    print(f"\n✓ Found {len(trial_dirs)} trial directories")
    for trial_dir in trial_dirs:
        trial_num = int(trial_dir.name.split("_")[1])
        has_checkpoint = (trial_dir / "final" / "model.safetensors").exists()
        print(f"  Trial {trial_num}: {'✓ checkpoint exists' if has_checkpoint else '✗ no checkpoint'}")

if __name__ == "__main__":
    main()

