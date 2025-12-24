"""Plot training progression from evaluation results."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re

def load_eval_results(results_dir="results/eval_logs"):
    """Load all evaluation summaries."""
    summaries = {}
    for summary_file in sorted(glob.glob(f"{results_dir}/summary_eval_step_*.json")):
        match = re.search(r"eval_step_(\d+)_", summary_file)
        if match:
            step = int(match.group(1))
            with open(summary_file) as f:
                summaries[step] = json.load(f)
    
    # Also load analysis summaries if available
    analysis_summaries = {}
    for analysis_file in sorted(glob.glob("results/analysis/summary_step_*.json")):
        match = re.search(r"summary_step_(\d+)\.json", analysis_file)
        if match:
            step = int(match.group(1))
            with open(analysis_file) as f:
                analysis_summaries[step] = json.load(f)
    
    return summaries, analysis_summaries

def plot_accuracy_progression(summaries, output_file="results/training_progression_accuracy.png"):
    """Plot accuracy metrics over training steps."""
    steps = sorted(summaries.keys())
    
    if not steps:
        print("No evaluation results found!")
        return
    
    accuracy = [summaries[s].get('num_correct', 0) / summaries[s].get('num_test_cases', 1) * 100 
                for s in steps]
    format_accuracy = []
    
    # Try to get format accuracy from analysis summaries
    for s in steps:
        # Check if we have analysis data
        analysis_file = f"results/analysis/summary_step_{s}.json"
        if Path(analysis_file).exists():
            with open(analysis_file) as f:
                analysis = json.load(f)
                format_accuracy.append(analysis['metrics'].get('format_accuracy', 0) * 100)
        else:
            format_accuracy.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, accuracy, 'o-', label='Accuracy (Correct Answers)', linewidth=2, markersize=8)
    ax.plot(steps, format_accuracy, 's-', label='Format Accuracy', linewidth=2, markersize=8)
    
    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Training Progression: Accuracy Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy plot: {output_file}")
    plt.close()

def plot_categorization_progression(summaries, output_file="results/training_progression_categorization.png"):
    """Plot category breakdown over training steps."""
    steps = sorted(summaries.keys())
    
    if not steps:
        return
    
    correct = []
    wrong_answer = []
    format_failure = []
    
    for s in steps:
        analysis_file = f"results/analysis/summary_step_{s}.json"
        if Path(analysis_file).exists():
            with open(analysis_file) as f:
                analysis = json.load(f)
                cat = analysis['categorization']
                correct.append(cat['correct']['percentage'])
                wrong_answer.append(cat['wrong_answer']['percentage'])
                format_failure.append(cat['format_failure']['percentage'])
        else:
            correct.append(0)
            wrong_answer.append(0)
            format_failure.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.array(steps)
    ax.fill_between(x, 0, format_failure, label='Format Failure', alpha=0.7, color='#d62728')
    ax.fill_between(x, format_failure, 
                    np.array(format_failure) + np.array(wrong_answer), 
                    label='Wrong Answer', alpha=0.7, color='#ff7f0e')
    ax.fill_between(x, 
                    np.array(format_failure) + np.array(wrong_answer),
                    np.array(format_failure) + np.array(wrong_answer) + np.array(correct),
                    label='Correct', alpha=0.7, color='#2ca02c')
    
    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Training Progression: Result Categorization', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved categorization plot: {output_file}")
    plt.close()

def plot_metrics_progression(summaries, output_file="results/training_progression_metrics.png"):
    """Plot response length and entropy over training steps."""
    steps = sorted(summaries.keys())
    
    if not steps:
        return
    
    response_length = [summaries[s].get('avg_response_length', 0) for s in steps]
    token_entropy = [summaries[s].get('avg_token_entropy', 0) for s in steps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Response Length
    ax1.plot(steps, response_length, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax1.set_xlabel('Evaluation Step', fontsize=12)
    ax1.set_ylabel('Average Response Length (tokens)', fontsize=12)
    ax1.set_title('Average Response Length Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Token Entropy
    ax2.plot(steps, token_entropy, 's-', color='#ff7f0e', linewidth=2, markersize=8)
    ax2.set_xlabel('Evaluation Step', fontsize=12)
    ax2.set_ylabel('Average Token Entropy', fontsize=12)
    ax2.set_title('Average Token Entropy Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics plot: {output_file}")
    plt.close()

def plot_format_failures_progression(summaries, output_file="results/training_progression_format_failures.png"):
    """Plot format failure breakdown over training steps."""
    steps = sorted(summaries.keys())
    
    if not steps:
        return
    
    issues = {
        'missing_think_close': [],
        'missing_answer_open': [],
        'missing_answer_close': [],
        'incomplete_generation': []
    }
    
    for s in steps:
        analysis_file = f"results/analysis/summary_step_{s}.json"
        if Path(analysis_file).exists():
            with open(analysis_file) as f:
                analysis = json.load(f)
                format_analysis = analysis.get('format_failure_analysis', {})
                for issue in issues.keys():
                    if issue in format_analysis:
                        issues[issue].append(format_analysis[issue]['percentage'])
                    else:
                        issues[issue].append(0)
        else:
            for issue in issues.keys():
                issues[issue].append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.array(steps)
    bottom = np.zeros(len(steps))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    labels = ['Missing </think>', 'Missing <answer>', 'Missing </answer>', 'Incomplete Generation']
    
    for i, (issue, label) in enumerate(zip(issues.keys(), labels)):
        values = np.array(issues[issue])
        ax.bar(x, values, bottom=bottom, label=label, alpha=0.7, color=colors[i])
        bottom += values
    
    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Percentage of Format Failures (%)', fontsize=12)
    ax.set_title('Format Failure Breakdown Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved format failures plot: {output_file}")
    plt.close()

def generate_comprehensive_report():
    """Generate all progression plots and summary."""
    print("="*80)
    print("GENERATING TRAINING PROGRESSION PLOTS")
    print("="*80)
    
    summaries, analysis_summaries = load_eval_results()
    
    if not summaries:
        print("⚠ No evaluation results found in results/eval_logs/")
        return
    
    print(f"\nFound {len(summaries)} evaluation steps")
    print(f"Steps: {sorted(summaries.keys())}\n")
    
    # Create output directory
    Path("results").mkdir(exist_ok=True)
    
    # Generate all plots
    plot_accuracy_progression(summaries)
    plot_categorization_progression(summaries)
    plot_metrics_progression(summaries)
    plot_format_failures_progression(summaries)
    
    # Generate summary table
    print("\n" + "="*80)
    print("TRAINING PROGRESSION SUMMARY")
    print("="*80)
    
    steps = sorted(summaries.keys())
    print(f"\n{'Step':<6} {'Accuracy':<12} {'Format Acc':<12} {'Response Len':<15} {'Token Entropy':<15}")
    print("-" * 80)
    
    for step in steps:
        summary = summaries[step]
        accuracy = (summary.get('num_correct', 0) / summary.get('num_test_cases', 1)) * 100
        
        # Try to get format accuracy
        analysis_file = f"results/analysis/summary_step_{step}.json"
        if Path(analysis_file).exists():
            with open(analysis_file) as f:
                analysis = json.load(f)
                format_acc = analysis['metrics'].get('format_accuracy', 0) * 100
        else:
            format_acc = 0.0
        
        resp_len = summary.get('avg_response_length', 0)
        entropy = summary.get('avg_token_entropy', 0)
        
        print(f"{step:<6} {accuracy:>10.2f}% {format_acc:>10.2f}% {resp_len:>13.1f} {entropy:>14.3f}")
    
    print("\n" + "="*80)
    print("✓ All plots saved to results/ directory")
    print("="*80)

if __name__ == "__main__":
    generate_comprehensive_report()

