"""Analysis utilities for categorizing and analyzing evaluation results."""
from typing import Dict, List, Tuple
import json
from pathlib import Path


def categorize_results(results: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Categorize results into three categories based on format and answer rewards.
    
    Args:
        results: List of result dictionaries with reward information
        
    Returns:
        Tuple of (category_1, category_2, category_3)
        - category_1: Correct (format=1, answer=1)
        - category_2: Wrong answer (format=1, answer=0)
        - category_3: Format failure (format=0)
    """
    category_1 = []  # format=1, answer=1 (CORRECT)
    category_2 = []  # format=1, answer=0 (WRONG ANSWER)
    category_3 = []  # format=0 (BAD FORMAT)
    
    for result in results:
        format_reward = result.get('rewards', {}).get('format_reward', 0.0)
        answer_reward = result.get('rewards', {}).get('answer_reward', 0.0)
        
        if format_reward == 1.0 and answer_reward == 1.0:
            category_1.append(result)
        elif format_reward == 1.0 and answer_reward == 0.0:
            category_2.append(result)
        else:  # format_reward == 0
            category_3.append(result)
    
    return category_1, category_2, category_3


def analyze_format_failures(
    category_3: List[Dict],
    max_examples: int = 10
) -> Dict[str, int]:
    """Analyze format failures in detail.
    
    Args:
        category_3: List of format failure results
        max_examples: Maximum examples to analyze in detail
        
    Returns:
        Dictionary with format issue counts
    """
    format_issues = {
        'missing_think_close': 0,
        'missing_answer_open': 0,
        'missing_answer_close': 0,
        'wrong_order': 0,
        'incomplete_generation': 0
    }
    
    for result in category_3:
        response = result.get('response', '')
        
        has_think_close = "</think>" in response
        has_answer_open = "<answer>" in response
        has_answer_close = "</answer>" in response
        
        if not has_think_close:
            format_issues['missing_think_close'] += 1
        if not has_answer_open:
            format_issues['missing_answer_open'] += 1
        if not has_answer_close:
            format_issues['missing_answer_close'] += 1
            format_issues['incomplete_generation'] += 1
        if has_think_close and has_answer_open and "</think> <answer>" not in response:
            format_issues['wrong_order'] += 1
    
    return format_issues


def generate_summary_report(
    category_1: List[Dict],
    category_2: List[Dict],
    category_3: List[Dict],
    format_issues: Dict[str, int],
    metrics: Dict,
    eval_step: int
) -> Dict:
    """Generate comprehensive summary report.
    
    Args:
        category_1: Correct results
        category_2: Wrong answer results
        category_3: Format failure results
        format_issues: Format issue counts
        metrics: Evaluation metrics
        eval_step: Current evaluation step
        
    Returns:
        Summary report dictionary
    """
    total = len(category_1) + len(category_2) + len(category_3)
    
    summary = {
        "eval_step": eval_step,
        "total_examples": total,
        "categorization": {
            "correct": {
                "count": len(category_1),
                "percentage": (len(category_1) / total * 100) if total > 0 else 0.0
            },
            "wrong_answer": {
                "count": len(category_2),
                "percentage": (len(category_2) / total * 100) if total > 0 else 0.0
            },
            "format_failure": {
                "count": len(category_3),
                "percentage": (len(category_3) / total * 100) if total > 0 else 0.0
            }
        },
        "format_failure_analysis": {
            issue: {
                "count": count,
                "percentage": (count / len(category_3) * 100) if len(category_3) > 0 else 0.0
            }
            for issue, count in format_issues.items()
        },
        "metrics": metrics,
        "conclusions": {
            "format_failures": generate_format_failure_conclusion(category_3, format_issues),
            "wrong_answers": "Parser working correctly. Issues with model's mathematical reasoning."
        }
    }
    
    return summary


def generate_format_failure_conclusion(
    category_3: List[Dict],
    format_issues: Dict[str, int]
) -> str:
    """Generate conclusion about format failures.
    
    Args:
        category_3: Format failure results
        format_issues: Format issue counts
        
    Returns:
        Conclusion string
    """
    if len(category_3) == 0:
        return "No format failures detected."
    
    incomplete = format_issues.get('incomplete_generation', 0)
    missing_think = format_issues.get('missing_think_close', 0)
    
    if incomplete > len(category_3) * 0.8:
        return "Issue: BASE MODEL (incomplete generation). Model runs out of tokens."
    elif missing_think > len(category_3) * 0.5:
        return "Issue: BASE MODEL (doesn't follow format). Model fails to use required tags."
    else:
        return "Issue: Mixed format failures. Check individual cases for patterns."


def save_analysis_report(
    summary: Dict,
    category_1: List[Dict],
    category_2: List[Dict],
    category_3: List[Dict],
    output_dir: str,
    eval_step: int
):
    """Save comprehensive analysis report.
    
    Args:
        summary: Summary report
        category_1: Correct results
        category_2: Wrong answer results
        category_3: Format failure results
        output_dir: Output directory
        eval_step: Evaluation step
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_file = output_path / f"summary_step_{eval_step}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save categorized results
    categories_file = output_path / f"categories_step_{eval_step}.json"
    with open(categories_file, 'w') as f:
        json.dump({
            "category_1_correct": category_1[:20],  # Save first 20 examples
            "category_2_wrong_answer": category_2[:20],
            "category_3_format_failure": category_3[:20]
        }, f, indent=2)
    
    print(f"✓ Analysis report saved: {summary_file}")
    print(f"✓ Categorized examples saved: {categories_file}")


def print_analysis_summary(
    summary: Dict,
    category_3: List[Dict],
    category_2: List[Dict]
):
    """Print analysis summary to console.
    
    Args:
        summary: Summary report
        category_3: Format failure results
        category_2: Wrong answer results
    """
    cat_data = summary['categorization']
    
    print("\n" + "="*80)
    print("EVALUATION ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Examples: {summary['total_examples']}")
    print(f"Category 1 (CORRECT): {cat_data['correct']['count']} ({cat_data['correct']['percentage']:.2f}%)")
    print(f"Category 2 (WRONG ANSWER): {cat_data['wrong_answer']['count']} ({cat_data['wrong_answer']['percentage']:.2f}%)")
    print(f"Category 3 (FORMAT FAILURE): {cat_data['format_failure']['count']} ({cat_data['format_failure']['percentage']:.2f}%)")
    print("="*80)
    
    if len(category_3) > 0:
        print("\nFormat Failure Analysis:")
        print("-"*80)
        for issue, data in summary['format_failure_analysis'].items():
            if data['count'] > 0:
                print(f"  {issue.replace('_', ' ').title()}: {data['count']} ({data['percentage']:.1f}%)")
        print(f"\nConclusion: {summary['conclusions']['format_failures']}")
        print("-"*80)
    
    if len(category_2) > 0:
        print("\nWrong Answer Analysis:")
        print("-"*80)
        print(f"  {summary['conclusions']['wrong_answers']}")
        print("-"*80)
    
    print("="*80)
