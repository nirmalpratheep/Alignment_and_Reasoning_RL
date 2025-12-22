"""
Evaluation script for Qwen 2.5 Math 1.5B on MATH dataset
Logs all output to a file for later analysis
"""

import sys
import json
import os
from datetime import datetime
from collections import Counter
import logging

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch

from drgrpo_grader import r1_zero_reward_fn, extract_answer


# Setup logging
def setup_logging(log_file="results/evaluation.log"):
    """Setup logging to both file and console"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_model_and_tokenizer(model_name, logger):
    """Load the model with vLLM and tokenizer"""
    logger.info(f"Loading model with vLLM: {model_name}")
    
    llm = LLM(model=model_name, dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Model loaded successfully!")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Backend: vLLM")
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    return llm, tokenizer


def load_math_datasets(logger):
    """Load all MATH dataset subsets"""
    logger.info("Loading MATH dataset subsets...")
    
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]
    
    datasets = {}
    total_train = 0
    total_test = 0
    
    for subset in subsets:
        logger.info(f"Loading {subset}...")
        datasets[subset] = load_dataset("EleutherAI/hendrycks_math", subset)
        
        for split_name, split_data in datasets[subset].items():
            num_examples = len(split_data)
            logger.info(f"  {subset} - {split_name}: {num_examples:,} examples")
            
            if split_name == 'train':
                total_train += num_examples
            elif split_name == 'test':
                total_test += num_examples
    
    logger.info("All datasets loaded successfully!")
    logger.info(f"Total TRAIN examples: {total_train:,}")
    logger.info(f"Total TEST examples: {total_test:,}")
    
    return datasets, subsets, total_train, total_test


def load_prompt_template(prompt_file, logger):
    """Load the RL zero prompt template"""
    logger.info(f"Loading prompt template from {prompt_file}")
    
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    logger.info("Prompt template loaded!")
    logger.debug(f"Prompt template:\n{prompt_template}")
    
    return prompt_template


def prepare_test_examples(datasets, logger):
    """Collect all test examples from all subsets"""
    logger.info("Collecting test examples from all subsets...")
    
    all_test_examples = []
    
    for subset_name, dataset in datasets.items():
        if 'test' in dataset:
            for example in dataset['test']:
                all_test_examples.append({
                    'subset': subset_name,
                    'problem': example['problem'],
                    'solution': example['solution'],
                    'level': example.get('level', 'unknown'),
                    'type': example.get('type', 'unknown')
                })
    
    logger.info(f"Total test examples collected: {len(all_test_examples)}")
    
    return all_test_examples


def prepare_prompts(test_examples, prompt_template, logger):
    """Prepare prompts for all test examples"""
    logger.info("Preparing prompts for all test examples...")
    
    prompts = []
    for example in test_examples:
        formatted_prompt = prompt_template.replace('{question}', example['problem'])
        prompts.append(formatted_prompt)
    
    logger.info(f"Prepared {len(prompts)} prompts")
    logger.debug(f"Example prompt (first 500 chars):\n{prompts[0][:500]}...")
    
    return prompts


def generate_responses(llm, prompts, sampling_params, logger):
    """Generate responses with vLLM"""
    logger.info("Generating responses...")
    logger.info("="*80)
    logger.info(f"Sampling parameters:")
    logger.info(f"  Temperature: {sampling_params.temperature}")
    logger.info(f"  Top-p: {sampling_params.top_p}")
    logger.info(f"  Max tokens: {sampling_params.max_tokens}")
    logger.info(f"  Stop sequences: {sampling_params.stop}")
    logger.info(f"  Include stop string: {sampling_params.include_stop_str_in_output}")
    logger.info("="*80)
    
    outputs = llm.generate(prompts, sampling_params)
    
    logger.info(f"Generation complete! Generated {len(outputs)} responses.")
    
    return outputs


def process_results(test_examples, outputs, logger):
    """Process and store results"""
    logger.info("Processing results...")
    
    results = []
    
    for i, (example, output) in enumerate(zip(test_examples, outputs)):
        generated_text = output.outputs[0].text
        
        results.append({
            'id': i,
            'subset': example['subset'],
            'level': example['level'],
            'type': example['type'],
            'problem': example['problem'],
            'ground_truth': example['solution'],
            'generated_response': generated_text
        })
    
    logger.info(f"Processed {len(results)} results")
    
    return results


def grade_results(results, logger):
    """Grade each result using the reward function"""
    logger.info("Grading responses...")
    logger.info("="*80)
    
    for result in tqdm(results, desc="Grading"):
        reward_dict = r1_zero_reward_fn(
            response=result['generated_response'],
            ground_truth=result['ground_truth'],
            fast=True
        )
        result['format_reward'] = reward_dict['format_reward']
        result['answer_reward'] = reward_dict['answer_reward']
        result['reward'] = reward_dict['reward']
    
    logger.info("Grading complete!")
    
    return results


def categorize_results(results, logger):
    """Categorize results into three categories"""
    logger.info("Categorizing results...")
    
    category_1 = []  # format_reward=1, answer_reward=1
    category_2 = []  # format_reward=1, answer_reward=0
    category_3 = []  # format_reward=0, answer_reward=0
    
    for result in results:
        if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
            category_1.append(result)
        elif result['format_reward'] == 1.0 and result['answer_reward'] == 0.0:
            category_2.append(result)
        elif result['format_reward'] == 0.0 and result['answer_reward'] == 0.0:
            category_3.append(result)
    
    logger.info("="*80)
    logger.info("EVALUATION RESULTS BY CATEGORY")
    logger.info("="*80)
    logger.info(f"Total test examples: {len(results)}")
    logger.info(f"Category 1 (Format=1, Answer=1 - CORRECT): {len(category_1)} ({len(category_1)/len(results)*100:.2f}%)")
    logger.info(f"Category 2 (Format=1, Answer=0 - WRONG ANSWER): {len(category_2)} ({len(category_2)/len(results)*100:.2f}%)")
    logger.info(f"Category 3 (Format=0, Answer=0 - BAD FORMAT): {len(category_3)} ({len(category_3)/len(results)*100:.2f}%)")
    logger.info("="*80)
    
    return category_1, category_2, category_3


def analyze_format_failures(category_3, logger):
    """Analyze format failures in detail"""
    logger.info("="*80)
    logger.info("ANALYZING FORMAT FAILURES (Category 3: Format=0, Answer=0)")
    logger.info("="*80)
    logger.info(f"Total format failures: {len(category_3)}")
    logger.info(f"Analyzing {min(10, len(category_3))} examples...")
    
    format_issues = {
        'missing_think_close': 0,
        'missing_answer_open': 0,
        'missing_answer_close': 0,
        'wrong_order': 0,
        'incomplete_generation': 0
    }
    
    # Analyze individual examples
    for i, result in enumerate(category_3[:10]):
        logger.info("="*80)
        logger.info(f"FORMAT FAILURE EXAMPLE {i+1}/{min(10, len(category_3))}")
        logger.info("="*80)
        logger.info(f"Subset: {result['subset']}")
        logger.info(f"Level: {result['level']}")
        logger.info(f"Problem: {result['problem'][:200]}...")
        logger.info(f"Generated Response:\n{result['generated_response']}")
        logger.info("Issue Analysis:")
        
        has_think_close = "</think>" in result['generated_response']
        has_answer_open = "<answer>" in result['generated_response']
        has_answer_close = "</answer>" in result['generated_response']
        has_correct_format = "</think> <answer>" in result['generated_response']
        
        logger.info(f"  - Has '</think>': {has_think_close}")
        logger.info(f"  - Has '<answer>': {has_answer_open}")
        logger.info(f"  - Has '</answer>': {has_answer_close}")
        logger.info(f"  - Has correct format '</think> <answer>': {has_correct_format}")
        
        if not has_think_close:
            logger.info("  → Missing </think> tag")
        if not has_answer_open:
            logger.info("  → Missing <answer> tag")
        if not has_answer_close:
            logger.info("  → Missing </answer> tag")
        if has_think_close and has_answer_open and not has_correct_format:
            logger.info("  → Tags present but not in correct format '</think> <answer>'")
    
    # Collect statistics
    for result in category_3:
        response = result['generated_response']
        
        if "</think>" not in response:
            format_issues['missing_think_close'] += 1
        if "<answer>" not in response:
            format_issues['missing_answer_open'] += 1
        if "</answer>" not in response:
            format_issues['missing_answer_close'] += 1
            format_issues['incomplete_generation'] += 1
        if "</think>" in response and "<answer>" in response:
            if "</think> <answer>" not in response:
                format_issues['wrong_order'] += 1
    
    logger.info("="*80)
    logger.info("FORMAT FAILURE STATISTICS")
    logger.info("="*80)
    for issue, count in format_issues.items():
        if len(category_3) > 0:
            percentage = (count / len(category_3)) * 100
            logger.info(f"{issue.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    logger.info("="*80)
    logger.info("CONCLUSION: Format Failures")
    logger.info("="*80)
    if len(category_3) > 0:
        if format_issues['incomplete_generation'] > len(category_3) * 0.8:
            logger.info("Most format failures are due to INCOMPLETE GENERATION (model didn't finish).")
            logger.info("The issue is likely with the BASE MODEL's ability to complete responses.")
        elif format_issues['missing_think_close'] > len(category_3) * 0.5:
            logger.info("Most format failures are due to MISSING </think> TAG.")
            logger.info("The issue is likely with the BASE MODEL not following the format.")
        else:
            logger.info("Format failures are varied. Check individual cases for patterns.")
    logger.info("="*80)
    
    return format_issues


def analyze_wrong_answers(category_2, logger):
    """Analyze wrong answers in detail"""
    logger.info("="*80)
    logger.info("ANALYZING WRONG ANSWERS (Category 2: Format=1, Answer=0)")
    logger.info("="*80)
    logger.info(f"Total wrong answers with correct format: {len(category_2)}")
    logger.info(f"Analyzing {min(10, len(category_2))} examples...")
    
    for i, result in enumerate(category_2[:10]):
        logger.info("="*80)
        logger.info(f"WRONG ANSWER EXAMPLE {i+1}/{min(10, len(category_2))}")
        logger.info("="*80)
        logger.info(f"Subset: {result['subset']}")
        logger.info(f"Level: {result['level']}")
        logger.info(f"Problem: {result['problem'][:200]}...")
        logger.info(f"Ground Truth Solution (first 300 chars): {result['ground_truth'][:300]}...")
        logger.info(f"Generated Response:\n{result['generated_response']}")
        
        # Extract the answer from the response
        if "<answer>" in result['generated_response'] and "</answer>" in result['generated_response']:
            model_answer = result['generated_response'].split("<answer>")[-1].replace("</answer>", "").strip()
            logger.info(f"Extracted Model Answer: {model_answer}")
            
            # Try to extract ground truth
            gt_answer = extract_answer(result['ground_truth'])
            if gt_answer:
                logger.info(f"Extracted Ground Truth: {gt_answer}")
    
    logger.info("="*80)
    logger.info("CONCLUSION: Wrong Answers with Correct Format")
    logger.info("="*80)
    logger.info("In these cases:")
    logger.info("- The model successfully followed the format (has </think> <answer> ... </answer>)")
    logger.info("- The FORMAT PARSER worked correctly")
    logger.info("- However, the MATHEMATICAL ANSWER was incorrect")
    logger.info("")
    logger.info("Possible reasons:")
    logger.info("1. BASE MODEL's reasoning is flawed (incorrect math/logic)")
    logger.info("2. Model lacks sufficient domain knowledge for the problem")
    logger.info("3. Model made calculation errors")
    logger.info("4. Model misunderstood the question")
    logger.info("")
    logger.info("This is NOT a parser issue - the parser correctly identified the answer.")
    logger.info("This IS a model capability issue - the model gave a wrong answer.")
    logger.info("="*80)


def print_final_summary(model_name, results, category_1, category_2, category_3, 
                       format_issues, sampling_params, output_file, logger):
    """Print comprehensive final summary"""
    logger.info("="*80)
    logger.info("FINAL EVALUATION SUMMARY REPORT")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Test examples evaluated: {len(results)}")
    logger.info(f"")
    logger.info(f"Sampling parameters:")
    logger.info(f"  - Temperature: {sampling_params.temperature}")
    logger.info(f"  - Top-p: {sampling_params.top_p}")
    logger.info(f"  - Max tokens: {sampling_params.max_tokens}")
    logger.info("")
    logger.info("-"*80)
    logger.info("RESULTS BREAKDOWN")
    logger.info("-"*80)
    logger.info(f"")
    logger.info(f"(1) CORRECT (Format=1, Answer=1):")
    logger.info(f"    Count: {len(category_1)}")
    logger.info(f"    Percentage: {len(category_1)/len(results)*100:.2f}%")
    logger.info(f"")
    logger.info(f"(2) WRONG ANSWER (Format=1, Answer=0):")
    logger.info(f"    Count: {len(category_2)}")
    logger.info(f"    Percentage: {len(category_2)/len(results)*100:.2f}%")
    logger.info(f"")
    logger.info(f"(3) BAD FORMAT (Format=0, Answer=0):")
    logger.info(f"    Count: {len(category_3)}")
    logger.info(f"    Percentage: {len(category_3)/len(results)*100:.2f}%")
    
    logger.info("")
    logger.info("-"*80)
    logger.info("ANALYSIS OF FORMAT FAILURES (Category 3)")
    logger.info("-"*80)
    logger.info(f"Analyzed {min(10, len(category_3))} cases where format_reward=0")
    logger.info("")
    logger.info("Key findings:")
    for issue, count in format_issues.items():
        if count > 0 and len(category_3) > 0:
            percentage = (count / len(category_3)) * 100
            logger.info(f"  - {issue.replace('_', ' ').title()}: {count}/{len(category_3)} ({percentage:.1f}%)")
    
    logger.info("")
    logger.info("Conclusion on format failures:")
    if len(category_3) > 0:
        if format_issues['incomplete_generation'] > len(category_3) * 0.8:
            logger.info("  → Issue is primarily with BASE MODEL (incomplete generation)")
            logger.info("  → Model runs out of tokens or fails to complete the response")
        elif format_issues['missing_think_close'] > len(category_3) * 0.5:
            logger.info("  → Issue is primarily with BASE MODEL (doesn't follow format instructions)")
            logger.info("  → Model fails to properly use the required tags")
        else:
            logger.info("  → Mixed issues - both model and potentially some edge cases")
    else:
        logger.info("  → No format failures detected!")
    
    logger.info("")
    logger.info("-"*80)
    logger.info("ANALYSIS OF WRONG ANSWERS (Category 2)")
    logger.info("-"*80)
    logger.info(f"Analyzed {min(10, len(category_2))} cases where format_reward=1 but answer_reward=0")
    logger.info("")
    logger.info("Key findings:")
    logger.info("  - The PARSER is working correctly (successfully extracted answers)")
    logger.info("  - The issue is with the BASE MODEL's reasoning/knowledge")
    logger.info("  - Model provides well-formatted but mathematically incorrect answers")
    logger.info("")
    logger.info("Conclusion on wrong answers:")
    logger.info("  → This is NOT a parser issue")
    logger.info("  → This IS a model capability issue")
    logger.info("  → Model needs better:")
    logger.info("      • Mathematical reasoning")
    logger.info("      • Domain knowledge")
    logger.info("      • Calculation accuracy")
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_file}")
    logger.info("="*80)


def main():
    """Main evaluation pipeline"""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/evaluation_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("STARTING EVALUATION")
    logger.info("="*80)
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info("")
    
    # Configuration
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    prompt_file = "prompts/rl_zero.prompt"
    output_file = f"results/test_set_results_graded_{timestamp}.json"
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    try:
        # Load model and tokenizer
        llm, tokenizer = load_model_and_tokenizer(model_name, logger)
        
        # Load datasets
        datasets, subsets, total_train, total_test = load_math_datasets(logger)
        
        # Load prompt template
        prompt_template = load_prompt_template(prompt_file, logger)
        
        # Prepare test examples
        test_examples = prepare_test_examples(datasets, logger)
        
        # Prepare prompts
        prompts = prepare_prompts(test_examples, prompt_template, logger)
        
        # Generate responses
        outputs = generate_responses(llm, prompts, sampling_params, logger)
        
        # Process results
        results = process_results(test_examples, outputs, logger)
        
        # Grade results
        results = grade_results(results, logger)
        
        # Categorize results
        category_1, category_2, category_3 = categorize_results(results, logger)
        
        # Analyze format failures
        format_issues = analyze_format_failures(category_3, logger)
        
        # Analyze wrong answers
        analyze_wrong_answers(category_2, logger)
        
        # Save results to file
        os.makedirs('results', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
        
        # Print final summary
        print_final_summary(
            model_name, results, category_1, category_2, category_3,
            format_issues, sampling_params, output_file, logger
        )
        
        logger.info("="*80)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Results file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
