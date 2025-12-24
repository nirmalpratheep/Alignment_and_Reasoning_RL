from dataset_loader import MathDatasetLoader
from drgrpo_grader import r1_zero_reward_fn
from training_utils import train_sft_model, prepare_sft_batch
import json
import os
import wandb
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
from datetime import datetime


# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Configuration
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100



def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load model and tokenizer from Hugging Face
    
    Args:
        model_name: Model identifier from Hugging Face Hub (default: Qwen/Qwen2.5-Math-1.5B)
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("="*80)
    print("LOADING MODEL AND TOKENIZER")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Load model with optimizations
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✓ Model loaded to {DEVICE} with BF16 precision")
    
    print("="*80)
    return model, tokenizer


def load_vllm_model(model_name: str = MODEL_NAME):
    """Load model using vLLM for efficient inference
    
    Args:
        model_name: Model identifier from Hugging Face Hub
    
    Returns:
        LLM: vLLM model for batch inference
    """
    print("="*80)
    print("LOADING vLLM MODEL FOR INFERENCE")
    print("="*80)
    print(f"Model: {model_name}")
    
    llm = LLM(
        model=model_name,
        dtype="float16",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✓ vLLM model loaded successfully")
    print("="*80)
    return llm


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    """Generate a response using the loaded model
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated response
    """
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Encode with attention mask
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (excluding the input prompt)
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def batch_generate_responses(llm, prompts: List[str], max_tokens: int = 1024) -> List[str]:
    """Generate responses for a batch of prompts using vLLM
    
    Args:
        llm: vLLM model
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate
    
    Returns:
        List[str]: Generated responses
    """
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses


def load_full_math_dataset():
    """Load the full MATH dataset with all 7 subsets and return train and test examples"""
    # Load the full MATH dataset with all 7 subsets
    loader = MathDatasetLoader()
    datasets, subsets, total_train, total_test = loader.load_all_subsets()

    print("="*80)
    print("FULL MATH DATASET LOADED")
    print("="*80)
    print(f"Subsets loaded: {subsets}")
    print(f"Total training examples: {total_train}")
    print(f"Total test examples: {total_test}")
    print()

    # Get and display statistics
    stats = loader.get_statistics()
    print("Subset Statistics:")
    print("-"*80)
    for subset, subset_stats in stats['subset_stats'].items():
        print(f"{subset:30s} - Train: {subset_stats['train']:5d}, Test: {subset_stats['test']:5d}")

    print("="*80)
    print(f"Total: Train: {total_train}, Test: {total_test}")
    print("="*80)

    # Collect all examples
    print("\nCollecting training examples...")
    train_examples = loader.collect_train_examples(include_metadata=True)
    print(f"Total training examples collected: {len(train_examples)}")

    print("\nCollecting test examples...")
    test_examples = loader.collect_test_examples(include_metadata=True)
    print(f"Total test examples collected: {len(test_examples)}")
    
    return train_examples, test_examples


def load_prompt_template(prompt_file: str) -> str:
    """Load the prompt template"""
    with open(prompt_file, 'r') as f:
        return f.read()


def extract_answer_from_solution(solution: str) -> str:
    """Extract the final answer from the solution (from \boxed{} or ####)"""
    # First try to extract from \boxed{}
    if "\\boxed{" in solution:
        # Find the content within \boxed{}
        start = solution.rfind("\\boxed{")
        if start != -1:
            start += len("\\boxed{")
            # Find matching closing brace
            brace_count = 1
            i = start
            while i < len(solution) and brace_count > 0:
                if solution[i] == "{":
                    brace_count += 1
                elif solution[i] == "}":
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                return solution[start:i-1].strip()
    
   
    return ""


def prepare_sft_dataset(examples: list, prompt_template: str) -> list:
    """Convert examples to SFT format (prompt-response pairs) with grading"""
    sft_data = []
    
    for example in examples:
        problem = example['problem']
        solution = example['solution']
        
        # Create prompt by replacing {question} in template
        prompt = prompt_template.replace('{question}', problem)
        
        # Extract answer from solution
        answer_value = extract_answer_from_solution(solution)
        
        # Create response: solution (thinking) + closing tags with answer
        # Format must match drgrpo_grader.py expectation: "</think> <answer>"
        response = solution + "\n</think> <answer>" + f"{answer_value}</answer>"
        
        # Grade the response using drgrpo_grader
        try:
            reward_dict = r1_zero_reward_fn(response=response, ground_truth=solution, fast=True)
        except Exception as e:
            reward_dict = {
                'format_reward': 0.0,
                'answer_reward': 0.0,
                'reward': 0.0
            }
        
        sft_data.append({
            'prompt': prompt,
            'response': response,
            'problem': problem,
            'solution': solution,
            'answer': answer_value,
            'format_reward': reward_dict.get('format_reward', 0.0),
            'answer_reward': reward_dict.get('answer_reward', 0.0),
            'reward': reward_dict.get('reward', 0.0)
        })
    
    return sft_data


def split_train_val(data: list, train_ratio: float = 0.8) -> tuple:
    """Split data into train and validation sets"""
    split_idx = int(len(data) * train_ratio)
    train = data[:split_idx]
    val = data[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train)} samples ({len(train)/len(data)*100:.1f}%)")
    print(f"  Val:   {len(val)} samples ({len(val)/len(data)*100:.1f}%)")
    
    return train, val


def create_batches(data: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Create batches from data"""
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches


def categorize_results(results: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Categorize results into three categories based on format and answer rewards"""
    category_1 = []  # format_reward=1, answer_reward=1 (CORRECT)
    category_2 = []  # format_reward=1, answer_reward=0 (WRONG ANSWER)
    category_3 = []  # format_reward=0, answer_reward=0 (BAD FORMAT)
    
    for result in results:
        format_reward = result.get('format_reward', 0.0)
        answer_reward = result.get('answer_reward', 0.0)
        
        if format_reward == 1.0 and answer_reward == 1.0:
            category_1.append(result)
        elif format_reward == 1.0 and answer_reward == 0.0:
            category_2.append(result)
        else:  # format_reward=0
            category_3.append(result)
    
    return category_1, category_2, category_3


def analyze_format_failures(results: List[Dict], batch_idx: int = 0) -> None:
    """Analyze format failures in detail"""
    category_1, category_2, category_3 = categorize_results(results)
    
    print("\n" + "="*80)
    print(f"BATCH {batch_idx} - RESULT CATEGORIZATION")
    print("="*80)
    print(f"Total examples: {len(results)}")
    print(f"Category 1 (Format=1, Answer=1 - CORRECT): {len(category_1)} ({len(category_1)/len(results)*100:.2f}%)")
    print(f"Category 2 (Format=1, Answer=0 - WRONG ANSWER): {len(category_2)} ({len(category_2)/len(results)*100:.2f}%)")
    print(f"Category 3 (Format=0, Answer=0 - BAD FORMAT): {len(category_3)} ({len(category_3)/len(results)*100:.2f}%)")
    print("="*80)
    
    if len(category_3) > 0:
        print(f"\nAnalyzing {min(5, len(category_3))} format failure examples...")
        print("-"*80)
        
        format_issues = {
            'missing_think_close': 0,
            'missing_answer_open': 0,
            'missing_answer_close': 0,
            'wrong_order': 0,
            'incomplete_generation': 0
        }
        
        for i, result in enumerate(category_3[:5]):
            print(f"\nFormat Failure Example {i+1}:")
            print(f"  Problem: {result['problem'][:100]}...")
            print(f"  Response (first 200 chars): {result['response'][:200]}...")
            
            response = result['response']
            has_think_close = "</think>" in response
            has_answer_open = "<answer>" in response
            has_answer_close = "</answer>" in response
            
            if not has_think_close:
                format_issues['missing_think_close'] += 1
                print(f"  Issue: Missing </think> tag")
            if not has_answer_open:
                format_issues['missing_answer_open'] += 1
                print(f"  Issue: Missing <answer> tag")
            if not has_answer_close:
                format_issues['missing_answer_close'] += 1
                format_issues['incomplete_generation'] += 1
                print(f"  Issue: Missing </answer> tag (incomplete)")
            if has_think_close and has_answer_open and "</think> <answer>" not in response:
                format_issues['wrong_order'] += 1
                print(f"  Issue: Tags present but not in correct format")
        
        print("\n" + "-"*80)
        print("Format Failure Statistics:")
        for issue, count in format_issues.items():
            if count > 0:
                percentage = (count / len(category_3)) * 100
                print(f"  {issue.replace('_', ' ').title()}: {count}/{len(category_3)} ({percentage:.1f}%)")
        print("="*80)
    
    if len(category_2) > 0:
        print(f"\nAnalyzing {min(3, len(category_2))} wrong answer examples...")
        print("-"*80)
        
        for i, result in enumerate(category_2[:3]):
            print(f"\nWrong Answer Example {i+1}:")
            print(f"  Problem: {result['problem'][:100]}...")
            print(f"  Model Answer: {result['answer']}")
            
            # Extract ground truth answer
            if "\\boxed{" in result['solution']:
                gt_start = result['solution'].rfind("\\boxed{")
                if gt_start != -1:
                    gt_start += len("\\boxed{")
                    brace_count = 1
                    j = gt_start
                    while j < len(result['solution']) and brace_count > 0:
                        if result['solution'][j] == "{":
                            brace_count += 1
                        elif result['solution'][j] == "}":
                            brace_count -= 1
                        j += 1
                    if brace_count == 0:
                        gt_answer = result['solution'][gt_start:j-1].strip()
                        print(f"  Ground Truth Answer: {gt_answer}")
        
        print("\n" + "-"*80)
        print("Wrong Answer Analysis:")
        print("  The PARSER is working correctly (format_reward=1)")
        print("  The issue is with mathematical ACCURACY (answer_reward=0)")
        print("  Model provides well-formatted but mathematically incorrect answers")
        print("="*80)


def evaluate_batch(batch: List[Dict], val_data: List[Dict], model, tokenizer, checkpoint_path: str = None) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Evaluate a batch by generating responses from the model and computing metrics
    
    Returns:
        metrics: Dictionary with evaluation metrics
        batch_results: List of results with generated responses and rewards for batch
        val_results: List of results with generated responses and rewards for validation set
    """
    metrics = {
        'batch_format_reward': 0.0,
        'batch_answer_reward': 0.0,
        'batch_reward': 0.0,
        'batch_accuracy': 0.0,
        'batch_format_accuracy': 0.0
    }
    
    # Load model from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading model from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # Generate responses for batch
    print(f"  Generating responses for {len(batch)} batch samples...")
    batch_prompts = [item['prompt'] for item in batch]
    batch_responses = []
    for prompt in tqdm(batch_prompts, desc="  Batch generation"):
        response = generate_response(model, tokenizer, prompt, max_tokens=1024)
        batch_responses.append(response)
    
    # Grade batch responses and create results
    batch_results = []
    total_format = 0
    total_answer = 0
    total_reward = 0
    
    for i, item in enumerate(batch):
        generated_response = batch_responses[i]
        ground_truth_solution = item.get('solution', '')
        
        # Grade the generated response
        reward_dict = r1_zero_reward_fn(response=generated_response, ground_truth=ground_truth_solution, fast=True)
        
        # Create result dictionary
        result = {
            'problem': item.get('problem', ''),
            'solution': ground_truth_solution,
            'response': generated_response,
            'format_reward': reward_dict.get('format_reward', 0.0),
            'answer_reward': reward_dict.get('answer_reward', 0.0),
            'reward': reward_dict.get('reward', 0.0)
        }
        batch_results.append(result)
        
        total_format += reward_dict.get('format_reward', 0.0)
        total_answer += reward_dict.get('answer_reward', 0.0)
        total_reward += reward_dict.get('reward', 0.0)
    
    if len(batch) > 0:
        metrics['batch_format_reward'] = total_format / len(batch)
        metrics['batch_answer_reward'] = total_answer / len(batch)
        metrics['batch_reward'] = total_reward / len(batch)
        metrics['batch_accuracy'] = total_answer / len(batch)
        metrics['batch_format_accuracy'] = total_format / len(batch)
    
    # Evaluate on validation set (sample subset for efficiency)
    val_sample_size = min(128, len(val_data))
    val_sample = val_data[:val_sample_size] if len(val_data) > val_sample_size else val_data
    
    print(f"  Generating responses for {len(val_sample)} validation samples...")
    val_prompts = [item['prompt'] for item in val_sample]
    val_responses = []
    for prompt in tqdm(val_prompts, desc="  Val generation"):
        response = generate_response(model, tokenizer, prompt, max_tokens=1024)
        val_responses.append(response)
    
    val_results = []
    val_format = 0
    val_answer = 0
    val_reward = 0
    
    for i, item in enumerate(val_sample):
        generated_response = val_responses[i]
        ground_truth_solution = item.get('solution', '')
        
        # Grade the generated response
        reward_dict = r1_zero_reward_fn(response=generated_response, ground_truth=ground_truth_solution, fast=True)
        
        # Create result dictionary
        result = {
            'problem': item.get('problem', ''),
            'solution': ground_truth_solution,
            'response': generated_response,
            'format_reward': reward_dict.get('format_reward', 0.0),
            'answer_reward': reward_dict.get('answer_reward', 0.0),
            'reward': reward_dict.get('reward', 0.0)
        }
        val_results.append(result)
        
        val_format += reward_dict.get('format_reward', 0.0)
        val_answer += reward_dict.get('answer_reward', 0.0)
        val_reward += reward_dict.get('reward', 0.0)
    
    if len(val_sample) > 0:
        metrics['val_format_reward'] = val_format / len(val_sample)
        metrics['val_answer_reward'] = val_answer / len(val_sample)
        metrics['val_reward'] = val_reward / len(val_sample)
        metrics['val_accuracy'] = val_answer / len(val_sample)
        metrics['val_format_accuracy'] = val_format / len(val_sample)
    
    return metrics, batch_results, val_results



def save_sft_dataset(train_data: list, val_data: list, output_dir: str = "results/sft_data") -> None:
    """Save SFT dataset to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved train data: {train_file} ({len(train_data)} samples)")
    
    # Save val
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved val data: {val_file} ({len(val_data)} samples)")


def load_and_prepare_datasets(prompt_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load and prepare SFT datasets"""
    # Load dataset
    train_examples, test_examples = load_full_math_dataset()
    
    # Load prompt template
    print("\n" + "="*80)
    print("PREPARING SFT DATASET")
    print("="*80)
    prompt_template = load_prompt_template(prompt_file)
    print("Prompt template loaded")
    
    # Prepare SFT data
    print("\nConverting examples to SFT format...")
    sft_train_data = prepare_sft_dataset(train_examples, prompt_template)
    print(f"Created {len(sft_train_data)} train SFT samples")
    
    sft_val_data = prepare_sft_dataset(test_examples, prompt_template)
    print(f"Created {len(sft_val_data)} validation SFT samples")
    
    return sft_train_data, sft_val_data


def save_datasets(train_data: List[Dict], val_data: List[Dict], output_dir: str = "results/sft_data") -> None:
    """Save SFT datasets to JSON files"""
    print("\n" + "="*80)
    print("SAVING SFT DATASETS")
    print("="*80)
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved train data: {train_file} ({len(train_data)} samples)")
    
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved val data: {val_file} ({len(val_data)} samples)")


def run_training_batches(train_data: List[Dict], val_data: List[Dict], 
                        batch_size: int = 128, num_batches: int = 5,
                        model=None, tokenizer=None, checkpoint_path: str = None) -> None:
    """Run training loop with batch evaluation and result categorization"""
    print("\n" + "="*80)
    print("TRAINING WITH BATCH EVALUATION AND RESULT CATEGORIZATION")
    print("="*80)
    
    # Load model from checkpoint if path provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained model from: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        print("✓ Model loaded successfully")
    elif model is None or tokenizer is None:
        print("Warning: No model/tokenizer provided and no checkpoint path. Using base model.")
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Limit evaluation to 50 samples (matching training limit)
    max_eval_samples = 50
    eval_train_data = train_data[:max_eval_samples]
    
    batches = create_batches(eval_train_data, batch_size)
    num_batches = min(num_batches, len(batches))
    
    print(f"Running {num_batches} batches (batch_size={batch_size}, total_samples={len(eval_train_data)})")
    
    all_val_results = []
    
    for batch_idx in range(num_batches):
        batch = batches[batch_idx]
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing {len(batch)} samples...")
        
        # Evaluate current batch and validation set
        metrics, batch_results, val_results = evaluate_batch(batch, val_data, model, tokenizer, checkpoint_path=None)
        
        # Accumulate validation results (only from first batch to avoid duplicates)
        if batch_idx == 0:
            all_val_results = val_results
        
        # Log to wandb
        wandb_log = {
            'batch_idx': batch_idx + 1,
            'batch_size': len(batch),
            'train/format_reward': metrics['batch_format_reward'],
            'train/answer_reward': metrics['batch_answer_reward'],
            'train/total_reward': metrics['batch_reward'],
            'train/accuracy': metrics['batch_accuracy'],
            'train/format_accuracy': metrics['batch_format_accuracy'],
            'val/format_reward': metrics['val_format_reward'],
            'val/answer_reward': metrics['val_answer_reward'],
            'val/total_reward': metrics['val_reward'],
            'val/accuracy': metrics['val_accuracy'],
            'val/format_accuracy': metrics['val_format_accuracy']
        }
        
        wandb.log(wandb_log)
        
        # Print metrics
        print(f"  Train - Format: {metrics['batch_format_accuracy']:.3f}, Answer: {metrics['batch_accuracy']:.3f}, Reward: {metrics['batch_reward']:.3f}")
        print(f"  Val   - Format: {metrics['val_format_accuracy']:.3f}, Answer: {metrics['val_accuracy']:.3f}, Reward: {metrics['val_reward']:.3f}")
        
        # Categorize and analyze batch results (using generated responses)
        analyze_format_failures(batch_results, batch_idx=batch_idx + 1)
    
    # Final analysis on validation set (using generated responses)
    print("\n" + "="*80)
    print("FINAL VALIDATION SET ANALYSIS")
    print("="*80)
    analyze_format_failures(all_val_results, batch_idx=0)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Create timestamp for unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"math-sft-{timestamp}"
    
    # Initialize wandb with timestamped name
    wandb.init(
        project="math-sft",
        name=run_name,
        notes=f"SFT training with BF16 precision - {timestamp}",
        tags=["sft", "math", "bf16", "gradient-checkpointing"]
    )
    print(f"✓ W&B run initialized: {run_name}")
    
    # Load model and tokenizer
    print("\n" + "="*80)
    print("STEP 1: LOADING MODEL AND TOKENIZER")
    print("="*80)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Load vLLM model for efficient batch inference (optional - set to True to use)
    use_vllm = False  # Set to True if you want to use vLLM for faster inference
    if use_vllm:
        llm = load_vllm_model(MODEL_NAME)
    else:
        llm = None
    
    # Load and prepare datasets
    print("\n" + "="*80)
    print("STEP 2: LOADING AND PREPARING DATASETS")
    print("="*80)
    sft_train_data, sft_val_data = load_and_prepare_datasets("prompts/rl_zero.prompt")
    
    # Save datasets
    print("\n" + "="*80)
    print("STEP 3: SAVING DATASETS")
    print("="*80)
    save_datasets(sft_train_data, sft_val_data)
    
    # Run SFT training with loss calculation and W&B logging
    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL WITH LOSS TRACKING")
    print("="*80)
    train_sft_model(
        model=model,
        tokenizer=tokenizer,
        train_data=sft_train_data,
        val_data=sft_val_data,
        batch_size=1,
        num_epochs=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_batches=50
    )
    
    # Run evaluation with batch metrics
    print("\n" + "="*80)
    print("STEP 5: EVALUATING WITH GRADING METRICS")
    print("="*80)
    # Load the trained model checkpoint
    checkpoint_path = "results/checkpoints/epoch_1"
    run_training_batches(sft_train_data, sft_val_data, batch_size=128, num_batches=5,
                        model=model, tokenizer=tokenizer, checkpoint_path=checkpoint_path)
    
    # Finish wandb
    wandb.finish()
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)



