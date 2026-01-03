"""Data preparation utilities for SFT training."""
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.dataset_loader import MathDatasetLoader
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def extract_answer_from_solution(solution: str) -> str:
    """Extract the final answer from the solution (from \\boxed{} or ####).
    
    Args:
        solution: Solution text containing answer
        
    Returns:
        Extracted answer string
    """
    # Try to extract from \boxed{}
    if "\\boxed{" in solution:
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
    """Convert examples to SFT format (prompt-response pairs) with grading.
    
    Args:
        examples: List of dataset examples with 'problem' and 'solution'
        prompt_template: Prompt template with {question} placeholder
        
    Returns:
        List of SFT data dictionaries with prompt, response, and grading info
    """
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


class MathDataset(Dataset):
    """PyTorch Dataset wrapper for MATH examples."""
    
    def __init__(self, examples):
        """Initialize dataset with examples.
        
        Args:
            examples: List of dataset examples (either raw or SFT-formatted)
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_datasets(config, prepare_sft_format=False, rank=0, world_size=1):
    """Load and prepare datasets with DistributedSampler for multi-GPU training.
    
    Args:
        config: Configuration object
        prepare_sft_format: If True, prepare data in SFT format (prompt-response pairs).
                          If False, return raw examples (for GRPO).
        rank: Current process rank (for distributed training)
        world_size: Total number of processes (for distributed training)
        
    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
        - train_loader: DataLoader with DistributedSampler for training
        - val_loader: DataLoader with DistributedSampler for validation
        - tokenizer: Loaded tokenizer
    """
    is_main = (rank == 0)
    
    if is_main:
        print("="*80)
        print("LOADING DATASETS")
        print("="*80)
    
    # Load MATH dataset
    loader = MathDatasetLoader()
    datasets, subsets, total_train, total_test = loader.load_all_subsets()
    
    if is_main:
        print(f"Loaded {len(subsets)} subsets")
        print(f"Total train: {total_train}, Total test: {total_test}")
    
    # Collect examples
    train_examples = loader.collect_train_examples(include_metadata=True)
    test_examples = loader.collect_test_examples(include_metadata=True)
    
    # Load tokenizer
    if is_main:
        print(f"\nLoading tokenizer from: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data based on format requested
    if prepare_sft_format:
        # Load prompt template
        with open(config.data.prompt_file, 'r') as f:
            prompt_template = f.read()
        
        if is_main:
            print("\nPreparing SFT format...")
        train_data = prepare_sft_dataset(train_examples, prompt_template)
        val_data = prepare_sft_dataset(test_examples, prompt_template)
    else:
        # Return raw examples (for GRPO)
        train_data = train_examples
        val_data = test_examples
    
    # Create PyTorch Datasets
    train_dataset = MathDataset(train_data)
    val_dataset = MathDataset(val_data)
    
    # Create DistributedSamplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # Don't shuffle validation
        seed=42
    )
    
    # Get batch size (handle both SFT and GRPO config formats)
    batch_size = config.training.get('batch_size') or config.training.get('batch_size_per_gpu')
    if batch_size is None:
        raise ValueError("Config must specify either training.batch_size or training.batch_size_per_gpu")
    
    # Custom collate function to handle dictionary batches
    def collate_fn(batch):
        """Custom collate that returns list of dicts instead of dict of tensors."""
        return batch  # Just return the list of dictionaries as-is
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config.data.get('num_workers', 2),
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for consistent FSDP training
        collate_fn=collate_fn  # Use custom collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=config.data.get('num_workers', 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn  # Use custom collate
    )
    
    if is_main:
        print(f"\n✓ Dataset Distribution:")
        print(f"  Total train samples: {len(train_data)}")
        print(f"  Total val samples: {len(val_data)}")
        print(f"  Samples per GPU (train): {len(train_data) // world_size}")
        print(f"  Samples per GPU (val): {len(val_data) // world_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Total batches per epoch: {len(train_loader)}")
        print("="*80)
    
    return train_loader, val_loader, tokenizer
