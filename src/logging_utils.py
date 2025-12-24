"""Logging utilities for detailed evaluation metrics."""
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from transformers import AutoTokenizer


def compute_token_entropy(
    response: str,
    tokenizer: AutoTokenizer,
    logits: torch.Tensor = None
) -> float:
    """Compute average token entropy of a response.
    
    Args:
        response: Generated text response
        tokenizer: Tokenizer for encoding
        logits: Optional logits from generation (if available)
        
    Returns:
        Average entropy across tokens
    """
    if logits is not None:
        # If we have logits, compute entropy from them
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean().item()
    else:
        # Fallback: estimate entropy from token frequency (approximate)
        tokens = tokenizer.encode(response, add_special_tokens=False)
        if len(tokens) == 0:
            return 0.0
        
        # Simple entropy based on token distribution
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        # Approximate entropy (Shannon entropy of uniform distribution)
        if unique_tokens > 0:
            return np.log2(unique_tokens)
        return 0.0


def compute_response_length(response: str, tokenizer: AutoTokenizer) -> int:
    """Compute response length in tokens.
    
    Args:
        response: Generated text response
        tokenizer: Tokenizer for encoding
        
    Returns:
        Number of tokens in response
    """
    tokens = tokenizer.encode(response, add_special_tokens=False)
    return len(tokens)


class DetailedEvaluationLogger:
    """Logger for detailed per-test-case evaluation information."""
    
    def __init__(self, log_dir: str, eval_step: int):
        """Initialize detailed logger.
        
        Args:
            log_dir: Directory to save logs
            eval_step: Current evaluation step
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"eval_step_{eval_step}_{timestamp}.jsonl"
        
        self.test_cases = []
        self.correct_lengths = []
        self.incorrect_lengths = []
        
    def log_test_case(
        self,
        prompt: str,
        response: str,
        ground_truth: str,
        format_reward: float,
        answer_reward: float,
        total_reward: float,
        token_entropy: float,
        response_length: int,
        problem: str = None,
        solution: str = None
    ):
        """Log a single test case with all details.
        
        Args:
            prompt: Input prompt
            response: Generated response
            ground_truth: Ground truth answer (same as solution)
            format_reward: Format reward (0 or 1)
            answer_reward: Answer correctness reward (0 or 1)
            total_reward: Total reward
            token_entropy: Average token entropy
            response_length: Length of response in tokens
            problem: Original problem/question (optional)
            solution: Original solution text (optional)
        """
        test_case = {
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "rewards": {
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "total_reward": total_reward
            },
            "metrics": {
                "token_entropy": token_entropy,
                "response_length": response_length
            }
        }
        
        # Add problem and solution fields if provided
        if problem is not None:
            test_case["problem"] = problem
        if solution is not None:
            test_case["solution"] = solution
        
        self.test_cases.append(test_case)
        
        # Track lengths for correct/incorrect
        if total_reward == 1.0:
            self.correct_lengths.append(response_length)
        else:
            self.incorrect_lengths.append(response_length)
    
    def save(self):
        """Save all test cases to file and return summary statistics."""
        # Write detailed logs (one JSON per line)
        with open(self.log_file, 'w') as f:
            for test_case in self.test_cases:
                f.write(json.dumps(test_case) + '\n')
        
        # Compute summary statistics
        all_lengths = [tc['metrics']['response_length'] for tc in self.test_cases]
        all_entropies = [tc['metrics']['token_entropy'] for tc in self.test_cases]
        
        summary = {
            "num_test_cases": len(self.test_cases),
            "avg_response_length": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_response_length_correct": np.mean(self.correct_lengths) if self.correct_lengths else 0.0,
            "avg_response_length_incorrect": np.mean(self.incorrect_lengths) if self.incorrect_lengths else 0.0,
            "avg_token_entropy": np.mean(all_entropies) if all_entropies else 0.0,
            "num_correct": len(self.correct_lengths),
            "num_incorrect": len(self.incorrect_lengths)
        }
        
        # Save summary
        summary_file = self.log_file.parent / f"summary_{self.log_file.stem}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Detailed logs saved: {self.log_file}")
        print(f"✓ Summary saved: {summary_file}")
        
        return summary
