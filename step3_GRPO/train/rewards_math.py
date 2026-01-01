"""Math verification reward function for GRPO training."""
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.drgrpo_grader import r1_zero_reward_fn


def compute_math_rewards(
    completions: List[str],
    ground_truths: List[str],
) -> List[float]:
    """Compute rewards for math problem completions.
    
    Args:
        completions: List of model-generated solutions
        ground_truths: List of correct solutions
        
    Returns:
        List of reward values (0.0 or 1.0 for binary rewards)
    """
    rewards = []
    
    for completion, ground_truth in zip(completions, ground_truths):
        # Prepare problem dict for grader
        problem = {'solution': ground_truth}
        
        # Compute reward using existing grader
        # r1_zero_reward_fn expects lists
        reward_value = r1_zero_reward_fn([problem], [completion])
        
        # Extract scalar reward
        if isinstance(reward_value, list):
            reward_value = reward_value[0] if reward_value else 0.0
        
        rewards.append(float(reward_value))
    
    return rewards


def batch_compute_rewards(
    prompts: List[str],
    completions: List[str],
    ground_truths: List[str],
    group_size: int = 8,
) -> List[float]:
    """Compute rewards for batched group-based sampling.
    
    For GRPO, we generate K completions per prompt and compute rewards
    for all completions.
    
    Args:
        prompts: List of prompts (length N)
        completions: List of completions (length N * group_size)
        ground_truths: List of ground truths (length N, repeated for each group)
        group_size: Number of completions per prompt
        
    Returns:
        List of rewards (length N * group_size)
    """
    # Expand ground truths to match completions
    # Each prompt has group_size completions
    expanded_ground_truths = []
    for i in range(len(prompts)):
        gt = ground_truths[i]
        expanded_ground_truths.extend([gt] * group_size)
    
    assert len(completions) == len(expanded_ground_truths), \
        f"Mismatch: {len(completions)} completions vs {len(expanded_ground_truths)} ground truths"
    
    # Compute rewards for all completions
    rewards = compute_math_rewards(completions, expanded_ground_truths)
    
    return rewards
