"""Group-relative advantage computation for GRPO."""
import torch
from typing import Tuple


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
    use_std_normalization: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Compute group-relative advantages for GRPO.
    
    GRPO computes advantages by comparing rewards within each group
    (K completions per prompt) rather than using a value network.
    
    Formula: A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
    
    Args:
        rewards: Tensor of shape (batch_size * group_size,)
        group_size: Number of completions per prompt
        eps: Small constant for numerical stability
        use_std_normalization: Whether to normalize by std
        
    Returns:
        advantages: Tensor of shape (batch_size * group_size,)
        stats: Dictionary with advantage statistics
    """
    batch_size = len(rewards) // group_size
    
    # Reshape to (batch_size, group_size)
    rewards_grouped = rewards.view(batch_size, group_size)
    
    # Compute group statistics
    group_means = rewards_grouped.mean(dim=1, keepdim=True)  # (batch_size, 1)
    group_stds = rewards_grouped.std(dim=1, keepdim=True)    # (batch_size, 1)
    
    # Compute advantages
    advantages_grouped = rewards_grouped - group_means
    
    if use_std_normalization:
        # Normalize by std (with eps for stability)
        advantages_grouped = advantages_grouped / (group_stds + eps)
    
    # Flatten back to (batch_size * group_size,)
    advantages = advantages_grouped.view(-1)
    
    # Compute statistics for logging
    stats = {
        'advantage_mean': advantages.mean().item(),
        'advantage_std': advantages.std().item(),
        'advantage_max': advantages.max().item(),
        'advantage_min': advantages.min().item(),
        'reward_mean': rewards.mean().item(),
        'reward_std': rewards.std().item(),
        'reward_max': rewards.max().item(),
        'reward_min': rewards.min().item(),
        'group_mean_avg': group_means.mean().item(),
        'group_std_avg': group_stds.mean().item(),
    }
    
    return advantages, stats


def compute_grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute GRPO policy gradient loss.
    
    GRPO loss = -mean(advantages * log_probs) + kl_coef * KL
    
    Args:
        log_probs: Log probabilities from current policy, shape (N,)
        ref_log_probs: Log probabilities from reference policy, shape (N,)
        advantages: Group-relative advantages, shape (N,)
        kl_coef: KL penalty coefficient
        
    Returns:
        loss: Scalar loss tensor
        stats: Dictionary with loss statistics
    """
    # Policy gradient loss (REINFORCE with group baseline)
    policy_loss = -(advantages * log_probs).mean()
    
    # KL divergence from reference policy (optional)
    kl_div = (log_probs - ref_log_probs).mean()
    
    # Total loss
    loss = policy_loss + kl_coef * kl_div
    
    # Compute statistics
    stats = {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'kl_divergence': kl_div.item(),
        'log_prob_mean': log_probs.mean().item(),
        'ref_log_prob_mean': ref_log_probs.mean().item(),
    }
    
    return loss, stats


def compute_entropy(log_probs: torch.Tensor, probs: torch.Tensor = None) -> torch.Tensor:
    """Compute policy entropy for exploration.
    
    Args:
        log_probs: Log probabilities
        probs: Probabilities (if None, computed from log_probs)
        
    Returns:
        entropy: Mean entropy
    """
    if probs is None:
        probs = log_probs.exp()
    
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy
