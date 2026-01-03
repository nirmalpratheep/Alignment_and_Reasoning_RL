"""vLLM-based rollout engine for GRPO training.

This module provides a high-throughput generation engine using vLLM for the
rollout phase of GRPO training. It handles:
- vLLM initialization and lifecycle management
- Loading weights from FSDP checkpoints
- Batch generation with continuous batching and PagedAttention
- GPU memory management
"""
import os
import torch
from vllm import LLM, SamplingParams
from typing import List, Optional
from pathlib import Path
import time

from step3_GRPO.train.fsdp_utils import print_rank_0


class VLLMRolloutEngine:
    """Manages vLLM engine for high-throughput rollout generation.
    
    This class provides an interface between FSDP training and vLLM inference.
    It handles loading updated model weights from checkpoints and generating
    completions with optimized batching.
    
    Key Features:
    - Lazy initialization (only creates vLLM engine when needed)
    - Checkpoint-based weight loading for FSDP compatibility
    - Configurable GPU assignment and memory limits
    - Automatic cleanup to prevent memory leaks
    
    Example:
        >>> engine = VLLMRolloutEngine(config, gpu_ids=[1], temp_dir="/tmp")
        >>> engine.load_from_checkpoint("checkpoint/step_100")
        >>> completions = engine.generate_batch(prompts, sampling_params)
        >>> engine.cleanup()
    """
    
    def __init__(
        self, 
        config, 
        gpu_ids: List[int],
        temp_dir: str = "results/grpo_temp_checkpoints"
    ):
        """Initialize VLLMRolloutEngine.
        
        Args:
            config: Configuration object with vLLM settings
            gpu_ids: List of GPU IDs to use for vLLM (e.g., [1] for single GPU)
            temp_dir: Directory for temporary checkpoints
        """
        self.config = config
        self.gpu_ids = gpu_ids
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # vLLM engine (initialized lazily)
        self.llm: Optional[LLM] = None
        self.current_checkpoint: Optional[str] = None
        
        # Extract vLLM config
        self.tensor_parallel_size = len(gpu_ids)
        self.gpu_memory_utilization = config.grpo.get('gpu_memory_utilization', 0.85)
        self.max_model_len = config.grpo.get('max_model_len', 2048)
        
        print_rank_0(f"✓ VLLMRolloutEngine initialized:")
        print_rank_0(f"  - GPU IDs: {gpu_ids}")
        print_rank_0(f"  - Tensor parallel size: {self.tensor_parallel_size}")
        print_rank_0(f"  - GPU memory utilization: {self.gpu_memory_utilization}")
        print_rank_0(f"  - Max model length: {self.max_model_len}")
        
    def load_from_checkpoint(self, checkpoint_path: str, force_reload: bool = False):
        """Load or reload vLLM engine with weights from checkpoint.
        
        If the engine is already initialized with the same checkpoint, this is a no-op
        unless force_reload=True. This avoids unnecessary reloading.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            force_reload: Force reload even if checkpoint hasn't changed
        """
        checkpoint_path = str(checkpoint_path)
        
        # Skip if already loaded (unless force reload)
        if not force_reload and self.current_checkpoint == checkpoint_path:
            print_rank_0(f"  vLLM engine already loaded with {checkpoint_path}, skipping reload")
            return
        
        print_rank_0(f"  Loading vLLM engine from: {checkpoint_path}")
        load_start = time.time()
        
        # Set CUDA_VISIBLE_DEVICES for vLLM
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
        
        # Cleanup existing engine
        if self.llm is not None:
            print_rank_0("  Cleaning up previous vLLM engine...")
            del self.llm
            torch.cuda.empty_cache()
        
        # Initialize new vLLM engine
        # Note: vLLM will automatically use all visible GPUs
        self.llm = LLM(
            model=checkpoint_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            swap_space=self.config.grpo.get('swap_space', 4),  # GB of CPU swap
            enforce_eager=self.config.grpo.get('enforce_eager', False),  # Disable CUDA graph
        )
        
        self.current_checkpoint = checkpoint_path
        load_time = time.time() - load_start
        print_rank_0(f"  ✓ vLLM engine loaded ({load_time:.1f}s)")
        
    def generate_batch(
        self, 
        prompts: List[str], 
        sampling_params: SamplingParams
    ) -> List[str]:
        """Generate completions using vLLM continuous batching.
        
        This uses vLLM's optimized batching with PagedAttention for maximum
        throughput. Unlike HuggingFace generate(), this automatically handles
        dynamic batching without manual chunking.
        
        Args:
            prompts: List of prompt strings
            sampling_params: vLLM SamplingParams object
            
        Returns:
            List of generated completion strings (same order as prompts)
            
        Raises:
            RuntimeError: If vLLM engine not initialized (call load_from_checkpoint first)
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized. Call load_from_checkpoint() first.")
        
        print_rank_0(f"  Generating {len(prompts)} completions with vLLM...")
        gen_start = time.time()
        
        # vLLM handles batching automatically with continuous batching
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text from outputs
        completions = [output.outputs[0].text for output in outputs]
        
        gen_time = time.time() - gen_start
        throughput = len(prompts) / gen_time
        print_rank_0(f"  ✓ Generation complete ({gen_time:.1f}s, {throughput:.1f} completions/s)")
        
        return completions
    
    def cleanup(self):
        """Free GPU memory occupied by vLLM.
        
        Call this when done with rollouts to release GPU resources back to FSDP.
        """
        if self.llm is not None:
            print_rank_0("Cleaning up vLLM engine...")
            del self.llm
            self.llm = None
            self.current_checkpoint = None
            torch.cuda.empty_cache()
            print_rank_0("✓ vLLM engine cleaned up")


def create_sampling_params(config) -> SamplingParams:
    """Create vLLM SamplingParams from GRPO config.
    
    Args:
        config: Configuration object with grpo.* settings
        
    Returns:
        SamplingParams configured for GRPO rollout
    """
    return SamplingParams(
        temperature=config.grpo.temperature,
        max_tokens=config.grpo.max_tokens,
        min_tokens=config.grpo.get('min_tokens', 4),
        top_p=1.0,  # GRPO uses temperature sampling only
        skip_special_tokens=False,  # Keep special tokens for reward computation
    )
