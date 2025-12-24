"""vLLM utilities for efficient inference and checkpoint loading."""
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
from typing import Optional


def init_vllm(
    model_id: str,
    device: str = "cuda:1",
    dtype: str = "float16",
    seed: int = 42,
    gpu_memory_utilization: float = 0.9
) -> LLM:
    """Initialize vLLM instance for inference.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to run on (e.g., "cuda:1")
        dtype: Data type for model weights
        seed: Random seed for reproducibility
        gpu_memory_utilization: GPU memory utilization fraction
        
    Returns:
        Initialized vLLM instance
    """
    print(f"Initializing vLLM on {device}...")
    
    # Extract GPU index from device string
    if "cuda:" in device:
        gpu_id = int(device.split(":")[-1])
        tensor_parallel_size = 1
    else:
        gpu_id = 0
        tensor_parallel_size = 1
    
    llm = LLM(
        model=model_id,
        dtype=dtype,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # Disable torch.compile to avoid memory profiling issues
    )
    
    print(f"✓ vLLM initialized on {device}")
    return llm


def create_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop_sequences: Optional[list] = None,
    include_stop_str: bool = True
) -> SamplingParams:
    """Create sampling parameters for vLLM generation.
    
    Args:
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        max_tokens: Maximum tokens to generate
        stop_sequences: List of stop sequences
        include_stop_str: Whether to include stop string in output
        
    Returns:
        SamplingParams object
    """
    if stop_sequences is None:
        stop_sequences = ["</answer>"]
    
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_sequences,
        include_stop_str_in_output=include_stop_str
    )


def load_policy_into_vllm_instance(
    policy: AutoModelForCausalLM,
    llm: LLM
) -> None:
    """Load policy model weights into vLLM instance.
    
    This function copies the weights from a HuggingFace model into a vLLM instance.
    
    Args:
        policy: HuggingFace model with trained weights
        llm: vLLM instance to load weights into
    """
    print("Loading policy weights into vLLM...")
    
    # Get the underlying model from vLLM
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    
    # Copy state dict
    with torch.no_grad():
        vllm_model.load_state_dict(policy.state_dict(), strict=False)
    
    print("✓ Policy weights loaded into vLLM")


def generate_with_vllm(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams
) -> list[str]:
    """Generate responses using vLLM.
    
    Args:
        llm: vLLM instance
        prompts: List of input prompts
        sampling_params: Sampling parameters
        
    Returns:
        List of generated responses
    """
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses
