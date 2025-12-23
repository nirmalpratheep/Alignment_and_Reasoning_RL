"""Memory calculation and analysis for training"""

import torch

def calculate_memory_requirements():
    """Calculate estimated memory requirements for training"""
    
    print("="*80)
    print("MEMORY ANALYSIS FOR QWEN2.5-MATH-1.5B TRAINING")
    print("="*80)
    
    # Model configuration
    model_params = 1.5e9  # 1.5 billion parameters
    batch_size = 4
    seq_length = 1024
    hidden_dim = 1024  # Typical for 1.5B model
    
    print(f"\nModel Configuration:")
    print(f"  Model: Qwen2.5-Math-1.5B")
    print(f"  Parameters: {model_params/1e9:.1f}B")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Precision: BF16 (2 bytes per parameter)")
    print(f"  Optimizer: AdamW")
    
    # Memory calculations
    print(f"\n{'='*80}")
    print("MEMORY BREAKDOWN:")
    print(f"{'='*80}")
    
    # 1. Model weights
    model_weights_bf16 = model_params * 2 / (1024**3)  # 2 bytes per param in BF16
    print(f"\n1. Model Weights (BF16):")
    print(f"   {model_params/1e9:.1f}B params × 2 bytes = {model_weights_bf16:.2f} GB")
    
    # 2. Model buffers (layer norms, etc)
    model_buffers = model_weights_bf16 * 0.1
    print(f"\n2. Model Buffers & LayerNorms:")
    print(f"   ~10% of weights = {model_buffers:.2f} GB")
    
    # 3. Forward pass activations (without gradient checkpointing)
    # Typically: batch_size × seq_length × hidden_dim × 4 (intermediate) × 2 bytes
    forward_activations = batch_size * seq_length * hidden_dim * 4 * 2 / (1024**3)
    print(f"\n3. Forward Pass Activations (without checkpointing):")
    print(f"   {batch_size} × {seq_length} × {hidden_dim} × 4 (hidden states) × 2 bytes")
    print(f"   = {forward_activations:.2f} GB")
    print(f"   WITH Gradient Checkpointing: ~{forward_activations * 0.1:.2f} GB (90% saved)")
    
    # 4. Gradients
    gradients_bf16 = model_params * 2 / (1024**3)
    print(f"\n4. Gradients (BF16):")
    print(f"   1 copy of all parameters = {gradients_bf16:.2f} GB")
    
    # 5. Optimizer State (AdamW = momentum + variance = 2 copies in FP32)
    optimizer_state = model_params * 4 * 2 / (1024**3)  # 2 copies in FP32 (4 bytes each)
    print(f"\n5. Optimizer State (AdamW with momentum + variance in FP32):")
    print(f"   2 copies of parameters × 4 bytes = {optimizer_state:.2f} GB")
    print(f"   ⚠️  THIS IS THE MAIN MEMORY CONSUMER!")
    
    # 6. Batch data
    batch_data = batch_size * seq_length * 2 * 2 / (1024**3)  # input_ids, labels, attention_mask, token_type_ids
    print(f"\n6. Batch Data (input_ids, labels, attention_mask):")
    print(f"   {batch_size} × {seq_length} × 3 × 2 bytes = {batch_data:.2f} GB")
    
    # Total
    print(f"\n{'='*80}")
    print("TOTAL MEMORY ESTIMATE:")
    print(f"{'='*80}")
    
    without_checkpointing = (
        model_weights_bf16 + 
        model_buffers + 
        forward_activations + 
        gradients_bf16 + 
        optimizer_state + 
        batch_data
    )
    
    with_checkpointing = (
        model_weights_bf16 + 
        model_buffers + 
        (forward_activations * 0.1) + 
        gradients_bf16 + 
        optimizer_state + 
        batch_data
    )
    
    print(f"\nWithout Gradient Checkpointing: {without_checkpointing:.2f} GB")
    print(f"With Gradient Checkpointing:    {with_checkpointing:.2f} GB")
    
    print(f"\nGPU Available: 22.07 GB")
    print(f"Current Usage: 21.71 GB (EXCEEDS AVAILABLE)")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("MEMORY OPTIMIZATION RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    print(f"\n1. REDUCE BATCH SIZE (Most effective):")
    for bs in [1, 2]:
        approx_mem = with_checkpointing * (bs / batch_size)
        print(f"   Batch size {bs}: ~{approx_mem:.2f} GB {'✓ Should work!' if approx_mem < 20 else '✗ Still too much'}")
    
    print(f"\n2. REDUCE SEQUENCE LENGTH:")
    for seq in [512, 768]:
        ratio = seq / seq_length
        approx_mem = with_checkpointing * ratio
        print(f"   Seq length {seq}: ~{approx_mem:.2f} GB {'✓ Should work!' if approx_mem < 20 else '✗ Still too much'}")
    
    print(f"\n3. USE 8-BIT OPTIMIZER (bnb):")
    optimizer_state_8bit = model_params * 1 / (1024**3)  # 1 byte per param
    total_with_8bit = without_checkpointing - optimizer_state + optimizer_state_8bit
    print(f"   Saves: {(optimizer_state - optimizer_state_8bit):.2f} GB")
    print(f"   Total with 8-bit: {total_with_8bit:.2f} GB {'✓ Should work!' if total_with_8bit < 20 else '✗ Still need more'}")
    
    print(f"\n4. USE LoRA (Parameter-Efficient Fine-Tuning):")
    lora_params = model_params * 0.001  # ~0.1% of original
    lora_mem = lora_params * 2 / (1024**3)
    print(f"   LoRA params: ~{lora_params/1e6:.0f}M (0.1% of original)")
    print(f"   Memory for LoRA weights: ~{lora_mem:.2f} GB")
    print(f"   Much more memory efficient!")
    
    print(f"\n5. COMBINATION APPROACH (RECOMMENDED):")
    print(f"   • Batch size: 1")
    print(f"   • Sequence length: 512")
    print(f"   • Gradient checkpointing: Enabled ✓")
    print(f"   • BF16: Enabled ✓")
    print(f"   • 8-bit optimizer: Enable")
    combo_mem = (
        model_weights_bf16 + 
        model_buffers + 
        (forward_activations * 0.1 * 0.5) +  # 50% of original seq
        gradients_bf16 + 
        optimizer_state_8bit + 
        (batch_data * 0.25)  # 1/4 of batch_data for BS=1, seq=512
    )
    print(f"   Estimated memory: ~{combo_mem:.2f} GB ✓")


if __name__ == "__main__":
    calculate_memory_requirements()
    
    print(f"\n{'='*80}")
    print("WHICH SOLUTION TO IMPLEMENT?")
    print(f"{'='*80}")
    print(f"""
IMMEDIATE FIX (Easiest):
  1. Reduce batch size from 4 to 1
  2. Reduce max_length from 1024 to 512
  → Should immediately free ~40% memory
  
BEST SOLUTION (Recommended):
  1. Use 8-bit optimizer (bitsandbytes)
  2. Reduce batch size to 1-2
  3. Reduce sequence length to 512
  → Saves ~6GB from optimizer state alone
  
FUTURE OPTIMIZATION:
  1. Implement LoRA for parameter-efficient fine-tuning
  2. Much smaller model modifications
  3. 10x less memory usage
""")
