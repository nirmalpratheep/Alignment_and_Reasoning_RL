# GRPO Training of Qwen-1.5-Math Using FSDP on 2× H100

**Production-grade single-node distributed GRPO training** with FlashAttention-2, torch.compile, and H100-specific optimizations.

## 🚀 Features

- **FSDP FULL_SHARD** distributed training across 2× H100 GPUs with NVLink
- **FlashAttention-2** for memory-efficient attention computation
- **torch.compile** with max-autotune for kernel fusion
- **bfloat16** mixed precision for H100 tensor cores
- **Custom GRPO** implementation without value model (40% lower memory vs PPO)
- **Concurrent evaluation** on GPU-1 with vLLM time-slicing
- **Unified W&B logging** with comprehensive metrics

## 📋 Technical Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Base | PyTorch 2.4+ | FSDP stability + compile |
| Sharding | FSDP FULL_SHARD | Memory efficient |
| Precision | bfloat16 | Native H100 |
| Attention | FlashAttention-2 | Mandatory for H100 |
| Compiler | torch.compile | Free speed boost |
| RL | Custom GRPO | No value model |
| Eval | vLLM | Fast inference |
| Tracking | W&B | Unified logging |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Single Node (2× H100 GPUs)          │
├─────────────────────────────────────────────┤
│                                             │
│  GPU-0 (Rank 0)         GPU-1 (Rank 1)     │
│  ┌─────────────┐        ┌─────────────┐    │
│  │ FSDP Train  │◄──────►│ FSDP Train  │    │
│  │             │  NVLink│             │    │
│  │ + Logging   │        │ + Eval      │    │
│  └─────────────┘        └─────────────┘    │
│                                             │
│  Both GPUs: FULL_SHARD training             │
│  GPU-1: Time-sliced periodic evaluation     │
└─────────────────────────────────────────────┘
```

## 🎯 GRPO Advantage

GRPO (Group Relative Policy Optimization) computes advantages via **group ranking** instead of a value network:

```python
# K completions per prompt → Group-relative advantage
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

**Benefits:**
- No value model training → ~40% lower memory
- Faster iteration cycles
- More stable for math reasoning
- Simpler codebase

## 📊 W&B Metrics

### Training Metrics (Every Step, Rank 0)
- **Loss**: policy_loss, kl_divergence, entropy, grad_norm
- **Rewards**: mean, std, max, min, correct_rate  
- **Advantages**: mean, std, max, min
- **Generation**: length, tokens/sec
- **System**: GPU memory & utilization (both GPUs)
- **Optimizer**: learning_rate, global_step

### Evaluation Metrics (Periodic, Rank 1)
- **Accuracy**: GSM8K, MATH overall, per-difficulty (levels 1-5)
- **Quality**: avg_solution_length, valid_format_rate
- **Timing**: eval_time_seconds, checkpoint_step
- **Samples**: W&B table with problem/prediction/ground_truth/correct

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Requires Python 3.10+, CUDA 12.1+
pip install -e .

# Install flash-attn (may require building from source)
pip install flash-attn --no-build-isolation
```

### 2. Configure Training

Edit `step3_GRPO/configs/fsdp_2gpu.yaml`:

```yaml
fsdp:
  world_size: 2
  sharding_strategy: "FULL_SHARD"
  mixed_precision: "bfloat16"

model:
  name: "results/checkpoints/checkpoint_6000"  # Your SFT checkpoint
  attn_implementation: "flash_attention_2"
  torch_compile: true

training:
  learning_rate: 1.0e-5
  batch_size_per_gpu: 128  # 256 total
  max_steps: 200

grpo:
  group_size: 8  # Completions per prompt
  temperature: 1.0
```

### 3. Launch Training

**Linux/Mac:**
```bash
bash step3_GRPO/launch.sh
```

**Windows (PowerShell):**
```powershell
.\step3_GRPO\launch.ps1
```

**Manual (torchrun):**
```bash
torchrun --nproc_per_node=2 step3_GRPO/train_grpo_fsdp.py --config step3_GRPO/configs/fsdp_2gpu.yaml
```

### 4. Monitor Training

```bash
# W&B dashboard
wandb login
# Visit https://wandb.ai/<entity>/math-grpo-h100

# GPU utilization
watch -n 1 nvidia-smi
```

## 📁 Project Structure

```
step3_GRPO/
├── train/
│   ├── fsdp_grpo_trainer.py    # Core FSDP trainer
│   ├── rewards_math.py         # Math verification rewards
│   └── advantage_group.py      # Group-relative advantages
├── eval/
│   └── vllm_eval.py            # vLLM evaluation
├── configs/
│   └── fsdp_2gpu.yaml          # H100-optimized config
├── benchmarks/
│   └── throughput.md           # Performance metrics
├── train_grpo_fsdp.py          # Main entry point
├── launch.sh                   # Bash launcher
├── launch.ps1                  # PowerShell launcher
└── README.md                   # This file
```

## 🔬 Key Implementation Details

### FSDP Configuration

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=transformer_auto_wrap_policy(...),
    mixed_precision=MixedPrecision(bfloat16),
    forward_prefetch=True,
    limit_all_gathers=True,
    sync_module_states=True,
)
```

### FlashAttention-2 + Compile

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model = torch.compile(model, mode="max-autotune")
```

### W&B Shared Run

```python
# Rank 0: Initialize
if rank == 0:
    wandb.init(project="math-grpo-h100", name=f"fsdp-grpo-{timestamp}")
    run_id = wandb.run.id

# Broadcast to all ranks
dist.broadcast_object_list([run_id], src=0)

# Rank 1: Join same run
if rank == 1:
    wandb.init(project="math-grpo-h100", id=run_id, resume="allow")
```

## 📈 Expected Performance

- **Throughput**: >2000 tokens/sec aggregate
- **GPU Memory**: ~70-80GB per H100 (within 80GB limit)
- **Training Speed**: ~200 steps in ~2-3 hours
- **Evaluation**: ~2-3 minutes per checkpoint

See `benchmarks/throughput.md` for detailed metrics.

## 🛠️ Troubleshooting

### FlashAttention-2 Installation

```bash
# If pre-built wheel fails, build from source
pip install packaging ninja
pip install flash-attn --no-build-isolation
```

### NCCL Communication Errors

```bash
# Check NVLink status
nvidia-smi topo -m

# Set NCCL env vars (already in launch scripts)
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
```

### OOM on Evaluation

Reduce vLLM batch size in config:

```yaml
evaluation:
  batch_size: 160  # Reduce from 320
  num_samples: 500  # Reduce from 1000
```

## 📚 References

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/...)
- **FlashAttention-2**: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2307.08691)
- **FSDP**: [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@misc{grpo-qwen-h100,
  title={GRPO Training of Qwen-1.5-Math Using FSDP on 2× H100},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/...}}
}
```

## 📄 License

MIT License - see LICENSE file for details.
