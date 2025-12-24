# Dual-GPU SFT Training Pipeline

Clean, modular implementation of supervised fine-tuning for math problem solving with separate training and evaluation processes on dual GPUs.

**Based on**: [Stanford CS336 Assignment 5: Alignment](https://github.com/stanford-cs336/assignment5-alignment)

## Architecture

```
GPU 0 (Training)          Queue          GPU 1 (Evaluation)
─────────────────────────────────────────────────────────
Training Loop      ────────────▶      vLLM Instance
  ↓                                      ↓
Checkpoint Save    ─ push path ──▶    Load Checkpoint
  ↓                (non-blocking)      ↓
Continue                               Run Validation
                                        ↓
                                      Log Metrics
```

## Features

- **Dual-GPU Architecture**: Training on GPU 0, continuous evaluation on GPU 1
- **Non-blocking Evaluation**: Training never waits for evaluation
- **YAML Configuration**: All parameters in `config/sft_config.yaml`
- **vLLM Integration**: Fast batch inference for evaluation
- **W&B Logging**: Comprehensive metrics tracking
- **Clean Modular Code**: Production-ready, testable components

## Quick Start

```bash
# Run dual-GPU training
python run_dual_gpu_training.py
```

## Configuration

Edit `config/sft_config.yaml` to customize:

```yaml
training:
  learning_rate: 2.0e-5
  eval_every: 500
  device: "cuda:0"

evaluation:
  device: "cuda:1"
  num_eval_samples: 100
```

## Project Structure

```
.
├── config/
│   └── sft_config.yaml          # Training configuration
├── src/
│   ├── config_loader.py         # YAML config loading
│   ├── vllm_utils.py            # vLLM initialization & utils
│   ├── training_worker.py       # GPU 0 training loop
│   └── eval_worker.py           # GPU 1 evaluation loop
├── run_dual_gpu_training.py     # Main orchestrator
├── drgrpo_grader.py            # Math answer grading
└── prompts/rl_zero.prompt      # Prompt template
```

## How It Works

### Main Process (GPU 0)
1. Loads model and starts training loop
2. Every N steps, saves checkpoint
3. Pushes checkpoint path to queue (non-blocking)
4. Continues training without waiting

### Eval Worker (GPU 1)
1. Initializes persistent vLLM instance
2. Watches queue for checkpoint paths
3. Loads weights into vLLM
4. Runs validation on test set
5. Logs metrics to W&B
6. Repeats

### Communication
- **Multiprocessing Queue** (maxsize=2)
- Non-blocking push from training
- Blocking get in eval worker
- Graceful shutdown via `None` signal

## Requirements

- 2 GPUs (CUDA-capable)
- Python ≥3.10
- Dependencies in `pyproject.toml`

## Reference

Implementation based on Stanford CS336 Assignment 5:
- **Repository**: https://github.com/stanford-cs336/assignment5-alignment
- **Grader**: Dr. GRPO math grading system
- **Dataset**: MATH (Hendrycks et al. 2021)
