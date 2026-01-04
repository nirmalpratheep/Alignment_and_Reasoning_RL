# Alignment and Reasoning RL - Math Reasoning Pipeline

A complete training pipeline for improving mathematical reasoning in LLMs, progressing from baseline evaluation through supervised fine-tuning to reinforcement learning.

**Based on**: [Stanford CS336 Assignment 5: Alignment](https://github.com/stanford-cs336/assignment5-alignment)

## Pipeline Overview

![Training Pipeline](C:/Users/ssuga/.gemini/antigravity/brain/5405abab-9dd4-402a-8704-3dcd4cabda32/pipeline_diagram_corrected_1767489056016.png)

## Training Pipeline Results

**Model**: Qwen 2.5 Math 1.5B  
**Dataset**: NuminaMath-CoT / MATH (5,000 samples)  
**Evaluation Format**: RL zero-shot (`<think>` reasoning + `<answer>` tags)

| Stage | Method | Accuracy | Format Accuracy | Improvement |
|-------|--------|----------|-----------------|-------------|
| **[Step 0](#step-0-baseline-evaluation)** | Baseline (Zero-shot) | 2.84% | 16.0% | - |
| **[Step 1](#step-1-hyperparameter-optimization)** | Optuna HPO | - | - | Search space |
| **[Step 2](#step-2-supervised-fine-tuning-sft)** | Custom SFT | 23.46% | 86.40% | +20.62% |
| **[Step 3](#step-3-grpo-reinforcement-learning)** | TRL GRPO + vLLM | **40.46%** | **96.72%** | **+17.00%** |

**Overall Improvement**: 2.84% → 40.46% (**+37.62%** absolute, **14.2× increase**)

---

## Step 0: Baseline Evaluation

**Purpose**: Evaluate pretrained model capabilities without any fine-tuning

### Architecture
- **Model**: Qwen 2.5 Math 1.5B (base, no instruction tuning)
- **Evaluation**: Zero-shot on MATH dataset with RL format
- **Metrics**: Accuracy, format compliance, error categorization

### Results
| Category | Count | Percentage |
|----------|-------|------------|
| Correct | 142 | **2.84%** |
| Wrong Answer | 657 | 13.14% |
| Format Failure | 4,201 | **84.02%** |

### Key Findings
- **84% format failures**: Base model doesn't follow instruction format
- **2.84% accuracy**: Expected for non-instruction-tuned model
- **Foundation established**: Identifies need for format-aware training

📄 **[Detailed README](step0_baseline/README.md)**

---

## Step 1: Hyperparameter Optimization

**Purpose**: Find optimal learning rate, batch size, and weight decay for SFT training

### Architecture
- **Framework**: Optuna with TPE (Tree-structured Parzen Estimator) Bayesian optimization
- **Pruning**: ASHA (Asynchronous Successive Halving) - aggressively prunes ~60-70% of underperforming trials
- **Dual-GPU Pipeline**:
  - **GPU 0**: Sequential trial training (one trial at a time)
  - **GPU 1**: Persistent vLLM evaluation worker (processes all  trials)
- **Model Reuse**: Transformers model loaded once, weights reloaded per trial (~27s saved/trial)
- **Early Stopping**: patience=3, min_delta=0.001
- **Search Space**:
  - Learning Rate: [5e-6, 1e-4] (log scale)
  - Batch Size: [32, 64, 128, 256] (categorical)
  - Weight Decay: [0.0, 0.1] (linear)

### Results
- **Trials**: 20 trials with TPE + ASHA pruning
- **Best Result**: Trial 4 achieved 24.00% accuracy
- **Optimal Hyperparameters**: LR=2.0e-5, Batch=32, WD=(configured)
- **W&B Project**: [math-sft-optuna-asha](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha)

### Key Features
- **Bayesian Optimization**: TPE intelligently explores promising hyperparameter regions
- **Aggressive Pruning**: ASHA stops bad trials early, focusing compute on good ones
- **Dual-GPU Efficiency**: Training and evaluation run concurrently
- **Model Weight Reuse**: Significant speedup by reusing loaded model
- **vLLM Integration**: Fast accuracy metrics during trials

📄 **[Detailed README](step1_sft_hyper/README.md)**

---

## Step 2: Supervised Fine-Tuning (SFT)

**Purpose**: Train model to follow instruction format and improve mathematical reasoning

### Architecture
- **Training Framework**: Custom PyTorch training loop (NOT TRL-based)
  - Hand-written forward-backward-optimize logic
  - Per-example training with gradient accumulation
- **Optimizer**: AdamW with L2 weight decay
- **LR Scheduler**: Linear warmup + linear decay schedule
- **Precision**: bfloat16 mixed precision
- **Memory Optimization**: Gradient checkpointing enabled
- **Dual-GPU Pipeline**:
  - **GPU 0**: Training loop (training_worker.py)
  - **GPU 1**: Persistent vLLM evaluation worker (eval_worker.py)
- **Hyperparameters** (from Step 1):
  - Learning Rate: 2.0e-5
  - Batch Size: 32 (effective: 64 with grad_accum=2)
  - Warmup: Configurable warmup steps
- **Evaluation**: Every 1000 steps on 5000 samples
- **W&B Logging**: [math-sft](https://wandb.ai/nirmalpratheep-self/math-sft)

### Results
| Metric | Baseline | Final (Step 6) | Change |
|--------|----------|----------------|--------|
| Accuracy | 2.84% | **23.46%** | **+20.62%** |
| Format Accuracy | 16.0% | **86.40%** | **+70.40%** |
| Format Failure | 84.02% | 13.60% | **-70.42%** |

### Key Achievements
- **Format compliance**: 16% → 86% format accuracy
- **Reasoning improvement**: 7× accuracy increase
- **Dual-GPU efficiency**: Concurrent training and evaluation
- **Foundation for RL**: Established stable format following

📄 **[Detailed README](step2_sft/README.md)**

---

## Step 3: GRPO (Reinforcement Learning)

**Purpose**: Further improve reasoning through policy optimization with group rewards

### Architecture
- **Framework**: HuggingFace TRL GRPOTrainer
  - Built-in GRPO algorithm implementation
  - Policy gradient optimization with group-based rewards
- **Generation**: vLLM colocate mode
  - Shares GPU memory between training and generation
  - High-throughput generation for RL rollouts
- **Reward Function**: `r1_zero_reward_fn` from utils/drgrpo_grader.py
  - Math-specific answer verification
  - Format compliance checking
- **Configuration**: `config/grpo_config.yaml`
  - Group size for group rewards
  - Temperature and sampling parameters
  - Max tokens for generation
  - Learning rate, batch size, training steps
- **Multi-GPU Support**: Uses HuggingFace Accelerate for distributed training

### Results
| Metric | Value |
|--------|-------|
| **Accuracy** | **40.46%** (2023/5000) |
| **Format Accuracy** | **96.72%** (4836/5000) |
| **Correct** | 40.46% |
| **Wrong Answer** | 56.26% |
| **Format Failure** | 3.28% |

### Key Achievements
- **+17% accuracy gain** over SFT (23.46% → 40.46%)
- **96.72% format compliance**: Near-perfect instruction following
- **14.2× total improvement** from baseline (2.84% → 40.46%)
- **vLLM efficiency**: Fast generation with memory sharing via colocate mode

📄 **[Detailed README](step3_GRPO/README.md)**

---

## Quick Start

```bash
# Install dependencies
uv sync --extra vllm  # Linux/WSL only (vLLM not supported on Windows)

# Step 0: Baseline evaluation
cd step0_baseline
python step0_baseLineModelEval.py

# Step 1: Hyperparameter search
cd step1_sft_hyper
python step1_sft_hyper.py

# Step 2: SFT training
cd step2_sft
python step2_sft.py

# Step 3: GRPO training
cd step3_GRPO
./launch_trl.sh  # Single GPU
./launch_trl.sh 2  # Multi-GPU
```

## Project Structure

```
.
├── step0_baseline/         # Baseline model evaluation
│   ├── step0_baseLineModelEval.py
│   ├── evaluate_model.py
│   └── README.md
├── step1_sft_hyper/        # Hyperparameter optimization
│   ├── step1_sft_hyper.py
│   └── README.md
├── step2_sft/              # Supervised Fine-Tuning
│   ├── step2_sft.py
│   └── README.md
├── step3_GRPO/             # GRPO Reinforcement Learning
│   ├── train_trl.py
│   ├── launch_trl.sh
│   └── README.md
├── src/                    # Core training/eval modules
│   ├── training_worker.py
│   ├── eval_worker.py
│   └── ...
├── utils/                  # Utilities (dataset, grader)
├── config/                 # Configuration files
└── prompts/                # Prompt templates
```

## Reference

This project is based on methods from Stanford CS336 Assignment 5:
- **Repository**: https://github.com/stanford-cs336/assignment5-alignment
- **Grader**: Adapted from Dr. GRPO's math grading system
- **Dataset**: MATH dataset (Hendrycks et al. 2021) / NuminaMath-CoT
- **Prompt Format**: RL zero-shot prompting with reasoning tags

