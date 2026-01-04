#!/bin/bash
# TRL GRPO Training Launch Script with vLLM Colocate Mode
#
# Usage:
#   cd ~/nirmal/Alignment_and_Reasoning_RL
#   ./step3_GRPO/launch_trl.sh [num_gpus]
#
# Examples:
#   ./step3_GRPO/launch_trl.sh      # Single GPU
#   ./step3_GRPO/launch_trl.sh 2    # 2 GPUs

# Don't use 'set -e' as it exits the shell when sourced

# Get script directory robustly (works with source and direct execution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" || SCRIPT_DIR="."
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Default settings
NUM_PROCESSES=${1:-1}
CONFIG_FILE="config/grpo_config.yaml"

echo "=============================================="
echo "  TRL GRPO Training with vLLM"
echo "=============================================="
echo "  Project: $PROJECT_ROOT"
echo "  Config: $CONFIG_FILE"
echo "  GPUs: $NUM_PROCESSES"
echo "=============================================="

# Initialize wandb
export WANDB_PROJECT="math-grpo-trl"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate the project's virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating venv..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "/lambda/nfs/nirmal/Alignment_and_Reasoning_RL/.venv/bin/activate" ]; then
    echo "Activating venv (lambda path)..."
    source "/lambda/nfs/nirmal/Alignment_and_Reasoning_RL/.venv/bin/activate"
fi

# Verify we're using the right Python
echo "Python: $(which python)"

# Launch with accelerate
if [ "$NUM_PROCESSES" -eq 1 ]; then
    echo "Running on single GPU..."
    python step3_GRPO/train_trl.py
else
    echo "Running on $NUM_PROCESSES GPUs with accelerate..."
    accelerate launch \
        --num_processes "$NUM_PROCESSES" \
        --num_machines 1 \
        --mixed_precision bf16 \
        --multi_gpu \
        step3_GRPO/train_trl.py
fi

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
