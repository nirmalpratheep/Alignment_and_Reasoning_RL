#!/bin/bash
# Launch script for 2× H100 FSDP GRPO training
#
# Usage:
#   bash step3_GRPO/launch.sh

# set -e  # Disabled to allow error handling without immediate exit

echo "================================="
echo "2× H100 FSDP GRPO Training Launch"
echo "================================="

# Get script directory (handle both relative and absolute paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: Virtual environment not found at $VENV_PATH"
    echo "Proceeding without virtual environment..."
fi

# Set NCCL environment variables for NVLink optimization
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Verify CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA not available?"
    echo "Dropping into interactive shell..."
    exec bash
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Navigate to project root (so relative paths in config work)
cd "$PROJECT_ROOT"

# Launch distributed training with torchrun
echo "Launching training with torchrun..."
echo "  - Number of GPUs: 2"
echo "  - Launcher: step3_grpo.py"
echo "  - Config: config/grpo_config.yaml"
echo ""

# Parse command line arguments for test mode
TEST_MODE=""
if [ "$1" == "--test" ]; then
    TEST_MODE="--test-mode"
    echo "  - Mode: TEST (setup verification only)"
    echo ""
fi

# Use python -m torch.distributed.run if torchrun is not available
if command -v torchrun &> /dev/null; then
    TORCHRUN_CMD="torchrun"
elif python -m torch.distributed.run --help &> /dev/null; then
    TORCHRUN_CMD="python -m torch.distributed.run"
else
    echo "ERROR: torchrun not found. Please install PyTorch."
    echo "Dropping into interactive shell..."
    exec bash
fi

$TORCHRUN_CMD \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    step3_GRPO/step3_grpo.py \
    $TEST_MODE

# Check if training failed
if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Training command failed!"
    echo "Dropping into interactive shell so you can debug..."
    exec bash
fi

echo ""
echo "Training completed!"
