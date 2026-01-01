#!/bin/bash
# Launch script for 2× H100 FSDP GRPO training
#
# Usage:
#   bash step3_GRPO/launch.sh

set -e

echo "================================="
echo "2× H100 FSDP GRPO Training Launch"
echo "================================="

# Set NCCL environment variables for NVLink optimization
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Verify CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA not available?"
    exit 1
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Navigate to step3_GRPO directory
cd "$(dirname "$0")"

# Launch distributed training with torchrun
echo "Launching training with torchrun..."
echo "  - Number of GPUs: 2"
echo "  - Config: configs/fsdp_2gpu.yaml"
echo ""

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_grpo_fsdp.py \
    --config configs/fsdp_2gpu.yaml

echo ""
echo "Training completed!"
