# PowerShell launch script for 2× H100 FSDP GRPO training
#
# Usage:
#   .\step3_GRPO\launch.ps1

$ErrorActionPreference = "Stop"

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "2× H100 FSDP GRPO Training Launch" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Set NCCL environment variables for NVLink optimization
$env:NCCL_DEBUG = "INFO"
$env:NCCL_IB_DISABLE = "1"
$env:NCCL_P2P_LEVEL = "NVL"

# Verify CUDA is available
try {
    $gpuInfo = nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    Write-Host "GPU Status:" -ForegroundColor Green
    Write-Host $gpuInfo
    Write-Host ""
} catch {
    Write-Host "ERROR: nvidia-smi not found. CUDA not available?" -ForegroundColor Red
    exit 1
}

# Navigate to step3_GRPO directory
Set-Location $PSScriptRoot

# Launch distributed training with torchrun
Write-Host "Launching training with torchrun..." -ForegroundColor Yellow
Write-Host "  - Number of GPUs: 2"
Write-Host "  - Config: configs/fsdp_2gpu.yaml"
Write-Host ""

torchrun `
    --nproc_per_node=2 `
    --nnodes=1 `
    --node_rank=0 `
    --master_addr=localhost `
    --master_port=29500 `
    train_grpo_fsdp.py `
    --config configs/fsdp_2gpu.yaml

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
