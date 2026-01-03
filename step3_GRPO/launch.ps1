# PowerShell Launch Script for 2× GPU FSDP GRPO Training
# Usage: .\step3_GRPO\launch.ps1 [-Test]

param(
    [switch]$Test = $false
)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "2× GPU FSDP GRPO Training Launch" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Activate virtual environment if it exists
$VenvPath = Join-Path $ProjectRoot ".venv"
if (Test-Path $VenvPath) {
    Write-Host "Activating virtual environment: $VenvPath" -ForegroundColor Yellow
    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        & $ActivateScript
    }
} else {
    Write-Host "WARNING: Virtual environment not found at $VenvPath" -ForegroundColor Yellow
    Write-Host "Proceeding without virtual environment..." -ForegroundColor Yellow
}

# Set NCCL environment variables for NVLink optimization
$env:NCCL_DEBUG = "INFO"
$env:NCCL_IB_DISABLE = "1"
$env:NCCL_P2P_LEVEL = "NVL"

# Verify CUDA is available
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: nvidia-smi not found. CUDA not available?" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "GPU Status:" -ForegroundColor Green
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
Write-Host ""

# Navigate to step3_GRPO directory
Set-Location $ScriptDir

# Build torchrun command
Write-Host "Launching training with torchrun..." -ForegroundColor Green
Write-Host "  - Number of GPUs: 2"
Write-Host "  - Launcher: step3_grpo.py"
Write-Host ""

# Check if torchrun is available
$TorchrunCmd = Get-Command torchrun -ErrorAction SilentlyContinue
if (-not $TorchrunCmd) {
    # Try python -m torch.distributed.run
    try {
        python -m torch.distributed.run --help | Out-Null
        $UsePythonModule = $true
    } catch {
        Write-Host "ERROR: torchrun not found. Please install PyTorch." -ForegroundColor Red
        exit 1
    }
}

# Launch distributed training
$LaunchArgs = @(
    "--nproc_per_node=2",
    "--nnodes=1",
    "--node_rank=0",
    "--master_addr=localhost",
    "--master_port=29500",
    "step3_grpo.py"

if ($TestModeArg) {
    $LaunchArgs += $TestModeArg
}

if ($UsePythonModule) {
    & python -m torch.distributed.run @LaunchArgs
} else {
    & torchrun @LaunchArgs
}

# Check exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "WARNING: Training command failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
