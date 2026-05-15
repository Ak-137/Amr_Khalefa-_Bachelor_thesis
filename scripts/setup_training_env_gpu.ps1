<#
Creates a virtual environment at repo root (default: venv/), installs PyTorch+torchvision
with CUDA 11.8 wheels, then experiments/baseline_run/requirements.txt.

Use a folder name other than .venv if OneDrive locks hidden folders; override:
  $env:VENV_NAME = "venv_gpu"
  powershell -ExecutionPolicy Bypass -File scripts\setup_training_env_gpu.ps1

Afterwards:
  .\venv\Scripts\Activate.ps1
  cd experiments\baseline_run
  python train.py ...
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvName = if ($env:VENV_NAME) { $env:VENV_NAME } else { "venv" }
$venvDir = Join-Path $root $venvName
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvDir ..."
    python -m venv $venvDir
    if (-not (Test-Path $venvPython)) {
        Write-Error "Failed to create venv. Install Python 3.10+ and ensure 'python' is on PATH."
    }
}

Write-Host "Installing PyTorch (CUDA 11.8) from download.pytorch.org (no pip self-upgrade) ..."
& $venvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu118"
Write-Host "Installing remaining requirements ..."
& $venvPython -m pip install -r (Join-Path $root "experiments\baseline_run\requirements.txt")
Write-Host "Verifying CUDA is visible to PyTorch ..."
& $venvPython -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"
Write-Host ""
Write-Host "Done. Activate with:  .\$venvName\Scripts\Activate.ps1"
Write-Host "Then:  cd experiments\baseline_run"
