<#
DCASE evaluator_repo pins scipy 1.10.1 / torch 1.13.1 (Python 3.8–3.11).
Use Python 3.9 or 3.10 — NOT 3.12 (training venv).

Run from repo root:
  powershell -ExecutionPolicy Bypass -File scripts\setup_evaluator_env.ps1

Then:
  cd evaluator_repo
  ..\.venv_eval\Scripts\Activate.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$evalRoot = Join-Path $root "evaluator_repo"
$venvDir = Join-Path $root ".venv_eval"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

$pyLauncher = $null
foreach ($ver in @("3.11", "3.10", "3.9")) {
    try {
        $null = & py "-$ver" -c "import sys; print(sys.version)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pyLauncher = $ver
            break
        }
    } catch { }
}

if (-not $pyLauncher) {
    Write-Error "Need Python 3.9, 3.10, or 3.11. Install from https://www.python.org/downloads/ (3.11 recommended)."
}

if (Test-Path $venvPython) {
    $v = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($v -eq $pyLauncher) {
        Write-Host "Reusing existing .venv_eval (Python $v)"
    } else {
        Write-Host "Removing .venv_eval (was Python $v, need $pyLauncher)..."
        Remove-Item -Recurse -Force $venvDir
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating .venv_eval with Python $pyLauncher at $venvDir ..."
    & py "-$pyLauncher" -m venv $venvDir
}

Write-Host "Installing evaluator_repo requirements (this may take several minutes) ..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $evalRoot "requirements.txt")
Write-Host "Verifying imports ..."
& $venvPython -c "import scipy, librosa, torch; print('scipy', scipy.__version__); print('torch', torch.__version__)"
Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv_eval\Scripts\Activate.ps1"
Write-Host "  cd evaluator_repo"
