<#
Resume latent_mahalanobis training after a crash during epoch 101 (covariance).

Requires checkpoint from end of epoch 100:
  experiments\latent_mahalanobis\models\checkpoint\latent_mahalanobis\DCASE2023T2-AE_DCASE2025T2ToyCar_id(0_)_seed13711\checkpoint.tar

Behavior: train(1..100) no-op (skipped via self.epoch), then runs epoch 101 only.

From repo root:
  .\venv\Scripts\Activate.ps1
  powershell -File scripts\run_latent_mahalanobis_resume_train.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ck = Join-Path $root "experiments\latent_mahalanobis\models\checkpoint\latent_mahalanobis\DCASE2023T2-AE_DCASE2025T2ToyCar_id(0_)_seed13711\checkpoint.tar"
if (-not (Test-Path $ck)) {
    Write-Error "No checkpoint found at:`n  $ck`nRun full train first (no --restart)."
}
$runDir = Join-Path $root "experiments\latent_mahalanobis"
Set-Location $runDir
& (Join-Path $root "venv\Scripts\python.exe") train.py `
  --dataset=DCASE2025T2ToyCar `
  -d `
  '-tag=id(0_)' `
  --use_ids 0 `
  --train_only `
  --mono=True `
  --restart
