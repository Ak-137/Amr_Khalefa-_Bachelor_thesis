$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "experiments\latent_mahalanobis"
Set-Location $runDir
& (Join-Path $root "venv\Scripts\python.exe") train.py `
  --dataset=DCASE2025T2ToyCar `
  -d `
  '-tag=id(0_)' `
  --use_ids 0 `
  --score MAHALA `
  --test_only `
  --mono=True
