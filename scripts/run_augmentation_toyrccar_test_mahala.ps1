<#
ToyRCCar Selective Mahalanobis test (experiments/augmentation_baseline).
Run from repo root: .\scripts\run_augmentation_toyrccar_test_mahala.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "experiments\augmentation_baseline"
Set-Location $runDir

python train.py `
  --dataset=DCASE2025T2ToyRCCar `
  -e `
  '-tag=id(0_)' `
  --use_ids 0 `
  --score MAHALA `
  --test_only `
  --mono=True
