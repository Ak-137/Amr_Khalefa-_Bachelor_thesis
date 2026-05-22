<#
ToyRCCar train with mild feature noise (experiments/augmentation_baseline).
Run from repo root: .\scripts\run_augmentation_toyrccar_train.ps1
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
  --train_only `
  --mono=True
