<#
ToyRCCar in DCASE2025 Task 2 lives under the *evaluation* split (eval_data), not dev_data.
This matches tools/01_train_legacy.sh for DCASE2025T2 with -e.
Run from repo root: .\scripts\run_baseline_toyrccar_train.ps1
Cwd is forced to experiments/baseline_run so baseline.yaml and imports resolve.
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "experiments\baseline_run"
Set-Location $runDir

python train.py `
  --dataset=DCASE2025T2ToyRCCar `
  -e `
  '-tag=id(0_)' `
  --use_ids 0 `
  --train_only `
  --mono=True
