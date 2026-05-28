$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "experiments\MemFCAE_mahala_corrected"
Set-Location $runDir

python train.py `
  --dataset=DCASE2025T2ToyRCCar `
  -e `
  '-tag=id(0_)' `
  --use_ids 0 `
  --score MAHALA `
  --test_only `
  --mono=True
