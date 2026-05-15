<#
.SYNOPSIS
  Copy baseline_original into experiments/<ExperimentName> (excluding .git).

.DESCRIPTION
  baseline_original is never modified. Each experiment is an independent copy
  so checkpoints, logs, and local edits cannot corrupt other runs.

.PARAMETER ExperimentName
  Folder name under experiments/, e.g. exp_mahala_v2

.EXAMPLE
  .\scripts\new_experiment_from_baseline.ps1 -ExperimentName exp_mahala_v2
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentName
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root "baseline_original"))) {
    Write-Error "baseline_original not found next to scripts. Expected repo root: $root"
}

$src = Join-Path $root "baseline_original"
$dst = Join-Path (Join-Path $root "experiments") $ExperimentName

if (Test-Path $dst) {
    Write-Error "Target already exists: $dst"
}

New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
robocopy $src $dst /E /XD .git /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

$yaml = Join-Path $dst "baseline.yaml"
if (-not (Test-Path $yaml)) {
    Write-Error "Copy failed: baseline.yaml missing in $dst"
}

# Point new experiment at shared data and isolated results (edit paths if you rename folders).
$content = Get-Content $yaml -Raw
$content = $content -replace '(?m)^--dataset_directory:.*$', '--dataset_directory: ../../datasets/dcase2025'
$content = $content -replace '(?m)^--result_directory:.*$', "--result_directory: ../../results/$ExperimentName"
Set-Content -Path $yaml -Value $content -NoNewline

Write-Host "Created experiment at: $dst"
Write-Host "Updated dataset + result paths in baseline.yaml for this experiment name."
