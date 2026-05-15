<#
End-to-end: stage baseline eval CSVs -> run official evaluator -> archive metrics.

Prerequisites:
  1. Trained model in experiments/baseline_run
  2. Ran test_only with -e for all eval machine types (or subset — evaluator needs ALL 8)
  3. .venv_eval installed

Example (after baseline test -e):
  powershell -File scripts\evaluation\run_full_eval_pipeline.ps1 -SystemName baseline_mse_seed13711
#>
param(
    [string]$ExperimentName = "baseline_run",
    [string]$Score = "MSE",
    [string]$SystemName = "baseline_mse_seed13711",
    [string]$TeamName = "thesis",
    [switch]$SmokeTestOnly
)

$scriptDir = $PSScriptRoot
if ($SmokeTestOnly) {
    & (Join-Path $scriptDir "create_dummy_submission.ps1") -SystemName "dummy_smoke_test" -TeamName $TeamName
    & (Join-Path $scriptDir "run_official_evaluator.ps1") -SystemName "dummy_smoke_test" -ExperimentName $ExperimentName
    exit $LASTEXITCODE
}

& (Join-Path $scriptDir "stage_baseline_csvs.ps1") -ExperimentName $ExperimentName -Score $Score -SystemName $SystemName -TeamName $TeamName
& (Join-Path $scriptDir "run_official_evaluator.ps1") -SystemName $SystemName -ExperimentName $ExperimentName
