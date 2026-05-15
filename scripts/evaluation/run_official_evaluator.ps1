<#
Run official DCASE2025 Task 2 evaluator (unchanged Python). Uses .venv_eval (Python 3.9).

  powershell -File scripts\evaluation\run_official_evaluator.ps1 -SystemName dummy_smoke_test
  powershell -File scripts\evaluation\run_official_evaluator.ps1 -SystemName baseline_mse_seed13711 -ExperimentName baseline_run
#>
param(
    [string]$TeamName = "thesis",
    [string]$SystemName = "dummy_smoke_test",
    [string]$ExperimentName = "baseline_run",
    [int]$Seed = 13711,
    [string]$Tag = "_id(0_)",
    [switch]$OutAll
)

. (Join-Path $PSScriptRoot "config.ps1")

if (-not (Test-Path $VenvEvalPython)) {
    throw "Missing $VenvEvalPython. Run scripts\setup_evaluator_env.ps1 first."
}

$submissionDir = Join-Path $EvaluatorTeamsRoot "$TeamName\$SystemName"
if (-not (Test-Path $submissionDir)) {
    throw "Submission folder not found: $submissionDir"
}

$metricsDir = Join-Path (Get-ExperimentEvalArchiveDir $ExperimentName) "official_metrics"
New-Item -ItemType Directory -Force -Path $metricsDir | Out-Null

$evaluatorMetricsDir = Join-Path $EvaluatorRoot "teams_result"
New-Item -ItemType Directory -Force -Path $evaluatorMetricsDir | Out-Null

$outAllFlag = if ($OutAll) { "True" } else { "False" }

Push-Location $EvaluatorRoot
try {
    & $VenvEvalPython dcase2025_task2_evaluator.py `
        --teams_root_dir "./teams" `
        --result_dir "./teams_result" `
        --additional_result_dir "./teams_additional_result" `
        --dir_depth 2 `
        --seed $Seed `
        -tag $Tag `
        --out_all $outAllFlag

    if ($LASTEXITCODE -ne 0) { throw "Evaluator exited with code $LASTEXITCODE" }

    $resultCsv = Join-Path $evaluatorMetricsDir "${SystemName}_result.csv"
    if (Test-Path $resultCsv) {
        Copy-Item $resultCsv (Join-Path $metricsDir "${SystemName}_result.csv") -Force
        Write-Host "Archived metrics -> $(Join-Path $metricsDir "${SystemName}_result.csv")"
    } else {
        Write-Warning "Expected result not found: $resultCsv (evaluator may have skipped incomplete submissions)."
    }
} finally {
    Pop-Location
}

Write-Host "Done. Official evaluator outputs remain in evaluator_repo\teams_result\"
