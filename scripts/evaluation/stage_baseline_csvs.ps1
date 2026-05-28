<#
Copy baseline test outputs (eval_data, -e) into evaluator_repo/teams/<Team>/<System>/.
Does not modify evaluator code.

  powershell -File scripts\evaluation\stage_baseline_csvs.ps1 `
    -ExperimentName baseline_run -Score MSE -SystemName baseline_mse_seed13711
#>
param(
    [string]$ExperimentName = "baseline_run",
    [string]$Score = "MSE",
    [string]$ExportDir = "",  # default: baseline for baseline_run, latent_mahalanobis for that experiment
    [string]$SystemName = "baseline_mse_seed13711",
    [string]$TeamName = "thesis"
)

. (Join-Path $PSScriptRoot "config.ps1")

if (-not $ExportDir) {
    $ExportDir = if ($ExperimentName -eq "latent_mahalanobis") { "latent_mahalanobis" } else { "baseline" }
}

$srcDir = Get-BaselineEvalCsvDir -ExperimentName $ExperimentName -Score $Score -ExportDir $ExportDir
if (-not (Test-Path $srcDir)) {
    throw "No eval CSV folder: $srcDir`nRun baseline test with -e first, e.g. test_only on DCASE2025T2ToyRCCar."
}

$destDir = Join-Path $EvaluatorTeamsRoot "$TeamName\$SystemName"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

$files = Get-ChildItem -Path $srcDir -Filter "*_Eval.csv"
if ($files.Count -eq 0) {
    throw "No *_Eval.csv in $srcDir"
}

foreach ($f in $files) {
    $destName = $f.Name
    if ($destName -like "*id(0_coralres_fixed)*") {
        $destName = $destName -replace "id\(0_coralres_fixed\)", "id(0_)"
    }
    Copy-Item -Path $f.FullName -Destination (Join-Path $destDir $destName) -Force
    Write-Host "staged $($f.Name) -> $destName"
}

# Archive copy under evaluation_runs/
$archivePred = Join-Path (Get-ExperimentEvalArchiveDir $ExperimentName) "predictions\eval_data"
New-Item -ItemType Directory -Force -Path $archivePred | Out-Null
Copy-Item -Path "$srcDir\*_Eval.csv" -Destination $archivePred -Force

$argsJson = Join-Path $BaselineRunDir "models\checkpoint\baseline\DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_seed13711\args.json"
if (Test-Path $argsJson) {
    $archiveCfg = Join-Path (Get-ExperimentEvalArchiveDir $ExperimentName) "configs"
    New-Item -ItemType Directory -Force -Path $archiveCfg | Out-Null
    Copy-Item $argsJson (Join-Path $archiveCfg "args_eval_example.json") -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Staged $($files.Count) file(s) to: $destDir"
Write-Host "Archived predictions to: $archivePred"
