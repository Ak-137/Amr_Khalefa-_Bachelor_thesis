<#
Stage eval CSVs and run the official DCASE2025 Task 2 evaluator for any experiment.

Auto-discovers results/<ExperimentName>/eval_data/*_<Score> (e.g. latent_mahalanobis_MAHALA).

  powershell -File scripts\evaluation\evaluate_experiment.ps1 -ExperimentName latent_mahalanobis
  powershell -File scripts\evaluation\evaluate_experiment.ps1 -ExperimentName baseline_run -Score MSE
  powershell -File scripts\evaluation\evaluate_experiment.ps1 -ExperimentName conditionalAE -ExportDir conditionalAE

Run from project root or from scripts/evaluation.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentName,
    [string]$Score = "MAHALA",
    [string]$ExportDir = "",
    [string]$SystemName = "",
    [string]$TeamName = "thesis"
)

$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
. (Join-Path $scriptDir "config.ps1")

$scoreSuffix = "_$Score"
$evalDataRoot = Join-Path $RepoRoot "results\$ExperimentName\eval_data"

if (-not (Test-Path $evalDataRoot)) {
    throw "Eval data root not found: $evalDataRoot`nRun test with -e for this experiment first."
}

if (-not $ExportDir) {
    $matchingFolders = @(Get-ChildItem -Path $evalDataRoot -Directory |
        Where-Object { $_.Name -like "*$scoreSuffix" })

    if ($matchingFolders.Count -eq 0) {
        $available = @(Get-ChildItem -Path $evalDataRoot -Directory | ForEach-Object { $_.Name })
        $hint = if ($available.Count -gt 0) { "`nAvailable: $($available -join ', ')" } else { "" }
        throw "No folder matching *$scoreSuffix under $evalDataRoot.$hint"
    }

    if ($matchingFolders.Count -gt 1) {
        Write-Host "Multiple folders match *$scoreSuffix under $evalDataRoot :"
        foreach ($f in $matchingFolders) {
            Write-Host "  $($f.Name)"
        }
        throw "Pass -ExportDir manually (prefix before $scoreSuffix), e.g. -ExportDir $($matchingFolders[0].Name.Substring(0, $matchingFolders[0].Name.Length - $scoreSuffix.Length))"
    }

    $folderName = $matchingFolders[0].Name
    $ExportDir = $folderName.Substring(0, $folderName.Length - $scoreSuffix.Length)
}

$srcDir = Get-BaselineEvalCsvDir -ExperimentName $ExperimentName -Score $Score -ExportDir $ExportDir
if (-not (Test-Path $srcDir)) {
    throw "Eval CSV folder not found: $srcDir`nCheck -ExportDir (inferred: $ExportDir) and -Score."
}

Write-Host "Found eval CSV folder: $srcDir"
Write-Host "Using ExportDir: $ExportDir"

if (-not $SystemName) {
    $SystemName = "${ExperimentName}_${Score}_seed$DefaultSeed"
}
Write-Host "SystemName: $SystemName"

Write-Host ""
Write-Host "=== Staging eval CSVs ==="
& (Join-Path $scriptDir "stage_baseline_csvs.ps1") `
    -ExperimentName $ExperimentName `
    -Score $Score `
    -ExportDir $ExportDir `
    -SystemName $SystemName `
    -TeamName $TeamName
if ($null -ne $LASTEXITCODE -and 0 -ne $LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Staged files ==="
$stagedDir = Join-Path $EvaluatorTeamsRoot "$TeamName\$SystemName"
Write-Host "Staged to: $stagedDir"

Write-Host ""
Write-Host "=== Running official evaluator ==="
& (Join-Path $scriptDir "run_official_evaluator.ps1") `
    -SystemName $SystemName `
    -ExperimentName $ExperimentName `
    -TeamName $TeamName
if ($null -ne $LASTEXITCODE -and 0 -ne $LASTEXITCODE) { exit $LASTEXITCODE }

$resultCsv = Join-Path $EvaluatorRoot "teams_result\${SystemName}_result.csv"
$archivedCsv = Join-Path (Get-ExperimentEvalArchiveDir $ExperimentName) "official_metrics\${SystemName}_result.csv"

Write-Host ""
if (Test-Path $resultCsv) {
    Write-Host "Final result CSV: $resultCsv"
} else {
    Write-Warning "Result CSV not found at expected path: $resultCsv"
}
if (Test-Path $archivedCsv) {
    Write-Host "Archived copy: $archivedCsv"
}
