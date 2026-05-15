<#
Creates minimal evaluator submissions for smoke-testing.
By default writes all 8 eval machine types (required by the official evaluator).

  powershell -File scripts\evaluation\create_dummy_submission.ps1
  powershell -File scripts\evaluation\create_dummy_submission.ps1 -MachineType ToyRCCar
#>
param(
    [string]$SystemName = "dummy_smoke_test",
    [string]$TeamName = "thesis",
    [string[]]$MachineTypes = @(),
    [string]$Section = "section_00"
)

. (Join-Path $PSScriptRoot "config.ps1")

if ($MachineTypes.Count -eq 0) {
    $MachineTypes = $EvalMachineTypes
}

$destDir = Join-Path $EvaluatorTeamsRoot "$TeamName\$SystemName"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

$rng = [System.Random]::new(13711)
$totalRows = 0

foreach ($MachineType in $MachineTypes) {
    $gtData = Join-Path $EvaluatorRoot "ground_truth_data\ground_truth_${MachineType}_${Section}_test.csv"
    if (-not (Test-Path $gtData)) {
        Write-Warning "Skip $MachineType (no ground truth): $gtData"
        continue
    }

    $anomalyLines = [System.Collections.Generic.List[string]]::new()
    $decisionLines = [System.Collections.Generic.List[string]]::new()

    Get-Content $gtData | ForEach-Object {
        $parts = $_ -split ","
        $anon = $parts[0]
        $score = [math]::Round($rng.NextDouble(), 6)
        $anomalyLines.Add("$anon,$score")
        $dec = if ($score -gt 0.5) { 1 } else { 0 }
        $decisionLines.Add("$anon,$dec")
    }

    $anomalyPath = Join-Path $destDir "anomaly_score_DCASE2025T2${MachineType}_${Section}_test_seed${DefaultSeed}${DefaultTag}_Eval.csv"
    $decisionPath = Join-Path $destDir "decision_result_DCASE2025T2${MachineType}_${Section}_test_seed${DefaultSeed}${DefaultTag}_Eval.csv"
    $anomalyLines | Set-Content -Path $anomalyPath -Encoding utf8
    $decisionLines | Set-Content -Path $decisionPath -Encoding utf8
    $totalRows += $anomalyLines.Count
    Write-Host "  $MachineType : $($anomalyLines.Count) rows"
}

Write-Host ""
Write-Host "Wrote dummy submission to: $destDir ($totalRows total score rows)"
Write-Host "Next: powershell -File scripts\evaluation\run_official_evaluator.ps1 -SystemName $SystemName"
