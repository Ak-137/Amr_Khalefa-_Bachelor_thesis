# Shared paths for thesis evaluation workflow (does not modify evaluator_repo Python code).
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EvaluatorRoot = Join-Path $RepoRoot "evaluator_repo"
$VenvEvalPython = Join-Path $RepoRoot ".venv_eval\Scripts\python.exe"
$BaselineRunDir = Join-Path $RepoRoot "experiments\baseline_run"

# Thesis-side archives (predictions + official metrics + copied configs)
$EvaluationRunsRoot = Join-Path $RepoRoot "evaluation_runs"

# Staging inside official evaluator layout: teams/<TeamName>/<SystemName>/*.csv
$EvaluatorTeamsRoot = Join-Path $EvaluatorRoot "teams"
$DefaultTeamName = "thesis"
$DefaultSeed = 13711
$DefaultTag = "_id(0_)"

# Official evaluator machine types (eval_data only — not dev ToyCar)
$EvalMachineTypes = @(
    "AutoTrash", "BandSealer", "CoffeeGrinder", "HomeCamera",
    "Polisher", "ScrewFeeder", "ToyPet", "ToyRCCar"
)

function Get-BaselineEvalCsvDir {
    param(
        [string]$ExperimentName = "baseline_run",
        [string]$Score = "MSE",
        [string]$ExportDir = "baseline"
    )
    Join-Path $RepoRoot "results\$ExperimentName\eval_data\${ExportDir}_${Score}"
}

function Get-ExperimentEvalArchiveDir {
    param([string]$ExperimentName)
    Join-Path $EvaluationRunsRoot $ExperimentName
}
