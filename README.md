# DCASE Task 2 Anomalous Sound Detection Experiments

This repository contains my experiments for DCASE Task 2 anomalous sound detection. The work is based on the official DCASE baseline code, with scoring checked through the official DCASE evaluator.

The project focuses on improving the baseline autoencoder approach for machine condition monitoring. The main improvements explored in this repository are:

- ConvAE-based model changes
- CORAL-based domain adaptation
- Top-K scoring/selection

Other experimental directions were also tested during development, but the main reproducible results are organized around the folders above.

## Repository Structure

```text
baseline_original/          Original baseline code used as reference
experiments/                Experiment folders and modified baseline variants
scripts/evaluation/         Helper scripts for staging CSVs and running the official evaluator
evaluator_repo/             Official DCASE evaluator workspace
results/                    Generated result files
evaluation_runs/            Evaluator outputs and staged submissions
logs/                       Training and test logs
datasets/                   Local datasets
```

Important experiment folders include:

```text
experiments/baseline_run/
experiments/convAE/
experiments/convAE_8/
experiments/CORAL_Res/
experiments/TopK_Baseline/
```

## Training and Testing

Run training and testing from inside the experiment folder you want to use.

Example:

```powershell
cd experiments\convAE
```

Train a model:

```powershell
python train.py --dataset=DCASE2025T2ToyRCCar --eval -tag "id(0_)" --use_ids 0 --mono=True --train_only
```

Run testing with Mahalanobis scoring:

```powershell
python train.py --dataset=DCASE2025T2ToyRCCar -e '-tag=id(0_)' --use_ids 0 --score MAHALA --test_only --mono=True
```

Notes:

- Replace the experiment folder with the specific variant you want to run.
- The commands above target `DCASE2025T2ToyRCCar` and section ID `0`.
- `--score MAHALA` can be changed when running experiments that use another supported score, such as `MSE`.

## Official Evaluator Workflow

The official evaluator is run from:

```powershell
scripts\evaluation
```

First, stage the baseline CSV files into the evaluator format:

```powershell
powershell -ExecutionPolicy Bypass -File .\stage_baseline_csvs.ps1 `
  -ExperimentName <experiment_name> `
  -Score <MSE_or_MAHALA> `
  -ExportDir <experiment_directory> `
  -SystemName <system_name>
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\stage_baseline_csvs.ps1 `
  -ExperimentName augmentation_baseline `
  -Score MAHALA `
  -ExportDir augmentation_baseline `
  -SystemName augmentation_seed13711
```

Then run the official evaluator:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_official_evaluator.ps1 `
  -ExperimentName <experiment_name> `
  -SystemName <system_name>
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_official_evaluator.ps1 `
  -SystemName augmentation_seed13711 `
  -ExperimentName augmentation_baseline
```

## Outputs

Training, testing, and evaluation generate files under folders such as:

```text
models/
results/
logs/
evaluation_runs/
evaluator_repo/teams/
evaluator_repo/teams_result/
evaluator_repo/teams_additional_result/
```

These folders can become large because they may contain model checkpoints, prediction CSVs, logs, and official evaluator outputs.

## Acknowledgements

This project builds on the official DCASE Task 2 baseline implementation and uses the official DCASE evaluator for final scoring.
