# PowerShell script to run training for CORAL_Res experiment on ToyRCCar
cd experiments/CORAL_Res
python train.py --dataset=DCASE2025T2ToyRCCar --eval -tag "id(0_)" --use_ids 0 --mono=True --train_only
