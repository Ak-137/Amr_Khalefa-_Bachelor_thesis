# PowerShell script to run testing for CORAL_Res experiment with corrected Mahalanobis on ToyRCCar
cd experiments/CORAL_Res
python train.py --dataset=DCASE2025T2ToyRCCar --eval -tag "id(0_)" --use_ids 0 --score MAHALA --test_only --mono=True
