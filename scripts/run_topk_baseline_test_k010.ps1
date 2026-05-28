# PowerShell script to run TopK_Baseline test_only for k010 (0.10 ratio)
cd experiments/TopK_Baseline
python train.py --dataset=DCASE2025T2ToyRCCar -e -tag "id(0_)" --use_ids 0 --mono=True --score MAHALA --test_only --use_topk_score True --topk_ratio 0.10
