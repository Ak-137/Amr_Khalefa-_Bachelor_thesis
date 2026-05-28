# Walkthrough: Top-K Anomaly Score Aggregation (TopK_Baseline)

This experiment implements and evaluates a Top-K anomaly score aggregation method inside the standard AutoEncoder baseline during test-time scoring.

## 1. Why Top-K Was Added

The original baseline aggregates segment-level or frame-level anomaly scores into a single file-level score by computing the arithmetic mean over all frames in the file. However, this averaging process can dilute short anomalous regions or transient sound events. By utilizing a Top-K score aggregation scheme instead, the file-level anomaly score is calculated using only the $K$ highest frame-level scores, emphasizing peak anomaly segments and improving target classification performance.

## 2. Frame-Level Scores Availability Audit

Prior to implementing this change, the baseline scoring and test pipeline was audited with the following findings:
- **Availability:** Yes, frame-level anomaly scores are fully available as intermediate 1D PyTorch tensors right before they are reduced using the `self.loss_reduction()` method (for `MAHALA` scoring) or the `.mean()` method (for `MSE` scoring).
- **Modification needed:** Yes, we modified the file-level reduction point to conditionally select the $K$ highest frame scores using `torch.topk(..., largest=True)` instead of executing the uniform reduction across all frames.

## 3. Exact Aggregation Formula

Given frame/segment anomaly scores for one audio file (say, $\mathbf{S} = [s_1, s_2, \dots, s_N]$):

If `--use_topk_score` is `False`:
$$\text{file\_score} = \frac{1}{N} \sum_{i=1}^N s_i$$

If `--use_topk_score` is `True`:
$$K = \max\left(1, \left\lceil \text{topk\_ratio} \times N \right\rceil\right)$$
$$\text{file\_score} = \frac{1}{K} \sum_{i=1}^K s_{(i)}$$
where $s_{(i)}$ is the $i$-th highest frame anomaly score.

## 4. Exact Files Modified

1. **[common.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor%282%29/experiments/TopK_Baseline/common.py):** Added new argparse parameters (`--use_topk_score`, `--topk_ratio`, and `--topk_mode`).
2. **[base_model.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor%282%29/experiments/TopK_Baseline/networks/base_model.py):**
   - Added Top-K safety validation checks (requires `DCASE2025T2ToyRCCar` and `test_only`).
   - Dynamically suffixes output and log directory paths with the ratio (e.g. `TopK_Baseline_k010`) to avoid overwriting results.
   - Keeps weight-loading paths pointed to `TopK_Baseline` so that weights and statistics are loaded successfully from the central location.
3. **[dcase2023t2_ae.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor%282%29/experiments/TopK_Baseline/networks/dcase2023t2_ae/dcase2023t2_ae.py):**
   - Implemented the Top-K aggregation rule for both `MAHALA` and `MSE` scoring.
   - Added diagnostic printing of the enabling status, topk_ratio, K used, number of frame scores, original mean score, and Top-K score for the first 5 processed files in each run.

## 5. Weights and Statistics Copy Source

The trained baseline weights and statistics were copied from the following paths to prevent retraining:
- **PTH Weights:**
  - Copy from: `experiments/baseline_run/models/saved_model/baseline/DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711.pth`
  - Copy to: `experiments/TopK_Baseline/models/saved_model/TopK_Baseline/DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711.pth`
- **Pickled Score Distributions:**
  - Copy from: `experiments/baseline_run/models/saved_model/baseline/score_distr_DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711_mahala.pickle`
  - Copy to: `experiments/TopK_Baseline/models/saved_model/TopK_Baseline/score_distr_DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711_mahala.pickle`
  - Copy from: `experiments/baseline_run/models/saved_model/baseline/score_distr_DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711_mse.pickle`
  - Copy to: `experiments/TopK_Baseline/models/saved_model/TopK_Baseline/score_distr_DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711_mse.pickle`
- **Checkpoint Files:**
  - Copy from: `experiments/baseline_run/models/checkpoint/baseline/DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711/*`
  - Copy to: `experiments/TopK_Baseline/models/checkpoint/TopK_Baseline/DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711/`

## 6. Manual Commands for Each Ratio

The following commands can be run from the `experiments/TopK_Baseline` directory to manually execute the Top-K scoring variant tests:

- **Top-K ratio = 0.01 ($1\%$):**
  ```bash
  python train.py --dataset=DCASE2025T2ToyRCCar -e -tag "id(0_)" --use_ids 0 --mono=True --score MAHALA --test_only --use_topk_score True --topk_ratio 0.01
  ```
- **Top-K ratio = 0.05 ($5\%$):**
  ```bash
  python train.py --dataset=DCASE2025T2ToyRCCar -e -tag "id(0_)" --use_ids 0 --mono=True --score MAHALA --test_only --use_topk_score True --topk_ratio 0.05
  ```
- **Top-K ratio = 0.10 ($10\%$):**
  ```bash
  python train.py --dataset=DCASE2025T2ToyRCCar -e -tag "id(0_)" --use_ids 0 --mono=True --score MAHALA --test_only --use_topk_score True --topk_ratio 0.10
  ```
- **Top-K ratio = 0.20 ($20\%$):**
  ```bash
  python train.py --dataset=DCASE2025T2ToyRCCar -e -tag "id(0_)" --use_ids 0 --mono=True --score MAHALA --test_only --use_topk_score True --topk_ratio 0.20
  ```

## 7. Operational Confirmations

- **Training/Testing:** No training, testing, or evaluator executions were run during this setup.
- **Untouched Baselines:** The original trained directories `experiments/baseline_run` and `experiments/baseline_original` remain completely untouched.
