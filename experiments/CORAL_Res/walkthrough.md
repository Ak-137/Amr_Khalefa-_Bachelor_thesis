# CORAL_Res Experiment Walkthrough

This document outlines the design, implementation, mathematical foundation, and manual execution instructions for the `CORAL_Res` experiment designed to perform residual-feature CORAL domain alignment before corrected Mahalanobis scoring for ToyRCCar anomaly sound detection.

---

## 1. What CORAL_Res Does

The `CORAL_Res` experiment isolates a single, key architectural improvement: **Residual Domain Adaptation via Correlation Alignment (CORAL)** combined with **Corrected Diagonal-Only Mahalanobis Distance Scoring**.

It implements the following inference pipeline:
```
input x
  → baseline FC AE reconstruction x_hat
  → residual r = x - x_hat
  → CORAL residual alignment (domain adaptation on residuals)
  → corrected Mahalanobis scoring (stable diagonal-only distance)
```

By computing and aligning the statistics of the reconstruction residuals $r$ rather than the raw inputs or latent codes, we perform domain adaptation in a space that directly represents the reconstructive capability of the baseline autoencoder.

---

## 2. Why Residual Features Instead of Latent Features?

1. **Reconstructive Representation**: For an anomaly detection autoencoder, the reconstructive residual ($x - \hat{x}$) captures exactly what the model *failed to reconstruct*. The reconstruction errors are highly domain-sensitive since target and source domains exhibit distinct acoustic profiles.
2. **Direct Downstream Target**: Mahalanobis distance scoring operates directly on the reconstruction errors to measure the anomalousness of a test sample. Therefore, aligning the covariance of the reconstruction errors directly improves the downstream metric calculation without needing backpropagation through the encoder.
3. **Architecture Preservation**: Operating on the residuals allows us to perform robust domain adaptation *without modifying the baseline FC AE architecture* or introducing complex latent alignment regularizers during training.

---

## 3. Mathematical Foundations

### A. CORAL Alignment on Reconstruction Residuals
Using training normal data, we extract source residuals $R_s$ and target residuals $R_t$:
1. **Compute Means**:
   $$\mu_s = \frac{1}{N_s}\sum_{i=1}^{N_s} r_{s, i}, \quad \mu_t = \frac{1}{N_t}\sum_{i=1}^{N_t} r_{t, i}$$
2. **Compute Covariances** (with diagonal regularization $\epsilon = 10^{-5}$):
   $$C_s = \text{cov}(R_s) + \epsilon I, \quad C_t = \text{cov}(R_t) + \epsilon I$$
3. **Whiten Source Residuals**:
   $$R_{s,\text{white}} = (R_s - \mu_s) C_s^{-1/2}$$
4. **Recolor Toward Target Covariance**:
   $$R_{s,\text{coral}} = R_{s,\text{white}} C_t^{1/2} + \mu_t$$

During scoring:
- If a sample domain is **source**, its residual $r$ is transformed using the fitted $R_{s,\text{coral}}$ formula.
- If a sample domain is **target**, its residual is kept unchanged (identity).

### B. Corrected Diagonal-Only Mahalanobis Distance Scoring
The old scoring method incorrectly computed $M = \Delta \Sigma^{-1} \Delta^T$ which introduced cross-frame correlation terms and scaled exponentially with sequence length. 
The corrected formulation calculates the distance per frame vector independently and takes the mean across the sequence:
$$\text{distance}_i = r_{aligned, i}^T \Sigma_{reg}^{-1} r_{aligned, i}$$
This is equivalent to extracting only the diagonal elements of the full matrix product:
$$\text{diag}(M) = \text{diag}(\Delta \Sigma_{reg}^{-1} \Delta^T)$$

We perform stable covariance inversion using:
- Covariance diagonal regularization $\lambda = 10^{-5}$: $\Sigma_{reg} = \Sigma + \lambda I$
- Stable pseudo-inverse via `torch.linalg.pinv`

---

## 4. Files Created or Modified in `experiments/CORAL_Res`

All modifications are strictly isolated within the `experiments/CORAL_Res` folder:

1. **[NEW] `experiments/CORAL_Res/`**: Copied full structure recursively from `baseline_original`.
2. **[MODIFY] [baseline.yaml](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor(2)/experiments/CORAL_Res/baseline.yaml)**:
   - Configured `export_dir: CORAL_Res` and `result_directory: ../../results/CORAL_Res`.
   - Set dataset to `DCASE2025T2ToyRCCar`, `eval: True`, `use_ids: 0`, and `seed: 13711`.
   - Set frame and mel parameters: `n_mels: 128`, `frames: 5`, `n_fft: 1024`, `hop_length: 512`, `mono: True`.
   - Added configuration parameters: `--use_coral_res: True`, `--coral_mode: source_to_target`, `--coral_eps: 1e-5`, `--mahala_reg_lambda: 1e-5`.
3. **[MODIFY] [common.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor(2)/experiments/CORAL_Res/common.py)**: Added argument parsing options for `--use_coral_res`, `--coral_mode`, `--coral_eps`, and `--mahala_reg_lambda`.
4. **[MODIFY] [networks/dcase2023t2_ae/network.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor(2)/experiments/CORAL_Res/networks/dcase2023t2_ae/network.py)**:
   - Added buffers inside `AENet` to store CORAL statistics (`coral_mu_s`, `coral_mu_t`, `coral_inv_sqrt_Cs`, `coral_sqrt_Ct`, `coral_fitted`) ensuring they are saved and loaded correctly as part of the state dictionary.
   - Implemented `coral_align_res` and `fit_coral_statistics` using robust real symmetric eigen-decomposition (`torch.linalg.eigh`).
5. **[MODIFY] [networks/criterion/mahala.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor(2)/experiments/CORAL_Res/networks/criterion/mahala.py)**:
   - Implemented diagonal-only Mahalanobis calculation in `mahalanobis()` using `torch.diagonal` and `torch.linalg.pinv`.
   - Integrated optional `coral_align_fn` transform.
   - Added robust parameter handling in `calc_inv_cov` to dynamically fetch `mahala_reg_lambda` and fallback gracefully to `1e-5` if missing.
6. **[MODIFY] [networks/dcase2023t2_ae/dcase2023t2_ae.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor(2)/experiments/CORAL_Res/networks/dcase2023t2_ae/dcase2023t2_ae.py)**:
   - Updated train loop to compute and fit normal training residual CORAL statistics (Phase 1) right before calculating the covariance matrices (Phase 2).
   - Applied domain adaptation to residuals when fitting covariance and calculating anomaly scores in both validation and testing.
7. **[MODIFY] [networks/base_model.py](file:///c:/Users/amrkh/OneDrive/Desktop/Bachelor(2)/experiments/CORAL_Res/networks/base_model.py)**:
   - Added safety check to halt execution with a `ValueError` if the dataset name is not `DCASE2025T2ToyRCCar`.
   - Added startup prints to report experiment metadata, configuration flags, and file paths.
   - Added warnings if testing without `MAHALA` scoring or if `use_coral_res` is disabled.
   - Added confirmation printing that no anomaly/test labels are used in CORAL parameter fitting.

---

## 5. Safe Integration Confirmations

* **No Modification to baseline_original**: All changes are strictly confined inside `experiments/CORAL_Res`. The original folder `baseline_original` was completely untouched.
* **No Existing Experiments Modified**: No changes were made to any other experiment folder.
* **No Training or Testing Executed**: No scripts, python files, or training commands were run during this setup.
* **ToyRCCar Isolation**: Verified that `DCASE2025T2ToyRCCar` is configured explicitly and validation checks are in place.
* **Zero Contamination**: Confirming that only normal training data is collected to compute mean and covariance statistics for CORAL and Mahalanobis calculations. No anomaly labels or test labels are loaded or utilized.

---

## 6. Manual Execution Instructions

Run these commands manually from the repository root:

### A. Training Command
To train the FC AE and calculate CORAL residual statistics and corrected Mahalanobis covariance parameters:
```powershell
cd experiments/CORAL_Res
python train.py --dataset=DCASE2025T2ToyRCCar --eval -tag "id(0_)" --use_ids 0 --mono=True --train_only
```

### B. Testing Command
To test using the corrected Mahalanobis scoring metric in the aligned residual space:
```powershell
cd experiments/CORAL_Res
python train.py --dataset=DCASE2025T2ToyRCCar --eval -tag "id(0_)" --use_ids 0 --score MAHALA --test_only --mono=True
```
