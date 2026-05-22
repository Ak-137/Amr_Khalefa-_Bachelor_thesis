import os
import csv
import torch
import numpy as np
from sklearn import metrics

# Paths
BASE_DIR = r"c:\Users\amrkh\OneDrive\Desktop\Bachelor(2)"
LIST_CSV_PATH = os.path.join(BASE_DIR, "experiments", "augmentation_baseline", "datasets", "eval_data_list_2025.csv")

BASELINE_PRED_PATH = os.path.join(BASE_DIR, "evaluation_runs", "baseline_run", "predictions", "eval_data", "anomaly_score_DCASE2025T2ToyRCCar_section_00_test_seed13711_id(0_)_Eval.csv")
AUG_PRED_PATH = os.path.join(BASE_DIR, "evaluation_runs", "augmentation_baseline", "predictions", "eval_data", "anomaly_score_DCASE2025T2ToyRCCar_section_00_test_seed13711_id(0_)_Eval.csv")

BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "experiments", "baseline_run", "models", "saved_model", "baseline", "DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711.pth")
AUG_MODEL_PATH = os.path.join(BASE_DIR, "experiments", "augmentation_baseline", "models", "saved_model", "augmentation_baseline", "DCASE2023T2-AE_DCASE2025T2ToyRCCar_id(0_)_Eval_seed13711.pth")

# 1. Parse eval_data_list_2025.csv to get ToyRCCar mappings
print("Parsing dataset list mapping...")
toy_rc_car_map = {}
current_machine = None

with open(LIST_CSV_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            current_machine = line
        else:
            if current_machine == "ToyRCCar":
                short, long_name = line.split(",")
                domain = "target" if "target" in long_name else "source"
                label = "anomaly" if "anomaly" in long_name else "normal"
                toy_rc_car_map[short] = {
                    "domain": domain,
                    "label": label,
                    "long_name": long_name
                }

print(f"Found {len(toy_rc_car_map)} ToyRCCar mapped files.")

# 2. Load predictions
def load_predictions(path):
    preds = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                short, score = row[0], float(row[1])
                preds[short] = score
    return preds

baseline_preds = load_predictions(BASELINE_PRED_PATH)
aug_preds = load_predictions(AUG_PRED_PATH)

print(f"Loaded {len(baseline_preds)} baseline and {len(aug_preds)} augmentation predictions.")

# 3. Align and analyze
def analyze_experiment(preds, name):
    print(f"\n=================== {name} ANALYSIS ===================")
    y_true = []
    y_pred = []
    
    # split lists
    src_normal = []
    src_anomaly = []
    tgt_normal = []
    tgt_anomaly = []
    
    for short, info in toy_rc_car_map.items():
        if short not in preds:
            continue
        score = preds[short]
        domain = info["domain"]
        label = info["label"]
        
        is_anomaly = 1 if label == "anomaly" else 0
        y_true.append(is_anomaly)
        y_pred.append(score)
        
        if domain == "source":
            if label == "anomaly":
                src_anomaly.append(score)
            else:
                src_normal.append(score)
        else: # target
            if label == "anomaly":
                tgt_anomaly.append(score)
            else:
                tgt_normal.append(score)
                
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate AUCs
    # AUC all
    auc_all = metrics.roc_auc_score(y_true, y_pred)
    # AUC source (requires only source normals and all anomalies)
    # wait, official evaluator code does this:
    # y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="source" or y_true[idx]==1]
    # y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="source" or y_true[idx]==1]
    # Let's reproduce exactly:
    y_true_s_auc = []
    y_pred_s_auc = []
    y_true_t_auc = []
    y_pred_t_auc = []
    for short, info in toy_rc_car_map.items():
        if short not in preds: continue
        score = preds[short]
        domain = info["domain"]
        label = info["label"]
        is_anomaly = 1 if label == "anomaly" else 0
        
        if domain == "source" or is_anomaly == 1:
            y_true_s_auc.append(is_anomaly)
            y_pred_s_auc.append(score)
        if domain == "target" or is_anomaly == 1:
            y_true_t_auc.append(is_anomaly)
            y_pred_t_auc.append(score)
            
    auc_src = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
    auc_tgt = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    
    print(f"Metrics (aligned check):")
    print(f"  AUC all    = {auc_all:.6f}")
    print(f"  AUC source = {auc_src:.6f}")
    print(f"  AUC target = {auc_tgt:.6f}")
    print(f"  pAUC       = {p_auc:.6f}")
    
    # Score distribution stats
    def get_stats(arr):
        if not arr: return "N/A"
        return f"Mean: {np.mean(arr):.6f}, Std: {np.std(arr):.6f}, Min: {np.min(arr):.6f}, Max: {np.max(arr):.6f}"
        
    print("\nScore Distributions:")
    print(f"  Source Normal:  {get_stats(src_normal)}")
    print(f"  Source Anomaly: {get_stats(src_anomaly)}")
    print(f"  Target Normal:  {get_stats(tgt_normal)}")
    print(f"  Target Anomaly: {get_stats(tgt_anomaly)}")
    
    # Separation analysis
    mean_src_norm = np.mean(src_normal)
    mean_src_anom = np.mean(src_anomaly)
    mean_tgt_norm = np.mean(tgt_normal)
    mean_tgt_anom = np.mean(tgt_anomaly)
    
    src_sep = mean_src_anom - mean_src_norm
    tgt_sep = mean_tgt_anom - mean_tgt_norm
    
    print("\nScore Separation (Anomaly - Normal):")
    print(f"  Source Separation = {src_sep:.6f}")
    print(f"  Target Separation = {tgt_sep:.6f}")
    
    # separability index d'
    def d_prime(anom, norm):
        return (np.mean(anom) - np.mean(norm)) / np.sqrt(0.5 * (np.var(anom) + np.var(norm)))
        
    print(f"  Source d' = {d_prime(src_anomaly, src_normal):.6f}")
    print(f"  Target d' = {d_prime(tgt_anomaly, tgt_normal):.6f}")
    
    return {
        "auc_all": auc_all,
        "auc_src": auc_src,
        "auc_tgt": auc_tgt,
        "p_auc": p_auc,
        "src_normal": src_normal,
        "src_anomaly": src_anomaly,
        "tgt_normal": tgt_normal,
        "tgt_anomaly": tgt_anomaly
    }

baseline_stats = analyze_experiment(baseline_preds, "BASELINE")
aug_stats = analyze_experiment(aug_preds, "AUGMENTATION")

# 4. Load models to inspect covariance matrices
print("\n=================== MODEL COVARIANCE INSPECTION ===================")
def inspect_model_pth(path, name):
    print(f"\nInspecting model: {name}")
    try:
        sd = torch.load(path, map_location="cpu")
        print("Keys in state_dict:")
        for k in sd.keys():
            if "cov" in k or "precision" in k or "weight" not in k:
                print(f"  {k}: shape {sd[k].shape}, min {sd[k].min().item():.6f}, max {sd[k].max().item():.6f}")
        
        # Extract covariance matrices if present
        cov_s_key = None
        cov_t_key = None
        for k in sd.keys():
            if "cov_source" in k: cov_s_key = k
            if "cov_target" in k: cov_t_key = k
            
        if cov_s_key and cov_t_key:
            cov_s = sd[cov_s_key].numpy()
            cov_t = sd[cov_t_key].numpy()
            
            # Analyze
            def analyze_cov(matrix, label):
                diag = np.diagonal(matrix)
                trace = np.trace(matrix)
                eigenvalues = np.linalg.eigvalsh(matrix)
                cond_num = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else float('inf')
                print(f"  Covariance {label}:")
                print(f"    Trace (Residual Variance Sum): {trace:.6f}")
                print(f"    Diag - Mean: {np.mean(diag):.6f}, Std: {np.std(diag):.6f}, Min: {np.min(diag):.6f}, Max: {np.max(diag):.6f}")
                print(f"    Off-diag - Mean of absolute values: {np.mean(np.abs(matrix - np.diag(diag))):.6f}")
                print(f"    Eigenvalues - Min: {eigenvalues[0]:.6e}, Max: {eigenvalues[-1]:.6e}")
                print(f"    Condition Number: {cond_num:.6f}")
                
            analyze_cov(cov_s, "Source")
            analyze_cov(cov_t, "Target")
        else:
            print("  Covariance keys not found in state_dict!")
            
    except Exception as e:
        print(f"  Failed to load/inspect model: {e}")

inspect_model_pth(BASELINE_MODEL_PATH, "BASELINE")
inspect_model_pth(AUG_MODEL_PATH, "AUGMENTATION")
