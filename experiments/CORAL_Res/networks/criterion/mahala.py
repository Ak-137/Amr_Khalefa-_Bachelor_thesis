import torch

_printed_debug = False

def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)

    return diff, mu

def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var

def mahalanobis(u, v, cov_x, use_precision=False, reduction=True, coral_align_fn=None):
    global _printed_debug
    num, dim = v.size()
    if use_precision == True:
        inv_cov = cov_x
    else:
        # Fallback raw covariance inversion with small regularization
        reg_lambda = 1e-5
        eye = torch.eye(cov_x.size(0), device=cov_x.device, dtype=cov_x.dtype)
        cov_reg = cov_x + reg_lambda * eye
        inv_cov = torch.linalg.pinv(cov_reg)
        
    delta = torch.sub(u, v)
    if coral_align_fn is not None:
        delta = coral_align_fn(delta)
        
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())
    
    # Extract the diagonal elements containing only standard per-frame distances
    diag_loss = torch.diagonal(m_loss)

    if not _printed_debug:
        print("\n================ [DEBUG SHAPES] ================")
        print(f"  - delta shape: {delta.shape}")
        print(f"  - covariance shape: {cov_x.shape}")
        print(f"  - inverse covariance shape: {inv_cov.shape}")
        print(f"  - per-frame score shape: {diag_loss.shape}")
        if reduction:
            print(f"  - final file score shape: {torch.mean(diag_loss).shape} (scalar)")
        else:
            print(f"  - final file score shape: {diag_loss.unsqueeze(1).shape}")
        print("================================================\n")
        _printed_debug = True

    if reduction:
        return torch.mean(diag_loss)
    else:
        # Return N x 1 shape to integrate seamlessly with the pipeline's loss_reduction_1d
        return diag_loss.unsqueeze(1), num

def loss_function_mahala(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None, update_cov=False, use_precision=False, reduction=True, coral_align_fn=None):
    ### Modified mahalanobis loss###
    if update_cov == False:
        loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision, reduction=reduction, coral_align_fn=coral_align_fn)
        return loss
    else:
        diff = x - recon_x
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, block_size))

        loss = diff**2
        if reduction:
            loss = torch.mean(loss, dim=1)
        
        return loss, cov_diff_source, cov_diff_target

def loss_reduction_mahala(loss):
    return torch.mean(loss)

def calc_inv_cov(model, device="cpu"):
    inv_cov_source=None
    inv_cov_target=None
    
    # Retrieve the configurable regularization lambda parameter securely
    reg_lambda = 1e-5
    if hasattr(model, "args") and model.args is not None:
        reg_lambda = getattr(model.args, "mahala_reg_lambda", 1e-5)
    
    cov_x_source = model.cov_source.data
    cov_x_source = cov_x_source.to(device).float()
    
    # Apply covariance regularization
    eye_s = torch.eye(cov_x_source.size(0), device=device, dtype=cov_x_source.dtype)
    cov_x_source_reg = cov_x_source + reg_lambda * eye_s
    
    # Track and log condition number
    cond_source = torch.linalg.cond(cov_x_source_reg).item()
    print(f"\n[COV REGULARIZATION] Source Covariance - Regularization Lambda: {reg_lambda:.1e}")
    print(f"[COV REGULARIZATION] Source Covariance - Condition Number: {cond_source:.4f}")
    
    # Stable pseudo-inverse
    inv_cov_source = torch.linalg.pinv(cov_x_source_reg)
    inv_cov_source = inv_cov_source.to(device).float()
    
    cov_x_target = model.cov_target.data
    cov_x_target = cov_x_target.to(device).float()
    
    # Apply covariance regularization
    eye_t = torch.eye(cov_x_target.size(0), device=device, dtype=cov_x_target.dtype)
    cov_x_target_reg = cov_x_target + reg_lambda * eye_t
    
    # Track and log condition number
    cond_target = torch.linalg.cond(cov_x_target_reg).item()
    print(f"[COV REGULARIZATION] Target Covariance - Regularization Lambda: {reg_lambda:.1e}")
    print(f"[COV REGULARIZATION] Target Covariance - Condition Number: {cond_target:.4f}\n")
    
    # Stable pseudo-inverse
    inv_cov_target = torch.linalg.pinv(cov_x_target_reg)
    inv_cov_target = inv_cov_target.to(device).float()
    
    return inv_cov_source, inv_cov_target
