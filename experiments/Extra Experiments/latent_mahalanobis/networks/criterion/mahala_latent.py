"""
Latent-space Mahalanobis for the latent_mahalanobis experiment.

Baseline residual Mahalanobis (mahala.py) uses reconstruction error reshaped to
(n_vectors, n_mels) and can form NxN coupling via delta @ inv_cov @ delta.T.

Here each encoder output z_i in R^latent_dim is scored independently:
    d_i = (z_i - mu)^T inv_cov (z_i - mu)
Aggregation (mean over vectors in a clip) happens afterward in dcase2023t2_ae.py.
"""
import torch

from networks.criterion.mahala import cov_v, cov_v_diff


def mahalanobis_per_sample(z, inv_cov, mu=None):
    """
    Standard per-row squared Mahalanobis distance.

    Args:
        z: (N, D) latent vectors (one row per mel-frame embedding).
        inv_cov: (D, D) inverse covariance (precision).
        mu: optional (D,) mean; if None, centers using batch mean (fitting only).

    Returns:
        (N,) squared distances d_i.
    """
    if mu is None:
        delta = z - torch.mean(z, dim=0, keepdim=True)
    else:
        delta = z - mu.unsqueeze(0)
    return torch.sum((delta @ inv_cov) * delta, dim=1)


def accumulate_latent_cov(z, cov_acc):
    """Add batch contribution to running latent covariance (centered z)."""
    diff, _ = cov_v_diff(z)
    return cov_acc + cov_v(diff, num=1)


def calc_inv_cov_latent(model, device="cpu"):
    """Invert source/target latent covariances stored on AENet."""
    inv_cov_source = torch.inverse(model.cov_source.data.to(device).float())
    inv_cov_target = torch.inverse(model.cov_target.data.to(device).float())
    return inv_cov_source, inv_cov_target
