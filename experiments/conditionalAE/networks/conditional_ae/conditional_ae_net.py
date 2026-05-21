import torch
from torch import nn

from networks.conditional_ae.domain_utils import domain_cond_from_names

COND_DIM = 2


class ConditionalAENet(nn.Module):
    """FC AE with domain vector concatenated to encoder input and decoder latent."""

    def __init__(self, input_dim, block_size, cond_dim=COND_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self._domain_cond = None
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        enc_in = self.input_dim + cond_dim
        dec_in = 8 + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

    def set_domain_from_names(self, names, device=None, dtype=torch.float32):
        if device is None:
            device = self.cov_source.device
        self._domain_cond = domain_cond_from_names(names, device, dtype)

    def forward(self, x, domain_cond=None):
        cond = domain_cond if domain_cond is not None else self._domain_cond
        if cond is None:
            raise RuntimeError("domain condition not set; call set_domain_from_names first")
        x_flat = x.view(-1, self.input_dim)
        if cond.shape[0] == 1 and x_flat.shape[0] > 1:
            cond = cond.expand(x_flat.shape[0], -1)
        elif cond.shape[0] != x_flat.shape[0]:
            raise RuntimeError(
                f"domain batch {cond.shape[0]} != feature batch {x_flat.shape[0]}"
            )
        z = self.encoder(torch.cat([x_flat, cond], dim=1))
        return self.decoder(torch.cat([z, cond], dim=1)), z
