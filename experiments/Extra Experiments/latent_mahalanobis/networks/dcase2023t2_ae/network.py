import torch
from torch import nn

# Bottleneck width of this AE; latent Mahalanobis uses (LATENT_DIM x LATENT_DIM) covariances.
LATENT_DIM = 8

class AENet(nn.Module):
    def __init__(self,input_dim, block_size):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.latent_dim = LATENT_DIM
        # block_size kept for API compatibility with baseline (n_mels); covariances are on z, not residuals.
        self.block_size = block_size
        self.cov_source = nn.Parameter(torch.zeros(LATENT_DIM, LATENT_DIM), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(LATENT_DIM, LATENT_DIM), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,8),
            nn.BatchNorm1d(8,momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x.view(-1,self.input_dim))
        return self.decoder(z), z
