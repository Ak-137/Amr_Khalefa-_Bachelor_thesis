import torch
from torch import nn

class AENet(nn.Module):
    def __init__(self,input_dim, block_size):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # CORAL Residual Domain Adaptation parameters
        self.coral_mu_s = nn.Parameter(torch.zeros(block_size), requires_grad=False)
        self.coral_mu_t = nn.Parameter(torch.zeros(block_size), requires_grad=False)
        self.coral_inv_sqrt_Cs = nn.Parameter(torch.eye(block_size), requires_grad=False)
        self.coral_sqrt_Ct = nn.Parameter(torch.eye(block_size), requires_grad=False)
        self.coral_fitted = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)

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

    def coral_align_res(self, residual, domain="source"):
        if getattr(self, 'coral_fitted', 0) == 0:
            return residual
        if domain == "source":
            diff = residual - self.coral_mu_s
            aligned = torch.matmul(diff, self.coral_inv_sqrt_Cs)
            aligned = torch.matmul(aligned, self.coral_sqrt_Ct) + self.coral_mu_t
            return aligned
        else:
            return residual

    def fit_coral_statistics(self, R_s, R_t, eps=1e-5):
        # R_s: tensor of shape (N_s, D)
        # R_t: tensor of shape (N_t, D)
        mu_s = torch.mean(R_s, dim=0)
        mu_t = torch.mean(R_t, dim=0)
        
        # Covariance C = (diff.t() @ diff) / (N - 1)
        diff_s = R_s - mu_s
        C_s = torch.matmul(diff_s.t(), diff_s) / (R_s.size(0) - 1)
        
        diff_t = R_t - mu_t
        C_t = torch.matmul(diff_t.t(), diff_t) / (R_t.size(0) - 1)
        
        # Regularize
        eye = torch.eye(C_s.size(0), device=R_s.device, dtype=R_s.dtype)
        C_s_reg = C_s + eps * eye
        C_t_reg = C_t + eps * eye
        
        # Whiten Cs: Cs^{-1/2}
        L_s, U_s = torch.linalg.eigh(C_s_reg)
        L_s = torch.clamp(L_s, min=1e-12)
        inv_sqrt_Ls = 1.0 / torch.sqrt(L_s)
        inv_sqrt_Cs = U_s @ torch.diag(inv_sqrt_Ls) @ U_s.t()
        
        # Recolor Ct: Ct^{1/2}
        L_t, U_t = torch.linalg.eigh(C_t_reg)
        L_t = torch.clamp(L_t, min=1e-12)
        sqrt_Lt = torch.sqrt(L_t)
        sqrt_Ct = U_t @ torch.diag(sqrt_Lt) @ U_t.t()
        
        # Save parameters
        self.coral_mu_s.data = mu_s
        self.coral_mu_t.data = mu_t
        self.coral_inv_sqrt_Cs.data = inv_sqrt_Cs
        self.coral_sqrt_Ct.data = sqrt_Ct
        self.coral_fitted.data = torch.tensor(1, dtype=torch.uint8)
        print(f"\n[CORAL RES FIT] Successfully fitted CORAL statistics using {R_s.size(0)} source and {R_t.size(0)} target residuals.")
        print(f"  - Source Mean Norm: {torch.norm(mu_s):.6f}, Target Mean Norm: {torch.norm(mu_t):.6f}")
        print(f"  - Cs eigenvalues: min {L_s.min().item():.6e}, max {L_s.max().item():.6e}")
        print(f"  - Ct eigenvalues: min {L_t.min().item():.6e}, max {L_t.max().item():.6e}\n")
