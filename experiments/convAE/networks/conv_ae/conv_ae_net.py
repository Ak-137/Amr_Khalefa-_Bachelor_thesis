"""
Convolutional autoencoder on mel patches.

Shape flow (default DCASE dev features: n_mels=128, frames=5):
- Loader / feature pipeline still yield flat tensors x with shape (B, n_mels * n_frames).
- Inside forward(), x is reshaped to (B, 1, n_mels, n_frames): one channel, frequency x time.
- Encoder uses 2D convolutions with stride in frequency only (time width stays small, e.g. 5).
- Bottleneck: vector z from a Linear on flattened conv features (compact latent).
- Decoder: Linear expands back to the encoder's last spatial map, then Conv2d refinement
  and a final nn.Upsample(size=(n_mels, n_frames)) so reconstruction always matches the
  input grid (robust to off-by-one from strided convs).
- Output recon is flattened to (B, n_mels * n_frames) so MSE and selective Mahalanobis
  on (recon - x) remain unchanged vs the FC baseline.

cov_source / cov_target buffers match AENet: (block_size, block_size) with block_size = n_mels.
"""

import torch
from torch import nn


class ConvAENet(nn.Module):
    def __init__(self, input_dim, n_mels, n_frames, block_size, latent_dim=128):
        super().__init__()
        if input_dim != n_mels * n_frames:
            raise ValueError(f"input_dim {input_dim} != n_mels * n_frames ({n_mels}*{n_frames})")
        self.input_dim = input_dim
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.latent_dim = latent_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        # Encoder: (B,1,H,W) -> (B,C,h,w); stride (2,1) downsamples mel bins only.
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            h = self.enc_conv(torch.zeros(1, 1, n_mels, n_frames))
            self._enc_channels = int(h.size(1))
            self._enc_h = int(h.size(2))
            self._enc_w = int(h.size(3))
            enc_flat_dim = self._enc_channels * self._enc_h * self._enc_w

        self.fc_enc = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_flat_dim)

        # Decoder: restore spatial layout then upsample exactly to (n_mels, n_frames).
        self.dec_conv = nn.Sequential(
            nn.Conv2d(self._enc_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(n_mels, n_frames), mode="nearest"),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        b = x.size(0)
        x_img = x.view(b, 1, self.n_mels, self.n_frames)
        h = self.enc_conv(x_img)
        h_flat = h.reshape(b, -1)
        z = self.fc_enc(h_flat)
        h_dec = self.fc_dec(z).view(b, self._enc_channels, self._enc_h, self._enc_w)
        out = self.dec_conv(h_dec)
        recon = out.view(b, -1)
        return recon, z
