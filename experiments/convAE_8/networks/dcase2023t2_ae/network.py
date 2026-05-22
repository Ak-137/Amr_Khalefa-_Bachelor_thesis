import torch
from torch import nn

class AENet(nn.Module):
    def __init__(self, input_dim, block_size):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # We will use this flag to print shape diagnostics for only the first forward pass.
        self._printed_shapes = False

        # Encoder Convolutional Layers
        # Input shape: (batch, 1, 5, 128)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.01, eps=1e-03)
        self.relu1 = nn.ReLU()
        
        # Shape: (batch, 16, 5, 64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-03)
        self.relu2 = nn.ReLU()
        
        # Shape: (batch, 32, 5, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-03)
        self.relu3 = nn.ReLU()
        
        # Shape: (batch, 64, 5, 16)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.01, eps=1e-03)
        self.relu4 = nn.ReLU()
        
        # Shape: (batch, 128, 5, 8) => flattened to 128 * 5 * 8 = 5120
        self.fc_enc = nn.Linear(128 * 5 * 8, 8)
        self.bn_enc = nn.BatchNorm1d(8, momentum=0.01, eps=1e-03)
        self.relu_enc = nn.ReLU()

        # Decoder Convolutional Layers
        self.fc_dec = nn.Linear(8, 128 * 5 * 8)
        self.bn_dec_fc = nn.BatchNorm1d(128 * 5 * 8, momentum=0.01, eps=1e-03)
        self.relu_dec_fc = nn.ReLU()
        
        # Shape: (batch, 128, 5, 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1))
        self.bn_dec1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-03)
        self.relu_dec1 = nn.ReLU()
        
        # Shape: (batch, 64, 5, 16)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1))
        self.bn_dec2 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-03)
        self.relu_dec2 = nn.ReLU()
        
        # Shape: (batch, 32, 5, 32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1))
        self.bn_dec3 = nn.BatchNorm2d(16, momentum=0.01, eps=1e-03)
        self.relu_dec3 = nn.ReLU()
        
        # Shape: (batch, 16, 5, 64)
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1))
        # Final Shape: (batch, 1, 5, 128)

    def forward(self, x):
        # 1. Reshape flat input (batch, 640) to 2D spectrogram-like input (batch, 1, 5, 128)
        x_reshaped = x.view(-1, 1, 5, 128)
        
        # 2. Encoder Forward
        conv1_out = self.relu1(self.bn1(self.conv1(x_reshaped)))
        conv2_out = self.relu2(self.bn2(self.conv2(conv1_out)))
        conv3_out = self.relu3(self.bn3(self.conv3(conv2_out)))
        conv4_out = self.relu4(self.bn4(self.conv4(conv3_out)))
        
        flat_conv = conv4_out.view(-1, 128 * 5 * 8)
        z = self.relu_enc(self.bn_enc(self.fc_enc(flat_conv)))
        
        # 3. Decoder Forward
        flat_dec = self.relu_dec_fc(self.bn_dec_fc(self.fc_dec(z)))
        dec_reshaped = flat_dec.view(-1, 128, 5, 8)
        
        deconv1_out = self.relu_dec1(self.bn_dec1(self.deconv1(dec_reshaped)))
        deconv2_out = self.relu_dec2(self.bn_dec2(self.deconv2(deconv1_out)))
        deconv3_out = self.relu_dec3(self.bn_dec3(self.deconv3(deconv2_out)))
        recon_reshaped = self.deconv4(deconv3_out)
        
        # 4. Flatten back to (batch, 640)
        recon_flat = recon_reshaped.view(-1, self.input_dim)
        
        # 5. Diagnostic prints for the first batch only
        if not self._printed_shapes:
            print("\n================ [DEBUG CONVAE SHAPES] ================")
            print(f"  - Input shape before reshape: {x.shape}")
            print(f"  - ConvAE input shape: {x_reshaped.shape}")
            print(f"  - Latent shape (z): {z.shape}")
            print(f"  - Reconstructed ConvAE output shape: {recon_reshaped.shape}")
            print(f"  - Final output shape passed to loss: {recon_flat.shape}")
            print("=======================================================\n")
            self._printed_shapes = True
            
        return recon_flat, z
