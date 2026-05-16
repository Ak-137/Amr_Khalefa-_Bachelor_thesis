"""
Same DCASE2023T2 training / test / Mahalanobis logic as networks.dcase2023t2_ae.dcase2023t2_ae;
only init_model() swaps AENet for ConvAENet (spectrogram-shaped conv path, flat I/O).
"""

from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.conv_ae.conv_ae_net import ConvAENet


class DCASE2023T2ConvAE(DCASE2023T2AE):
    def init_model(self):
        self.block_size = self.data.height
        return ConvAENet(
            input_dim=self.data.input_dim,
            n_mels=self.data.height,
            n_frames=self.data.width,
            block_size=self.block_size,
        )
