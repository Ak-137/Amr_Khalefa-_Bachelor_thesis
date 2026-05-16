from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.conv_ae.dcase2023t2_conv_ae import DCASE2023T2ConvAE

class Models:
    ModelsDic = {
        "DCASE2023T2-AE": DCASE2023T2AE,
        "DCASE2023T2-ConvAE": DCASE2023T2ConvAE,
    }

    def __init__(self,models_str):
        self.net = Models.ModelsDic[models_str]

    def show_list(self):
        return Models.ModelsDic.keys()
