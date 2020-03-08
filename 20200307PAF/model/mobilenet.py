import torch
import torch.nn as nn
import torchvision.models as models
from .helper import init, make_standard_block

from .MobileNetV2 import MobileNetV2, InvertedResidual

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class MobileNet(nn.Module):
    def __init__(self, use_bn=True):  # Original implementation doesn't use BN
        super(MobileNet, self).__init__()
        mobilenet = MobileNetV2()
        state_dict = torch.load('mobilenet_v2.pth.tar', map_location=device()) # add map_location='cpu' if no gpu
        mobilenet.load_state_dict(state_dict)
        
        
        layers_to_use = list(list(mobilenet.children())[0].children())[:7]

        self.mobilenet = nn.Sequential(*layers_to_use)
        self.feature_extractor = nn.Sequential(InvertedResidual(32, 64, 1, 6),
                                               InvertedResidual(64, 128, 1, 6))
        init(self.feature_extractor)

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.feature_extractor(x)
        return x
    
