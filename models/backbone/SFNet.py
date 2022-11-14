# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-10 16:33
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.MADNet import Inception
from models.backbone.ResNet import resnet_fpn_backbone


class SFNet(nn.Module):
    def __init__(self):
        super(SFNet, self).__init__()
        self.res50_fpn = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3,
                                             returned_layers=[3, 4])
        self.inception = Inception(256)
        self.upsample_c3 = nn.UpsamplingBilinear2d(scale_factor=2.)
        self.upsample_c4 = nn.UpsamplingBilinear2d(scale_factor=4.)
        self.out_channels = 1280
        self.s_scale = 8

    def forward(self, x):
        features = self.res50_fpn.forward(x)
        c3 = features['0']
        c3 = self.upsample_c3(c3)
        c3 = self.inception(c3)
        c4 = features['1']
        c4 = self.upsample_c4(c4)
        out = torch.cat([c3, c4], dim=1)
        return out


if __name__ == '__main__':
    backbone = SFNet()
    x = torch.randn([5, 3, 1024, 512])
    y = backbone(x)
    print()
