# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-10 16:33
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.MADNet import Inception
from models.backbone.Conv_4 import BackBone
from models.backbone.ResNet import resnet_fpn_backbone


class ModifiedSFNet(nn.Module):
    def __init__(self, roi_size):
        super(ModifiedSFNet, self).__init__()
        self.roi_size = roi_size
        self.res50_fpn = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5,
                                             returned_layers=[3, 4])
        self.inception = Inception(256)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.)
        self.upsample_support = nn.UpsamplingBilinear2d(size=self.roi_size)
        # self.bn = nn.BatchNorm2d(1280)
        # self.out_channels = 1280
        self.out_channels = 256
        self.s_scale = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(1280, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, is_support=False):
        features = self.res50_fpn.forward(x)
        c3 = features['0']
        c3 = self.inception(c3)
        c4 = features['1']
        c4 = self.upsample(c4)
        out = torch.cat([c3, c4], dim=1)
        out = self.encoder(out)
        if is_support:
            out = self.upsample_support(out)
        return out


if __name__ == '__main__':
    backbone = ModifiedSFNet((8, 8))
    x = torch.randn([5, 3, 1024, 512])
    y = backbone(x)
    print()
