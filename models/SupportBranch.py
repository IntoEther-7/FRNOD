# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-19 16:33
import torch
from torch import nn


class SupportBranch(nn.Module):
    def __init__(self, backbone, shot, channels, resolution):
        super(SupportBranch, self).__init__()
        self.backbone = backbone
        self.shot = shot
        self.channels = channels
        self.resolution = resolution

    def forward(self, support_list):
        s_list = []
        for support in support_list:
            support = self.backbone(support)
            support = support.view(self.shot, self.channels, self.resolution). \
                permute(0, 2, 1)  # shot, resolution, channel
            support = support.mean(0).unsqueeze(0)
            s_list.append(support)
        s = torch.cat(s_list)  # (way, resolution, channel)
        return s