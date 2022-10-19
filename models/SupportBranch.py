# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-19 16:33
from torch import nn


class SupportBranch(nn.Module):
    def __init__(self, backbone):
        super(SupportBranch, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
