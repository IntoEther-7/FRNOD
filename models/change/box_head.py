# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 18:06
from torch import nn
import torch.nn.functional as F


class FRTwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(FRTwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm1d(representation_size)

    def forward(self, query_features):
        r"""

        :param query_features: (roi数, channel, s, s)
        :return: query_features: (roi数, channel, s, s), x(roi数, representation_size)
        """
        x = query_features.flatten(start_dim=1)
        # ------------------如何防止这边x的值为=[0], 如果不加激活, 会梯度爆炸
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        # x = self.fc6(x)
        # x = self.fc7(x)

        return query_features, x
