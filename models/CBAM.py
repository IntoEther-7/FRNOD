# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-04 10:04
import torch
from torch import nn


class ChannelAttentionModule(nn.Module):
    r"""
    通道注意力模块
    """

    def __init__(self, channel, reduction=16):
        r"""

        :param channel: 输入通道数
        :param reduction: 通道缩放倍率
        """
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        r"""

        :param x: (n ,c, s, s)
        :return: (n, c, 1, 1)
        """
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r"""

        :param x: (n, c, s, s)
        :return: (n, 1, s, s)
        """
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ModifiedCBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ModifiedCBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, support, query):
        r'''

        :param support: (ns, c, s, s)
        :param query: (nq=1, c, ?, ?)
        :return:
        '''
        support_ca = self.channel_attention(support)  # (ns, c, 1, 1)
        query_sa = self.spatial_attention(query)  # (nq=1, 1, ?, ?)
        out = support_ca * query * query_sa  # (ns, c, ?, ?)
        return out


if __name__ == '__main__':
    x = torch.randn([6, 512, 320, 320])
    q = torch.randn([1, 512, 320, 320])
    cbam = ModifiedCBAM(512)
    y = cbam.forward(x, q)
    print()
