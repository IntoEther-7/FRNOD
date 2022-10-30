import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


# modified based on the following repos:
# https://github.com/Sha-Lab/FEAT/blob/master/model/networks/res12.py
# https://github.com/WangYueFt/rfs/blob/master/models/resnet.py
# https://github.com/kjunelee/MetaOptNet/blob/master/models/ResNet12_embedding.py


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DropBlock(nn.Module):
    def __init__(self, block_size):
        r"""
        Drop Block
        @param block_size: block的大小
        """
        # https://img-blog.csdnimg.cn/20181219175613737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE0ODQ1MTE5,size_16,color_FFFFFF,t_70
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        r"""
        如果是训练, 进行drop block
        @param x: 输入
        @param gamma: drop的概率
        @return: 经过drop block之后的图像
        """
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            # 用伯努利分布进行采样
            bernoulli = Bernoulli(gamma)
            # 产生一个掩膜mask, 为一个矩阵
            # shape为(batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)), 内容非0即1
            # 1代表该位置要drop一个block
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        r"""
        计算block的mask, 并翻转0和1
        @param mask: 一个mask矩阵, 非0即1, shape为
        (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))
        @return: 返回计算后的矩阵
        """
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # nonzero(), 获取非0元素的索引
        non_zero_idxs = mask.nonzero()
        # 非零元素个数
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(nn.Module):
    r"""
    残差块
    """
    expansion = 1  # 膨胀率

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, max_pool=True):
        r"""
        初始化一个残差块
        @param inplanes: 输入通道数
        @param planes: 输出通道数
        @param stride: 步长
        @param downsample: 是否进行下采样, 或者制定一种方式进行下采样
        @param drop_rate:
        @param drop_block: 是否drop block
        @param block_size:
        @param max_pool: 是否进行最大池化
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.max_pool = max_pool

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        if self.max_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, out_channels, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True):
        super(ResNet, self).__init__()
        self.name = 'resnet12'
        self.num_channel = out_channels
        self.out_channels = out_channels
        self.inplanes = 3
        self.layer1 = self._make_layer(block, n_blocks[0], out_channels // 2,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], out_channels // 2,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], out_channels,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], out_channels,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size,
                                       max_pool=max_pool)

        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,
                    max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,
                          max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet12(out_channels, drop_rate=0.0, max_pool=True, **kwargs):
    """Constructs a ResNet-12 models.
    """
    model = ResNet(out_channels, BasicBlock, [1, 1, 1, 1], drop_rate=drop_rate, max_pool=max_pool, **kwargs)
    return model


if __name__ == '__main__':
    model = resnet12(128)
    data = torch.randn(2, 3, 84, 84)
    x = model(data)
    print(x.size())
    print(x.shape)
