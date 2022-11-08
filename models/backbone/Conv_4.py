import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        r"""
        一个卷积块, 包含一个3x3卷积(padding=1)和BatchNorm2d
        @param input_channel: 输入通道数
        @param output_channel: 输出通道数
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self, inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self, num_channel=64):
        r"""
        Conv-4骨干网络: (3x3卷积块, ReLU, 最大池化)x4
        @param num_channel: 通道数
        """
        super().__init__()
        self.name = 'conv_4'
        self.num_channel = num_channel
        self.out_channels = num_channel
        self.s_scale = 16

        self.layers = nn.Sequential(
            ConvBlock(3, num_channel),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel, num_channel),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel, num_channel),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel, num_channel),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, inp):
        return self.layers(inp)
