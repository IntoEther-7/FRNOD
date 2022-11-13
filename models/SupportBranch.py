# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-19 16:33
import torch
from torch import nn

from models.backbone.ModifiedSFNet import ModifiedSFNet


class SupportBranch(nn.Module):
    def __init__(self, backbone, shot, channels, resolution, image_mean=None, image_std=None):
        super(SupportBranch, self).__init__()
        self.backbone = backbone
        self.shot = shot
        self.channels = channels
        self.resolution = resolution

        if image_mean is None:
            self.image_mean = [0.485, 0.456, 0.406]
        else:
            self.image_mean = image_mean

        if image_std is None:
            self.image_std = [0.229, 0.224, 0.225]
        else:
            self.image_std = image_std

    def forward(self, support_list):
        s_list = []
        for support in support_list:
            support = self.normalize(support)
            if isinstance(self.backbone, ModifiedSFNet):
                support = self.backbone.forward(support, is_support=True)
            else:
                support = self.backbone.forward(support)  # (shot, channels, s, s)
            support = support.mean(0).unsqueeze(0)
            s_list.append(support)
        s = torch.cat(s_list)  # (way, channels, s, s)
        return s

    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
