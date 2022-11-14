# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-10 15:29
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from models.backbone.Conv_4 import BackBone
from models.backbone.ResNet import resnet50

if __name__ == '__main__':
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3, returned_layers=[1, 2, 3, 4])
    x = torch.randn([5, 3, 1024, 512])
    y = backbone(x)
    print()
