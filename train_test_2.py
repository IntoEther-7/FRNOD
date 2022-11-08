# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

anchor_generator = AnchorGenerator(sizes=((64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
img = torch.ones([1, 3, 1024, 512])
features = torch.randn([1, 512, 32, 16])
il = ImageList(img, [(1024, 512)])
anchors = anchor_generator.forward(il, [features])
print()
