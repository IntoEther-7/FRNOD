# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 16:29
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import roi_align, MultiScaleRoIAlign

from models.backbone.Conv_4 import BackBone
from models.change.box_predictor import FRPredictor
from models.change.box_head import FRTwoMLPHead
from utils.data.dataset import FsodDataset

if __name__ == '__main__':
    is_cuda = False
    way = 3
    num_classes = 3
    support_shot = 2
    query_shot = 5
    # 超参
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 100
    positive_fraction = 0.5
    channels = 3
    pre_nms_top_n = {'training': 300, 'testing': 150}
    post_nms_top_n = {'training': 100, 'testing': 50}
    roi_size = (5, 5)
    support_size = (roi_size[0] * 16, roi_size[1] * 16)
    resolution = roi_size[0] * roi_size[1]
    nms_thresh = 0.5
    detections_per_img = 30
    scale = 1.
    representation_size = 1024

    backbone = BackBone(3)
    anchor_generator = AnchorGenerator()
    roi_pooler = MultiScaleRoIAlign(['0'], output_size=roi_size, sampling_ratio=2)
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, train_json, support_shot=support_shot, val_shot=query_shot)

    # support = torch.randn([num_classes, channels, roi_size[0], roi_size[1]])
    support = torch.randn([num_classes, resolution, channels])
    box_head = FRTwoMLPHead(in_channels=channels * resolution, representation_size=representation_size)
    box_predictor = FRPredictor(in_channels=representation_size, num_classes=num_classes,
                                support=support, catIds=[1, 2], Woodubry=True,
                                resolution=resolution, channels=channels, scale=scale)
    # box_predictor = FastRCNNPredictor(in_channels=3, num_classes=None)
    model = FasterRCNN(backbone=backbone,
                       num_classes=None,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       box_head=box_head,
                       box_predictor=box_predictor)
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)
    model.eval()
    x = [torch.rand(3, 300, 400) for i in range(10)]
    predictions = model.forward(x)
    print(predictions)  # [第一张图{'boxes':Tensor(n, 4), 'labels':Tensor(Tensor(n,)), 'scores':Tensor(n,), ...]
