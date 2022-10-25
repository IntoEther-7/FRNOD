# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
import json

import torch
from torchvision.models.detection import transform
from torchvision.ops import MultiScaleRoIAlign

from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process
from models.backbone.ResNet import resnet12
from models.backbone.Conv_4 import BackBone
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from models.QueryBranch import QueryBranch
from models.roi_align import FeatureAlign
from models.FRNOD import FRNOD
from models.SupportBranch import SupportBranch
from models.BoxRegression_dateout import BoxRegression

if __name__ == '__main__':
    # root = 'datasets/fsod'
    # train_json = 'datasets/fsod/annotations/fsod_train.json'
    # test_json = 'datasets/fsod/annotations/fsod_test.json'
    # fsod = FsodDataset(root, test_json, support_shot=2, query_shot=2)
    # s_c, s_n, q_c_list, q_anns = fsod.triTuple(catId=1)
    # s_c, s_n, q_c_list, q_anns = pre_process(s_c, q_c_list, q_anns, s_n)
    # torch.save([s_c, s_n, q_c_list, q_anns], 'tmp.pth')
    s_c, s_n, q_c_list, q_anns = torch.load('tmp.pth')
    # print(s_c, s_n, q_c_list, q_anns)

    way = 2
    support_shot = 2
    query_shot = 2
    # 超参
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 200
    positive_fraction = 0.5
    channels = 64
    pre_nms_top_n = {'training': 500, 'testing': 300}
    post_nms_top_n = {'training': 100, 'testing': 50}
    roi_size = (20, 20)
    resolution = roi_size[0] * roi_size[1]
    # 骨干网络
    backbone = BackBone(channels)
    # 支持分支
    support_branch = SupportBranch(backbone)
    # 查询分支
    t = transform.GeneralizedRCNNTransform(min_size=800, max_size=1000,
                                           image_mean=[0.48898794, 0.45319346, 0.40628443],
                                           image_std=[0.2889131, 0.28236272, 0.29298001])
    rpnHead = RPNHead(in_channels=channels, num_anchors=9)
    anchorGenerator = AnchorGenerator(sizes=((2, 4, 8),), aspect_ratios=((0.5, 1.0, 2.0),))
    rpn = RegionProposalNetwork(anchor_generator=anchorGenerator,
                                head=rpnHead,
                                fg_iou_thresh=fg_iou_thresh,
                                bg_iou_thresh=bg_iou_thresh,
                                batch_size_per_image=batch_size_per_image,
                                positive_fraction=positive_fraction,
                                pre_nms_top_n=pre_nms_top_n,
                                post_nms_top_n=post_nms_top_n,
                                nms_thresh=0.85)
    query_branch = QueryBranch(backbone, rpn=rpn, transform=t)
    # 特征对齐
    feature_align = FeatureAlign(output_size=(20, 20),
                                 sampling_ratio=2,
                                 fg_iou_thresh=fg_iou_thresh,
                                 bg_iou_thresh=bg_iou_thresh,
                                 batch_size_per_image=batch_size_per_image,
                                 positive_fraction=positive_fraction,
                                 bbox_reg_weights=None)
    # 框回归
    box_regression = BoxRegression(in_channels=channels * resolution, representation_size=512)
    # 网络
    frnod = FRNOD(way=2, shot=1, query_shot=5, backbone=backbone, support_branch=support_branch,
                  query_branch=query_branch, roi_head=feature_align, box_regression=box_regression,
                  post_nms_top_n=post_nms_top_n, is_cuda=)

    losses = frnod.forward_train_trituple(s_c, s_n, q_c_list, targets=q_anns, scale=1.)
