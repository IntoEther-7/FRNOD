# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
import json

import torch
from torchvision.models.detection import transform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import transforms

from models.box_predictor import FRPredictor
from models.roi_head import FRHead
from utils.data.dataset import FsodDataset
from utils.data.process import pre_process_tri
from models.backbone.ResNet import resnet12
from models.backbone.Conv_4 import BackBone
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator, RPNHead

from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from models.QueryBranch import QueryBranch
from models.roi_align import FeatureAlign
from models.FRNOD import FRNOD
from models.SupportBranch import SupportBranch
from models.BoxRegression_dateout import BoxRegression
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':

    is_cuda = False
    way = 2
    support_shot = 2
    query_shot = 5
    # 超参
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 100
    positive_fraction = 0.5
    channels = 320
    pre_nms_top_n = {'training': 300, 'testing': 150}
    post_nms_top_n = {'training': 100, 'testing': 50}
    roi_size = (7, 7)
    support_size = (roi_size[0] * 16, roi_size[1] * 16)
    resolution = roi_size[0] * roi_size[1]
    nms_thresh = 0.5
    detections_per_img = 30
    scale = 1.

    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, train_json, support_shot=support_shot, val_shot=query_shot)
    # 骨干网络
    backbone = BackBone(channels)
    # 支持分支
    support_branch = SupportBranch(backbone)
    # 查询分支
    t = transform.GeneralizedRCNNTransform(min_size=600, max_size=800,
                                           image_mean=[0.48898794, 0.45319346, 0.40628443],
                                           image_std=[0.2889131, 0.28236272, 0.29298001])
    rpnHead = RPNHead(in_channels=channels, num_anchors=9)
    anchorGenerator = AnchorGenerator(sizes=((4, 8, 16),), aspect_ratios=((0.5, 1.0, 2.0),))
    rpn = RegionProposalNetwork(anchor_generator=anchorGenerator,
                                head=rpnHead,
                                fg_iou_thresh=fg_iou_thresh,
                                bg_iou_thresh=bg_iou_thresh,
                                batch_size_per_image=batch_size_per_image,
                                positive_fraction=positive_fraction,
                                pre_nms_top_n=pre_nms_top_n,
                                post_nms_top_n=post_nms_top_n,
                                nms_thresh=nms_thresh)
    query_branch = QueryBranch(backbone, rpn=rpn, transform=t)
    # roi_head
    # feature_align = FeatureAlign(output_size=roi_size,
    #                              sampling_ratio=2,
    #                              rpn_fg_iou_thresh=rpn_fg_iou_thresh,
    #                              bg_iou_thresh=bg_iou_thresh,
    #                              batch_size_per_image=batch_size_per_image,
    #                              positive_fraction=positive_fraction,
    #                              bbox_reg_weights=None)
    fr_predictor = FRPredictor(in_channels=channels,
                               way=way,
                               channels=channels,
                               resolution=resolution,
                               representation_size=1024,
                               scale=scale)
    fr_head = FRHead(output_size=roi_size, sampling_ratio=2, bbox_predictor=fr_predictor, fg_iou_thresh=fg_iou_thresh,
                     bg_iou_thresh=bg_iou_thresh, batch_size_per_image=batch_size_per_image, bbox_reg_weights=None,
                     positive_fraction=positive_fraction, score_thresh=0.5, nms_thresh=nms_thresh,
                     detections_per_img=detections_per_img)
    # 框回归
    box_regression = BoxRegression(in_channels=channels * resolution, representation_size=1024)
    # 网络
    frnod = FRNOD(way=way, shot=support_shot, query_shot=query_shot, backbone=backbone, support_branch=support_branch,
                  query_branch=query_branch, roi_head=fr_head, box_regression=box_regression,
                  post_nms_top_n=post_nms_top_n,
                  is_cuda=is_cuda)

    # 优化器
    optimizer = torch.optim.SGD(frnod.parameters(), lr=0.01)

    # 精度
    # COCOeval()

    if is_cuda:
        frnod.cuda()

    # 训练
    loss_list = []
    for i in range(1, 801):
        print('--------------------epoch:   {}--------------------'.format(i))
        s_c, s_n, q_c_list, q_anns = fsod.triTuple(catId=i)
        s_c, s_n, q_c_list, q_anns \
            = pre_process_tri(s_c,
                              q_c_list,
                              q_anns,
                              s_n,
                              support_transforms=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Resize(support_size)]),
                              query_transforms=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Resize(600)]),
                              is_cuda=is_cuda)
        result, losses = frnod.forward_train_trituple(s_c, s_n, q_c_list, targets=q_anns, scale=scale)
        print(result)
        print(losses)
        # loss = losses['loss_frn'] + losses['loss_objectness'] + losses['loss_rpn_box_reg']
        loss = losses['loss_classifier'] + losses['loss_box_reg']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:    ', loss)
        loss_list.append(loss)
        if i % 10 == 0:
            torch.save({'models': frnod.state_dict()}, 'weights/frnod{}.pth'.format(i))

    torch.save(loss_list, 'weights/loss_list.json')
