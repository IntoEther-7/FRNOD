# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
import torch
from torchvision.models.detection import transform

from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process
from models.backbone.ResNet import resnet12
from models.FRNOD import FRNOD
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator, RPNHead
from torchvision.models.resnet import resnet18
from models.generalized_rcnn import GeneralizedRCNN

if __name__ == '__main__':
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, test_json, support_shot=2, query_shot=5)
    s_c, s_n, q_c_list, q_anns = fsod.triTuple(catId=1)
    s_c, s_n, q_c_list, q_anns = pre_process(s_c, q_c_list, q_anns, s_n)
    print('s_c      : ', s_c.shape)
    print('s_n      : ', s_n.shape)
    print('q_c_list : ', [i.shape for i in q_c_list])
    print('q_anns   : ', q_anns)
    # frnod = FRNOD(way=2, shot=2, backbone_name='resnet12', num_categories=200, status='train')

    t = transform.GeneralizedRCNNTransform(min_size=320, max_size=1000,
                                           image_mean=[0.471, 0.448, 0.408],
                                           image_std=[0.234, 0.239, 0.242])
    # print(q_c)
    # q_c_ImageList, _ = t.forward(q_c)
    # print(q_c_ImageList.tensors, q_c_ImageList.image_sizes)

    anchorGenerator = AnchorGenerator(sizes=((2, 4, 8),), aspect_ratios=((0.5, 1.0, 2.0),))
    rpnHead = RPNHead(in_channels=640, num_anchors=9)
    rpn = RegionProposalNetwork(anchor_generator=anchorGenerator,
                                head=rpnHead,
                                fg_iou_thresh=0.7,
                                bg_iou_thresh=0.3,
                                batch_size_per_image=256,
                                positive_fraction=0.5,
                                pre_nms_top_n={'training': 500, 'testing': 300},
                                post_nms_top_n={'training': 100, 'testing': 50},
                                nms_thresh=0.85)
    index = [i for i in range(len(q_c_list))]

    backbone = resnet12()

    # q_c_features = dict(zip(index, q_c))
    # boxes, losses = rpn.forward(images=q_c_ImageList, features=q_c_features)

    rcnn = GeneralizedRCNN(backbone=backbone, rpn=rpn, roi_heads=rpnHead, transform=t)
    rcnn.train()
    losses, detections = rcnn.forward(q_c_list, q_anns)
    print('losses    : ', len(losses), [i.shape for i in losses])
    print('detections: ', detections)
