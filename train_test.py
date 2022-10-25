# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
import torch
from torchvision.models.detection import transform
from torchvision.ops import MultiScaleRoIAlign

from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process
from models.backbone.ResNet import resnet12
from models.backbone.Conv_4 import BackBone
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from models.QueryBranch import QueryBranch
from models.roi_align import FeatureAlign

if __name__ == '__main__':
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, test_json, support_shot=2, val_shot=2)
    s_c, s_n, q_c_list, q_anns = fsod.triTuple(catId=1)
    s_c, s_n, q_c_list, q_anns = pre_process(s_c, q_c_list, q_anns, s_n)
    print('s_c      : ', s_c.shape)
    print('s_n      : ', s_n.shape)
    print('q_c_list : ', [i.shape for i in q_c_list])
    print('q_anns   : ', q_anns)
    # frnod = FRNOD(way=2, shot=2, backbone_name='resnet12', num_categories=200, status='train')

    t = transform.GeneralizedRCNNTransform(min_size=800, max_size=1000,
                                           image_mean=[0.48898794, 0.45319346, 0.40628443],
                                           image_std=[0.2889131, 0.28236272, 0.29298001])
    # print(q_c)
    # q_c_ImageList, _ = t.forward(q_c)
    # print(q_c_ImageList.tensors, q_c_ImageList.image_sizes)

    anchorGenerator = AnchorGenerator(sizes=((2, 4, 8),), aspect_ratios=((0.5, 1.0, 2.0),))
    rpnHead = RPNHead(in_channels=64, num_anchors=9)
    rpn = RegionProposalNetwork(anchor_generator=anchorGenerator,
                                head=rpnHead,
                                fg_iou_thresh=0.7,
                                bg_iou_thresh=0.3,
                                batch_size_per_image=256,
                                positive_fraction=0.5,
                                pre_nms_top_n={'training': 500, 'testing': 300},
                                post_nms_top_n={'training': 100, 'testing': 50},
                                nms_thresh=0.85)
    featureAlign = FeatureAlign(output_size=(20, 20),
                                sampling_ratio=2,
                                fg_iou_thresh=0.7,
                                bg_iou_thresh=0.3,
                                batch_size_per_image=256,
                                positive_fraction=0.5,
                                bbox_reg_weights=None)
    index = [i for i in range(len(q_c_list))]

    backbone = BackBone()

    # q_c_features = dict(zip(index, q_c))
    # boxes, losses = rpn.forward(images=q_c_ImageList, features=q_c_features)

    # roiAlign = MultiScaleRoIAlign(featmap_names=['0'], output_size=[5, 5], sampling_ratio=0.2)
    # roiHeads = RoIHeads(box_roi_pool=roiAlign,
    #                     box_head=None,
    #                     bbox_predictor=None,
    #                     fg_iou_thresh=0.7,
    #                     bg_iou_thresh=0.3,
    #                     batch_size_per_image=256,
    #                     positive_fraction=0.5,
    #                     bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
    #                     score_thresh=0.5,
    #                     nms_thresh=0.85,
    #                     detections_per_img=50)
    rcnn = QueryBranch(backbone=backbone, rpn=rpn, transform=t)
    rcnn.train()
    # print(q_c_list)
    print(q_anns)
    proposals, proposal_losses, features, query_images, targets = rcnn.forward(q_c_list, q_anns)
    del backbone, rcnn, rpn, rpnHead, anchorGenerator, t, root, train_json, test_json, fsod, s_c, s_n, q_c_list, q_anns
    proposals, matched_idxs, labels, regression_targets, boxes_features = featureAlign.forward(,
    print('proposals:   ',[i.shape for i in proposals])
    print('box_features:', boxes_features.shape)  # (202, 64, 20, 20)
    del featureAlign
    from models.FRNOD import FRNOD


    # print('proposals:       ', len(proposals), [i.shape for i in proposals])
    # print('proposal_losses: ', proposal_losses)

    from torchvision.ops.roi_align import RoIAlign
    from torchvision.models.detection.faster_rcnn import TwoMLPHead

    # from torchvision.models_dateout.detection.
