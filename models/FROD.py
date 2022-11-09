# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 19:47
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.detection import FasterRCNN
from models.FasterRCNN import FasterRCNN
from models.SupportBranch import SupportBranch
from models.change.box_head import FRTwoMLPHead
from models.change.box_predictor import FRPredictor


class FROD(nn.Module):
    def __init__(self,
                 # box_predictor params
                 way, shot, representation_size, roi_size,
                 resolution, channels, scale,
                 backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=64, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        super(FROD, self).__init__()

        self.shot = shot
        self.channels = channels
        self.resolution = resolution

        if self.channels < way * self.resolution:
            self.Woodubry = True
        else:
            self.Woodubry = False

        self.backbone = backbone
        # representation_size,support,resolution,channels,scale
        self.support_branch = SupportBranch(backbone, shot=self.shot, channels=self.channels,
                                            resolution=self.resolution, image_mean=image_mean, image_std=image_std)
        self.box_head = FRTwoMLPHead(in_channels=channels * resolution, representation_size=representation_size)
        self.box_predictor = FRPredictor(f_channels=representation_size, q_channels=self.channels,
                                         num_classes=num_classes, support=None,
                                         catIds=[1, 2], Woodubry=self.Woodubry, resolution=resolution,
                                         channels=channels,
                                         scale=scale)
        self.fast_rcnn = FasterRCNN(backbone, None,
                                    # transform parameters
                                    min_size, max_size,
                                    image_mean, image_std,
                                    # RPN parameters
                                    rpn_anchor_generator, rpn_head,
                                    rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
                                    rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
                                    rpn_nms_thresh,
                                    rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                    rpn_batch_size_per_image, rpn_positive_fraction,
                                    rpn_score_thresh,
                                    # Box parameters
                                    box_roi_pool, self.box_head, self.box_predictor,
                                    box_score_thresh, box_nms_thresh, box_detections_per_img,
                                    box_fg_iou_thresh, box_bg_iou_thresh,
                                    box_batch_size_per_image, box_positive_fraction,
                                    bbox_reg_weights)
        self.FRTwoMLPHead = FRTwoMLPHead(in_channels=channels * resolution, representation_size=representation_size)

    def forward(self, support_list, query_images, targets):
        r"""

        :param support_list: List[support]
        :param query_images: List[Images]
        :param targets:
        :return:
        """

        s = self.support_branch(support_list)  # (way + 1, channel, s, s)

        self.box_predictor.support = s
        self.fast_rcnn.rpn.s = s
        result = self.fast_rcnn.forward(query_images, targets)
        aux_loss = self.auxrank(s)
        if self.training:
            result.update({'loss_aux': aux_loss})
        return result

    def auxrank(self, support: torch.Tensor):
        r"""

        :param support: (way, channels, s, s) -> (way, shot * r, channels)
        :return:
        """
        way = support.size(0)
        channels = support.size(1)
        size_1 = support.size(2)
        size_2 = support.size(3)
        shot = size_1 * size_2
        support = support.view(way, channels, size_1 * size_2).permute(0, 2, 1)
        support = support / support.norm(2).unsqueeze(-1)
        L1 = torch.zeros((way ** 2 - way) // 2).long().to(support.device)
        L2 = torch.zeros((way ** 2 - way) // 2).long().to(support.device)
        counter = 0
        for i in range(way):
            for j in range(i):
                L1[counter] = i
                L2[counter] = j
                counter += 1
        s1 = support.index_select(0, L1)  # (s^2-s)/2, s, d
        s2 = support.index_select(0, L2)  # (s^2-s)/2, s, d
        dists = s1.matmul(s2.permute(0, 2, 1))  # (s^2-s)/2, s, s
        assert dists.size(-1) == shot
        frobs = dists.pow(2).sum(-1).sum(-1)
        return frobs.sum().mul(.03)
