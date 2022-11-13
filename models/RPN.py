# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-25 9:13
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou

from models.CBAM import ModifiedCBAM
from models.MADNet import MADNet


class RPN(RegionProposalNetwork):
    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0, out_channels=512):
        super(RPN, self).__init__(
            anchor_generator,
            head,
            #
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            #
            pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0)
        self.s = None
        # self.attention: ModifiedCBAM = ModifiedCBAM(out_channels, 16)
        self.attention: MADNet = MADNet(out_channels, 16)

    def forward(self,
                images,  # type: ImageList
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):  # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
                Args:
                    images (ImageList): images for which we want to compute the predictions
                    features (OrderedDict[Tensor]): features computed from the images that are
                        used for computing the predictions. Each tensor in the list
                        correspond to different feature levels
                    targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                        If provided, each element in the dict should contain a field `boxes`,
                        with the locations of the ground-truth boxes.

                Returns:
                    boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                        image.
                    losses (Dict[Tensor]): the losses for the model during training. During
                        testing, it is an empty dict.
                """
        # RPN uses all feature maps that are available
        features = list(features.values())
        # features_l = []
        # attention--------------------------------------------------
        # self.s (way, c, s, s)
        # way = self.s.shape[0]
        # channels = self.s.shape[1]
        # s_r = self.s.mean([2, 3]).reshape(way, channels, 1, 1)

        # ModifiedCBAM-----------------------------------
        # for feature in features:
        #     feature = self.attention.forward(self.s, feature)
        #     features_l.append(feature)
        # features = features_l

        # MADNet-----------------------------------------
        features_l = []
        losses_attention = []
        loss_attention = None
        for index, feature in enumerate(features):
            # MADNet-----------------------------------------
            feature, loss_a = self.attention.forward(self.s, feature, images, targets, 0)
            features_l.append(feature)
            losses_attention.append(loss_a)
            # ModifiedCBAM-----------------------------------
            # feature = self.attention.forward(self.s, feature)
            # features_l.append(feature)
        features = features_l
        if not loss_attention == None:
            loss_attention = torch.Tensor(losses_attention).mean()
        else:
            loss_attention = 0

        objectness, pred_bbox_deltas = self.head(features)  # objectness: [(way, AnchorNum A, ?, ?)]

        # 恢复
        # objectness = [torch.unsqueeze(on.mean(0), dim=0) for on in objectness]  # objectness: [(1, A, ?, ?)]
        # pred_bbox_deltas = [torch.unsqueeze(pbd.mean(0), dim=0) for pbd in pred_bbox_deltas]

        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]

        # attention-------------------------------------
        num_anchors_per_level = [s[0] * s[1] * s[2] * self.s.shape[0] for s in num_anchors_per_level_shape_tensors]
        # num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_attention": loss_attention
            }
        return boxes, losses

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # box_loss = box_iou(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds])

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss
