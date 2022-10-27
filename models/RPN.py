# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-25 9:13
from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers
import torch
from torch import nn


class RPN(RegionProposalNetwork):
    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RPN, self).__init__(
            anchor_generator,
            head,
            #
            fg_iou_thresh, bg_iou_thresh,
            batch_size_per_image, positive_fraction,
            #
            pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0)
        self.s = None

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
        features_l = []
        # attention?
        # self.s (way, c, s, s)
        way = self.s.shape[0]
        channels = self.s.shape[1]
        s_r = self.s.mean([2, 3]).reshape(way, channels, 1, 1)
        for feature in features:
            feature = feature * s_r
            features_l.append(feature)
        features = features_l
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors] * way
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
            }
        return boxes, losses
