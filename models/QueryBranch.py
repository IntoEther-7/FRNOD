# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union


class QueryBranch(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, transform):
        super(QueryBranch, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections



    def forward(self, query_images, targets=None):
        r"""
        输入query_images
        :param query_images: List[Tensor], images to be processed
        :param targets: List(图像)[List(图像中的多个标注)[Dict[str, Tensor]]]
        :returns: proposals, proposal_losses, features, query_images, targets
        """
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in query_images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        if self.transform:
            query_images, targets = self.transform(query_images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(query_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(query_images, features, targets)
        # print('proposals:       ', [i.shape for i in proposals])
        # print('proposal_losses: ', proposal_losses)
        # print('features:                ', features.__class__, features.keys(), [i.shape for i in features['0']])
        # print('query_images.image_sizes:', query_images.image_sizes)
        # print('targets:                 ', targets)
        return proposals, proposal_losses, features, query_images, targets
        # box_features = self.get_roi_align(features, proposals, query_images.image_sizes, targets)
        # detections, detector_losses = self.roi_heads(features, proposals, query_images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, query_images.image_sizes, original_image_sizes)
        #
        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)
        #
        # if torch.jit.is_scripting():
        #     if not self._has_warned:
        #         warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
        #         self._has_warned = True
        #     return losses, detections
        # else:
        #     return self.eager_outputs(losses, detections)
