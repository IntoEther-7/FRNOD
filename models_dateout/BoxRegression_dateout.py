# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-19 20:43
import torch
import torch.nn as nn
from torch.nn import functional as F
from models import _utils as det_utils


class BoxRegression(nn.Module):
    def __init__(self, in_channels, representation_size=1024):
        r"""

        :param in_channels: (N, channel, roi_size, roi_size)
        :param representation_size:
        """
        super(BoxRegression, self).__init__()
        self.fc6 =      nn.Linear(in_channels, representation_size)  # representation_size 1024
        self.fc7 =      nn.Linear(representation_size, representation_size)
        self.bbox_pred = nn.Linear(representation_size, 4)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        bbox_deltas = F.relu(self.bbox_pred(x))
        return bbox_deltas

    def box_loss(self, box_regression, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.

        Args:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        # labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = det_utils.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            size_average=False,
        )
        box_loss = box_loss / labels.numel()

        return box_loss


if __name__ == '__main__':
    b = BoxRegression(64 * 20 * 20, 1000)
    x = torch.randn([202, 64, 20, 20])
    y = b(x)
    print()
