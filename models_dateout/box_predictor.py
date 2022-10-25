# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 15:05
import torch
from torch import nn

#
# class FRPredictor(nn.Module):
#     """
#     Standard classification + bounding box regression layers
#     for Fast R-CNN.
#
#     Args:
#         in_channels (int): number of input channels
#         num_classes (int): number of output classes (including background)
#     """
#
#     def __init__(self, in_channels, num_classes, way):
#         super(FRPredictor, self).__init__()
#         self.cls_score = nn.Linear(in_channels, num_classes)
#         self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
#         self.way = way
#
#     def forward(self, x):
#         if x.dim() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#         x = x.flatten(start_dim=1)
#         scores = self.cls_score(x)
#         bbox_deltas = self.bbox_pred(x)
#
#         return scores, bbox_deltas
from torchvision.models.detection.faster_rcnn import TwoMLPHead


class FRPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(FRPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, s, x):
        r"""

        :param x: (Roi数, 1024)
        :return:
        """
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)

        bbox_deltas = self.bbox_pred(x)

        # 分类
        scores = self.cls_score(x) # (roi数, numclass)

        return scores, bbox_deltas

    def bbox_predictor(self, x: torch.Tensor):
        x = self.two_mlp_head.forward(x, )
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas

    def cls_predictor(self, support: torch.Tensor, boxes_features: torch.Tensor, n, Woodubry=True):
        # resize特征
        # (roi数, channel, s, s) -> (roi数, s, s, channel) -> (roi数 * 分辨率, channel)
        boxes_features = boxes_features. \
            permute(0, 2, 3, 1). \
            contiguous(). \
            view(n * self.resolution, self.channels)  # (shot * resolution, channel)
        Q_bar = self.reconstruct_feature_map(support, boxes_features, Woodubry)
        euclidean_matrix = self.euclidean_metric(boxes_features, Q_bar)  # [roi数 * resolution, way]
        metric_matrix = self.metric(euclidean_matrix,
                                    box_per_image=n,
                                    resolution=self.resolution)  # (roi数, way)
        logits = metric_matrix * self.scale  # (roi数, way)
        return logits

    def reconstruct_feature_map(self, support: torch.Tensor, query: torch.Tensor, Woodubry=True):
        r"""
        通过支持特征图对查询特征图进行重构
        :param support: 支持特征
        :param query: 查询特征
        :param alpha: alpha
        :param beta: beta
        :param Woodubry: 是否使用Woodbury等式, 不使用的话就是用优化后的Woodbury等式
        :return:
        """
        # kr/d
        alpha = self.r[0]
        beta = self.r[1]
        reg = support.size(1) / support.size(2)

        # λ
        lam = reg * alpha.exp() + 1e-6

        # γ
        rho = beta.exp()

        # size(way, channel, shot*resolution), support_T为转置
        support_t: torch.Tensor = support.permute(0, 2, 1)

        # 当 d > kr 时，Eq8 中的 Q 公式计算效率很高，
        # 因为最昂贵的步骤是反转不随 d 增长的 kr kr 矩阵。
        # 从左到右计算矩阵乘积也避免了在内存中存储可能很大的 d d 矩阵。
        # 但是，如果特征图很大或镜头数特别高（kr > d），则方程式。
        # 8 可能很快变得无法计算。在这种情况下，Qexists 的替代公式，根据计算要求将 d 替换为 kr。
        if Woodubry:
            # channel <kr 建议使用eq10
            # FRN论文, 公式10
            # https://ether-bucket-nj.oss-cn-nanjing.aliyuncs.com/img/image-20220831103223203.png
            # ScT * Sc
            st_s = support_t.matmul(support)  # (way, channel, channel)
            m_inv = (st_s + torch.eye(st_s.size(-1)).to(st_s.device).unsqueeze(0).mul(
                lam)).inverse()  # (way, channel, channel)
            hat = m_inv.matmul(st_s)
        else:
            # channel > kr 建议使用eq8
            # Sc * ScT
            # https://ether-bucket-nj.oss-cn-nanjing.aliyuncs.com/img/image-20220831095706524.png
            s_st = support.matmul(support_t)  # (way, shot*resolution, shot*resolution)
            m_inv = (s_st + torch.eye(s_st.size(-1)).to(s_st.device).unsqueeze(0).mul(
                lam)).inverse()  # (way, shot*resolution, shot*resolutions)
            hat = support_t.matmul(m_inv).matmul(support)  # (way, channel, channel)

        Q_bar = query.matmul(hat).mul(rho)  # (way, way*query_shot*resolution, channel)

        return Q_bar

    def euclidean_metric(self, query: torch.Tensor, Q_bar: torch.Tensor):
        r"""
        欧几里得度量矩阵
        :param query: 查询图特征
        :param Q_bar: 预算查询图特征
        :return: 返回欧几里得距离
        """
        # query:                                [roi数 * resolution, d]
        # query.unsqueeze(0):                   [1, roi数 * resolution, d]
        # Q_bar:                                [way * shot, roi数 * resolution, d]
        # Q_bar - query.unsqueeze(0):           [way * shot, roi数 * resolution, d]
        # 这里是利用广播机制, 将query转换成了[way * shot, roi数 * resolution, d], 进行相减
        # pow(2)各位单独幂运算                     [way * shot, roi数 * resolution, d]
        # sum(2)指定维度相加, 使矩阵第三维度相加, 导致其形状从[way * shot, roi数 * resolution, d]变成了
        #                                       [way * shot, roi数 * resolution]
        # x.permute(1,0)将x的第0个维度和第一个维度互换了
        #                                       [roi数 * resolution, way * shot]
        euclidean_matrix = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # [roi数 * resolution, way * shot]
        return euclidean_matrix

    def metric(self, euclidean_matrix, box_per_image, resolution):
        r"""
        利用欧几里得度量矩阵, 计算定义距离
        :param euclidean_matrix: 欧几里得度量矩阵, (way*query_shot*resolution, way)
        :param way: way
        :param query_shot: 广播用
        :param resolution: 分辨率
        :return: 返回距离计算
        """
        # euclidean_matrix: [roi数 * resolution, way]
        # .neg():           [roi数 * resolution, way]
        # .view():          [roi数, resolution, way]
        # .mean(1):         (query_shot, way)
        metric_matrix = euclidean_matrix. \
            neg(). \
            contiguous(). \
            view(box_per_image, resolution, self.way) \
            .mean(1)  # (roi数, way)

        return metric_matrix
