# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 18:06
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import transforms


class FRPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        f_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, f_channels, q_channels, num_classes,
                 support, catIds, Woodubry,
                 resolution, channels, scale):
        r"""

        :param f_channels:
        :param num_classes:
        :param support: support(way, s, s)
        :param Woodubry:
        """

        super(FRPredictor, self).__init__()
        self.cls_conv = nn.Sequential(nn.Conv2d(q_channels, q_channels, kernel_size=1),
                                      nn.BatchNorm2d(q_channels))
        self.bbox_pred = nn.Linear(f_channels, num_classes * 4)
        self.support = support  # 包含背景
        self.Woodubry = Woodubry
        self.catIds = catIds
        self.resolution = resolution
        self.channels = channels
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.way = num_classes - 1
        # α, β
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self, query_features_and_x):
        r"""

        :param query_features_and_x: query_features(roi数, channel, s, s), x(roi数, representation_size)
        :param Woodubry:
        :return: scores(roi数, num_classes), bbox_deltas(roi数, num_classes * 4)
        """
        # x: (roi数, representation_size)
        # query_features: (roi数, channel, s, s)
        query_features, x = query_features_and_x
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        support = self.support  # (way + 1, channel, s, s)
        support = support.view(self.way + 1, self.channels, self.resolution)  # (way + 1, channel, resolution)
        support = support.permute(0, 2, 1)  # (way + 1, resolution, channel)

        # ------------------特征整合, 视情况将其取消注释
        # support = self.cls_conv(support)  # (way + 1, channel, s, s)
        # query_features = self.cls_conv(query_features)

        # query_features: (roi数, channel, s, s)
        scores = self.cls_predictor(
            support=support,
            boxes_features=query_features,
            Woodubry=self.Woodubry)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

    def cls_predictor(self, support: torch.Tensor, boxes_features: torch.Tensor, Woodubry=True):
        r"""

        :param support: s(way, resolution, channel)
        :param catIds: [catId, ...]对应support的way通道
        :param boxes_features: query_features: (roi数, channel, s, s)
        :param Woodubry:
        :return:
        """
        # resize特征
        # (roi数, channel, s, s) -> (roi数, s, s, channel) -> (roi数 * 分辨率, channel)
        n = boxes_features.shape[0]
        # boxes_features = boxes_features. \
        #     permute(0, 2, 3, 1). \
        #     contiguous(). \
        #     view(n * self.resolution, self.channels)  # (shot * resolution, channel)
        boxes_features = boxes_features.permute(0, 2, 3, 1)
        boxes_features = boxes_features.contiguous()
        boxes_features = boxes_features.view(n * self.resolution, self.channels)  # (shot * resolution, channel)
        Q_bar = self.reconstruct_feature_map(support, boxes_features, Woodubry)
        euclidean_matrix = self.euclidean_metric(boxes_features, Q_bar)  # [roi数 * resolution, way + 1]
        metric_matrix = self.metric(euclidean_matrix,
                                    box_per_image=n,
                                    resolution=self.resolution)  # (roi数, way + 1)
        logits = metric_matrix * self.scale.exp()  # (roi数, way + 1), 防止logits的数值过小导致softmax评分差距不大
        return logits

    def reconstruct_feature_map(self, support: torch.Tensor, query: torch.Tensor, Woodubry=True):
        r"""
        通过支持特征图对查询特征图进行重构
        :param support: support(way, resolution, d)
        :param query: 查询特征
        :param alpha: alpha
        :param beta: beta
        :param Woodubry: 是否使用Woodbury等式, 不使用的话就是用优化后的Woodbury等式
        :return: 重构的特征
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
        # 计算的结果完全相同
        if Woodubry:
            # channel < kr 建议使用eq10
            # FRN论文, 公式10
            # https://ether-bucket-nj.oss-cn-nanjing.aliyuncs.com/img/image-20220831103223203.png
            # ScT * Sc
            st_s = support_t.matmul(support)  # (way, channel, channel)
            m_inv = torch.eye(st_s.size(-1)).to(st_s.device).unsqueeze(0)
            m_inv = m_inv.mul(lam)
            m_inv = (m_inv + st_s).inverse()
            # m_inv_1 = (st_s + torch.eye(st_s.size(-1)).to(st_s.device).unsqueeze(0).
            #            mul(lam)).inverse()  # (way, channel, channel)
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

        return Q_bar  # 重构的特征

    def euclidean_metric(self, query: torch.Tensor, Q_bar: torch.Tensor):
        r"""
        欧几里得度量矩阵
        :param query: 查询图特征
        :param Q_bar: 预算查询图特征
        :return: 返回欧几里得距离矩阵
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
        # euclidean_matrix = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # [roi数 * resolution, way]
        # query:                                [roi数 * resolution, channel]
        # query.unsqueeze(0):                   [1, roi数 * resolution, channel]
        # Q_bar:                                [way, roi数 * resolution, channel]

        euclidean_matrix = Q_bar - query.unsqueeze(0)  # [way + 1(背景), roi数 * resolution, channel]
        euclidean_matrix = euclidean_matrix.pow(2)  # [way + 1(背景), roi数 * resolution, channel], 距离不需要负值
        euclidean_matrix = euclidean_matrix.sum(2)  # [way + 1(背景), roi数 * resolution]
        euclidean_matrix = euclidean_matrix.permute(1, 0)  # [roi数 * resolution, way + 1(背景)]
        euclidean_matrix = euclidean_matrix / self.resolution
        return euclidean_matrix  # 距离矩阵

    def metric(self, euclidean_matrix, box_per_image, resolution):
        r"""
        利用欧几里得度量矩阵, 计算定义距离
        :param euclidean_matrix: 欧几里得度量矩阵, (way*query_shot*resolution, way)
        :param way: way
        :param query_shot: 广播用
        :param resolution: 分辨率
        :return: 返回距离计算, 负数, 也就是可以当scores
        """
        # euclidean_matrix: [roi数 * resolution, way]
        # .neg():           [roi数 * resolution, way]
        # .view():          [roi数, resolution, way]
        # .mean(1):         (query_shot, way)
        # metric_matrix_1 = euclidean_matrix. \
        #     neg(). \
        #     contiguous(). \
        #     view(box_per_image, resolution, self.way) \缩放到-1到1
        #     .mean(1)  # (roi数, way)
        # euclidean_matrix: [roi数 * resolution, way]
        metric_matrix = euclidean_matrix.neg()  # [roi数 * resolution, way + 1(背景)]
        metric_matrix = metric_matrix.contiguous()  # [roi数 * resolution, way + 1(背景)]
        metric_matrix = metric_matrix.view(box_per_image, resolution,
                                           self.way + 1)  # 包括了背景了, [roi数, resolution, way + 1(背景)]
        metric_matrix = metric_matrix.mean(1)  # (roi数, way + 1(背景))
        # metric_matrix += 2 / (self.way + 1)  # [-1,0] -> [0-1]
        # k = 2 / (metric_matrix.max(1).values - metric_matrix.min(1).values)
        # b = 1 - metric_matrix.max(1).values * k
        # metric_matrix = metric_matrix * k.unsqueeze(1) + b.unsqueeze(1)
        return metric_matrix
