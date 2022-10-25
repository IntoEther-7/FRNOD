# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-14 10:20
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNet, Conv_4
from models.feature_reconstruction_net import FRN
from torchvision.ops import roi_align
from models.region_proposal_network.rpn import RegionProposalNetwork


class FeatureReconstructionHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, way, shot):
        super(FeatureReconstructionHead, self).__init__()
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = nn.Linear(2048, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi_size = roi_size

        self.r = nn.Parameter(torch.zeros(2))
        self.alpha = self.r[0]
        self.beta = self.r[1]
        self.way = way

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def reconstruct_feature_map(self, support: torch.Tensor, query: torch.Tensor, alpha, beta, Woodubry=True):
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

        Q_bar = query.matmul(hat).matmul(rho)  # (way, way*query_shot*resolution, channel)

        return Q_bar

    def euclidean_metric(self, query: torch.Tensor, Q_bar: torch.Tensor):
        r"""
        欧几里得度量矩阵
        :param query: 查询图特征
        :param Q_bar: 预算查询图特征
        :return: 返回欧几里得距离
        """
        # query:                                [way*query_shot*resolution, d]
        # query.unsqueeze(0):                   [1, way*query_shot*resolution, d]
        # Q_bar:                                [way, way*query_shot*resolution, d]
        # Q_bar - query.unsqueeze(0):           [way, way*query_shot*resolution, d]
        # 这里是利用广播机制, 将query转换成了[way, way*query_shot*resolution, d], 进行相减
        # pow(2)各位单独幂运算                     [way, way*query_shot*resolution, d]
        # sum(2)指定维度相加, 使矩阵第三维度相加, 导致其形状从[way, way*query_shot*resolution, d]变成了
        #                                       [way, way*query_shot*resolution]
        # x.permute(1,0)将x的第0个维度和第一个维度互换了
        #                                       [way*query_shot*resolution, way]
        euclidean_matrix = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # (way*query_shot*resolution, way)
        return euclidean_matrix

    def metric(self, euclidean_matrix, way, query_shot, resolution):
        r"""
        利用欧几里得度量矩阵, 计算定义距离
        :param euclidean_matrix: 欧几里得度量矩阵, (way*query_shot*resolution, way)
        :param way: way
        :param query_shot: 广播用
        :param resolution: 分辨率
        :return: 返回距离计算
        """
        # euclidean_matrix: (way * query_shot*resolution, way)
        # .neg():           (way * query_shot*resolution, way)
        # .view():          (way * query_shot, resolution, way)
        # .mean(1):         (way * query_shot, way)
        metric_matrix = euclidean_matrix.neg().view(way * query_shot, resolution, way).mean(
            1)  # (way * query_shot, way)

        return metric_matrix

    def forward(self, support, query, query_rois, roi_indices, img_size):
        n = query.shape(0)  # 图片张数
        if query.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = query_rois.cuda()

        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * query.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * query.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)

        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        roi_features = roi_align(query, indices_and_rois, (self.roi_size, self.roi_size))

        # 分类
        Q_bar = self.reconstruct_feature_map(support=support, query=roi_features, alpha=self.alpha, beta=self.beta)
        euclidean_matrix = self.euclidean_metric(query=roi_features, Q_bar=Q_bar)
        metric_matrix = self.metric(euclidean_matrix=euclidean_matrix, way=self.way, query_shot=n)

        logits = metric_matrix * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        # 定位
        roi_cls_locs = self.cls_loc(roi_features.view(roi_features.size(0), -1))

        return roi_cls_locs, log_prediction


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
