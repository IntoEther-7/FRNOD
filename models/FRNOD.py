# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-11 15:26
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SupportBranch import SupportBranch
from models.QueryBranch import QueryBranch
from models.backbone import ResNet, Conv_4
from models.feature_reconstruction_net.feature_reconstruction_head import FeatureReconstructionHead
from models.region_proposal_network.rpn import RegionProposalNetwork
from models.roi_align import FeatureAlign
from models.BoxRegression import BoxRegression


class FRNOD(nn.Module):

    def __init__(self, way: int, shot: int, query_shot: int, backbone: nn.Module, support_branch: SupportBranch,
                 query_branch: QueryBranch, roi_align: FeatureAlign, box_regression: BoxRegression, post_nms_top_n,
                 is_cuda):
        r"""
        初始化FRNOD网络
        :param way: way
        :param shot: shot
        :param backbone_name: 骨干网络, 选项为'resnet12'和'conv_4'
        :param num_categories: 类别数
        :param status: 'train', 'pre_train', 'test'
        """
        super(FRNOD, self).__init__()
        self.way = way
        self.shot = shot
        self.query_shot = query_shot
        self.backbone = backbone
        self.support_branch = support_branch
        self.query_branch = query_branch
        self.roi_align = roi_align
        self.box_regression = box_regression
        self.post_nms_top_n = post_nms_top_n
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.is_cuda = is_cuda
        # self.criterion = nn.NLLLoss()
        # self.loss_frn_weight = nn.Parameter(torch.FloatTensor[1.0], requires_grad=True)
        # self.loss_objectness_weight = nn.Parameter(torch.FloatTensor[1.0], requires_grad=True)
        # self.loss_rpn_box_reg_weight = nn.Parameter(torch.FloatTensor[1.0], requires_grad=True)

        # 通道数
        self.channels = self.backbone.num_channel

        # 温度缩放因子, γ
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # 特征图分辨率r, H*W=25, 论文中的r
        self.resolution = self.roi_align.output_size[0] * self.roi_align.output_size[1]

        # α, β
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)
        # 如果是预训练就设置成为0
        if self.training:
            # 预训练中要识别的类数
            self.num_categories = way
            # 类别矩阵, FRN文中3.6的矩阵M, 预训练用到
            self.categories_matrix = nn.Parameter(torch.randn(self.num_categories, self.resolution, self.channels),
                                                  requires_grad=True)
            self.box_per_image = post_nms_top_n['training'] + 1
        else:
            self.box_per_image = post_nms_top_n['testing'] + 1

    def extract_features(self, input):
        r"""
        对输入图像进行特征提取
        :param input: 输入图像
        :return: 返回提取特征
        """
        # 这是batch大小, 共有多少张图
        batch_size = input.size(0)
        feature_map = self.backbone.forward_query(input)

        if self.backbone.name == 'resnet12':
            feature_map = feature_map / np.sqrt(640)

        # 特征图变形: (图片张数, 通道数, 分辨率)
        # 特征图维度调整: (图片张数, 分辨率, 通道数)
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        # return feature_map.view(batch_size, self.channels, -1).permute(0, 2, 1).contiguous()
        return feature_map.contiguous()

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

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = regression_loss.sum()
        num_pos = (gt_label > 0).sum().float()

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss

    def forward_train_trituple(self, s_c, s_n=None, q_c=None, targets=None, scale=1.):
        # 计算support特征图
        s_c = self.support_branch.forward(s_c)
        s_n = self.support_branch.forward(s_n)
        # 计算query特征图, 并提出建议区域
        proposals, proposal_losses, query_features, query_images, targets \
            = self.query_branch.forward(query_images=q_c, targets=targets)
        # 对建议区域进行精简化并进行RoIAlign
        proposals, matched_idxs, labels, regression_targets, boxes_features = \
            self.roi_align.forward(features=query_features,
                                   proposals=proposals,
                                   image_shapes=query_images.image_sizes,
                                   targets=targets)
        # resize之前, 有多少个roi
        n = boxes_features.shape[0]

        # resize特征
        # (roi数, channel, s, s) -> (roi数, s, s, channel) -> (roi数 * 分辨率, channel)
        boxes_features = boxes_features. \
            permute(0, 2, 3, 1). \
            contiguous(). \
            view(n * self.resolution, self.channels)  # (shot * resolution, channel)
        s_c = s_c.view(self.shot, self.channels, self.resolution).permute(0, 2, 1)  # shot, resolution, channel
        s_n = s_n.view(self.shot, self.channels, self.resolution).permute(0, 2, 1)  # shot, resolution, channel
        s_c = s_c.mean(0).unsqueeze(0)
        s_n = s_n.mean(0).unsqueeze(0)

        # 分类
        s = torch.cat([s_c, s_n])  # (way, resolution, channel)

        Q_bar = self.reconstruct_feature_map(s, boxes_features, True)
        euclidean_matrix = self.euclidean_metric(boxes_features, Q_bar)  # [roi数 * resolution, way]
        metric_matrix = self.metric(euclidean_matrix,
                                    box_per_image=n,
                                    resolution=self.resolution)  # (roi数, way)

        logits = metric_matrix * self.scale  # (roi数, way)
        log_prediction = F.log_softmax(logits, dim=1, dtype=torch.float)  # (roi数, way)

        gt_prediction = torch.zeros(n, dtype=torch.long)  # (n)
        if self.is_cuda:
            gt_prediction = gt_prediction.cuda()
        criterion = nn.NLLLoss()
        loss_frn = criterion(log_prediction, gt_prediction)
        proposal_losses['loss_frn'] = loss_frn
        losses = proposal_losses

        return losses

        # # 从s_c重构query
        # Q_bar_c = self.reconstruct_feature_map(s_c, boxes_features, True)
        # euclidean_matrix_c = self.euclidean_metric(boxes_features, Q_bar_c)
        # metric_matrix_c = self.metric(euclidean_matrix_c, query_shot=self.query_shot, resolution=self.resolution)
        #
        # # 从s_n重构query
        # Q_bar_n = self.reconstruct_feature_map(s_n, boxes_features, True)
        # euclidean_matrix_n = self.euclidean_metric(boxes_features, Q_bar_n)
        # metric_matrix_n = self.metric(euclidean_matrix_n, query_shot=self.query_shot, resolution=self.resolution)
