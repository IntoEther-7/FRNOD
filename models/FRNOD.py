# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-11 15:26
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNet, Conv_4
from models.feature_reconstruction_net.feature_reconstruction_head import FeatureReconstructionHead
from torchvision.ops import roi_align
from models.region_proposal_network.rpn import RegionProposalNetwork


class FRNOD(nn.Module):

    def __init__(self, way=None, shot=None, backbone_name='resnet12', num_categories=None, status='train'):
        r"""
        初始化FRNOD网络
        :param way: way
        :param shot: shot
        :param backbone_name: 骨干网络, 选项为'resnet12'和'conv_4'
        :param num_categories: 类别数
        :param status: 'train', 'pre_train', 'test'
        """
        super(FRNOD, self).__init__()

        # 设置way, shot
        self.status = status
        self.way = way
        self.shot = shot

        # 设置骨干网络
        self.backbone_name = backbone_name
        if self.backbone_name == 'resnet12':
            self.backbone = ResNet.resnet12()
            num_channel = 640
        elif self.backbone_name == 'conv_4':
            num_channel = 64
            self.backbone = Conv_4.BackBone(num_channel)
        else:
            print('骨干网络设置错误, 修复成resnet')
            self.backbone = ResNet.resnet12()
            num_channel = 640

        # 设置RPN
        self.rpn = RegionProposalNetwork(num_channel, num_channel,
                                         ratios=[0.5, 1, 2],
                                         anchor_scales=[8, 16, 32],
                                         feat_stride=16,
                                         mode="training")

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

        self.head = FeatureReconstructionHead(n_class=self.way, roi_size=5, spatial_scale=1., way=self.way,
                                              shot=self.shot)

        # 通道数
        self.channels = num_channel

        # 温度缩放因子, γ
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # 特征图分辨率r, H*W=25, 论文中的r
        self.resolution = 25

        # α, β
        # 如果是预训练就设置成为0
        if self.status == 'train':
            # 预训练中要识别的类数
            self.num_categories = num_categories
            # 类别矩阵, FRN文中3.6的矩阵M, 预训练用到
            self.categories_matrix = nn.Parameter(torch.randn(self.num_categories, self.resolution, self.channels),
                                                  requires_grad=True)

    def extract_features(self, input):
        r"""
        对输入图像进行特征提取
        :param input: 输入图像
        :return: 返回提取特征
        """
        # 这是batch大小, 共有多少张图
        batch_size = input.size(0)
        feature_map = self.backbone.forward(input)

        if self.backbone_name == 'resnet12':
            feature_map = feature_map / np.sqrt(640)

        # 特征图变形: (图片张数, 通道数, 分辨率)
        # 特征图维度调整: (图片张数, 分辨率, 通道数)
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        # return feature_map.view(batch_size, self.channels, -1).permute(0, 2, 1).contiguous()
        return feature_map.contiguous()

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
        metric_matrix = euclidean_matrix.neg().view(way * query_shot, resolution, way).mean(1)  # (way*query_shot, way)

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

    def forward_train(self, s_c=None, s_n=None, q_c=None, bboxes_gt=None, labels=None, scale=1.):

        print(q_c.shape)
        # 计算特征图
        support_c = self.extract_features(s_c)
        if s_n:
            support_n = self.extract_features(s_n)
        query_feature = self.extract_features(q_c)


        # 获取roi
        n_support = support_c.shape[0]
        img_size = q_c.shape[2:]
        # * rpn_locs：rpn对位置的修正，大小[1, all_anchor_num, 4]
        # * rpn_scores ：rpn判断区域前景背景，大小[1, all_anchor_num, 2]
        # * rois：rpn筛选出的roi的位置，大小[final_rpn_num， 4]
        # * roi_indices：rpn筛选出的roi对应的图片索引，大小[final_rpn_num]
        # * anchor：原图像的锚点，大小[all_anchor_num, 4]
        print(query_feature.shape)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(query_feature, img_size, 1.)

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], []

        n = q_c.shape[0]
        # bboxes_gt: (图片数n, 框数k, 参数4)
        for i in range(n):
            bbox = bboxes_gt[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[i]
            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()
            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            # ------------------------------------------------------ #
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label,
                                                                                self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())

        sample_rois = torch.stack(sample_rois, dim=0)
        sample_indexes = torch.stack(sample_indexes, dim=0)

        # 分类
        # 重新定位
        # head
        # roi_cls_locs, roi_scores = self.model_train([base_feature, sample_rois, sample_indexes, img_size], mode='head')
        roi_cls_locs, roi_scores = self.head.forward(support=support_c, query=query_feature, query_rois=rois,
                                                     roi_indices=roi_indices, img_size=img_size)

        for i in range(n):
            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
            # ------------------------------------------------------ #
            n_sample = roi_cls_locs.size()[1]

            roi_cls_loc = roi_cls_locs[i]
            roi_score = roi_scores[i]
            gt_roi_loc = gt_roi_locs[i]
            gt_roi_label = gt_roi_labels[i]

            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # -------------------------------------------------- #
            #   分别计算Classifier网络的回归损失和分类损失
            # -------------------------------------------------- #
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses = losses + [sum(losses)]
        return losses


class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        r"""

        :param anchor: size: (k个锚框, 4), numpy
        :param bbox: szie: (n个gt框, 4), numpy
        :return: iou计算, shape为[num_anchors, num_gt]
        """
        # ----------------------------------------------#
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        # ----------------------------------------------#
        ious = bbox_iou(anchor, bbox)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        # ---------------------------------------------------------#
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        # ---------------------------------------------------------#
        argmax_ious = ious.argmax(axis=1)
        # ---------------------------------------------------------#
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        # ---------------------------------------------------------#
        max_ious = np.max(ious, axis=1)
        # ---------------------------------------------------------#
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        # ---------------------------------------------------------#
        gt_argmax_ious = ious.argmax(axis=0)
        # ---------------------------------------------------------#
        #   保证每一个真实框都存在对应的先验框
        # ---------------------------------------------------------#
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        # ------------------------------------------ #
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        # ----------------------------------------------------- #
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        # ----------------------------------------------------- #
        #   判断正样本数量是否大于128，如果大于则限制在128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   平衡正负样本，保持总数量为256
        # ----------------------------------------------------- #
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # ---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            # ---------------------------------------------------------#
            gt_assignment = iou.argmax(axis=1)
            # ---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            # ---------------------------------------------------------#
            max_iou = iou.max(axis=1)
            # ---------------------------------------------------------#
            #   真实框的标签要+1因为有背景的存在
            # ---------------------------------------------------------#
            gt_roi_label = label[gt_assignment] + 1

        # ----------------------------------------------------------------#
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        # ----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # -----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        # -----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # ---------------------------------------------------------#
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        # ---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


def bbox_iou(bbox_a, bbox_b):
    print(bbox_a.shape, bbox_b.shape)
    print(bbox_a.__class__, bbox_b.__class__)
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


if __name__ == '__main__':
    way = 5
    shot = 2
    support = torch.randn([5, 3, 128, 128]).cuda()
    query = torch.randn([3, 3, 128, 128]).cuda()

    bboxs = torch.empty(0)
    n = 10  # n为图像张数
    for i in range(n):  # i为图像的index
        # print('------------------------')
        k = 5  # 框数
        bbox = torch.randint(low=0, high=128, size=(1, k, 4)).float()  # 对图像生成k个框
        # print(bbox.shape)
        bboxs = torch.cat([bboxs, bbox], 0)

    print('bboxs.shape', bboxs.shape)
    boxes_np = bboxs.numpy()

    # labels = [random.randint(0, way) for i in range(bboxs.shape[0])]
    labels = torch.randint(0, way, size=[bboxs.shape[0] * bboxs.shape[1]])

    # print(bboxs)

    model = FRNOD(way=10, shot=114, num_categories=3).cuda()
    model.forward_train(s_c=support, q_c=query, bboxes_gt=boxes_np, labels=labels)
