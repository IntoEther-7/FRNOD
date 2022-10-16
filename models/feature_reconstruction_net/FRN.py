import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbone import Conv_4, ResNet


class FRN(nn.Module):
    r"""
    FRN模型
    """

    def __init__(self, way=None, shots=None, resnet=False, is_pretraining=False, num_categories=None):
        r"""
        初始化FRN网络
        @param way: way
        @param shots: shots
        @param resnet: 是否启用resnet, 如不启用, 就用conv-4
        @param is_pretraining: 是否是预训练
        @param num_categories: 类别数
        """
        super().__init__()

        if resnet:
            # 通道数
            num_channel = 640
            # 特征提取器
            self.feature_extractor = ResNet.resnet12()

        else:
            # 通道数
            num_channel = 64
            # 特征提取器
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way

        # boolean, 标志是否是resnet作为骨干网络
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)

        if is_pretraining:
            # number of categories during pre-training
            self.num_categories = num_categories
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.categories_matrix = nn.Parameter(torch.randn(self.num_categories, self.resolution, self.d), requires_grad=True)

    def get_feature_map(self, inp):
        r"""
        进行特征提取, 返回提取图像的特征, 但变形为(N, HW, C)
        @param inp: 输入图像
        @return: 提取的图像特征
        """
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)

        if self.resnet:
            feature_map = feature_map / np.sqrt(640)

        return feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()  # N,HW,C

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        r"""
        get reconstruction distance

        重构支持特征并计算欧氏距离
        @param query: 查询特征
        @param support: 某一类的支持集特征
        @param alpha:
        @param beta:
        @param Woodbury: 是否使用woodbury等式, 不使用的话就是用优化后的Woodbury等式
        @return: 返回欧氏距离矩阵
        """
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        # kr/d, 也就是正则化项
        reg = support.size(1) / support.size(2)

        # correspond to lambda in the paper
        # lambda
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(
                lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d

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
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way

        # dist是一个矩阵, 包含了各个的精度
        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False):
        r"""
        返回欧氏距离加和矩阵, 对于单个图像的欧氏距离加和, 也就是batch中每个欧氏距离的结果(平方和)
        @param inp: input, 一个batch, 前way * shot个是支持图像, 后way * shot个是查询图像
        @param way: way
        @param shot: shot
        @param query_shot: ?
        @param return_support: 是否返回支持特征
        @return:
        """
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        # 计算特征图
        feature_map = self.get_feature_map(inp)

        support = feature_map[:way * shot].view(way, shot * resolution, d)
        query = feature_map[way * shot:].view(way * query_shot * resolution, d)

        # reconstruction distance
        # 形状: [way*query_shot*resolution, way]
        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha,
                                         beta=beta)  # way*query_shot*resolution, way
        # torch.neg() = -1 * input
        # neg(): 变成负数                   [way*query_shot*resolution, way]
        # view(): 分离图像                  [way * query_shot, resolution, way]
        # mean(1): 单个图像的欧氏距离加和      [way * query_shot, way]
        neg_l2_dist = recon_dist.neg().view(way * query_shot, resolution, way).mean(1)  # way*query_shot, way

        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        # way*query_shot, way
        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)

        # 按维度1返回最大值和索引
        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    def forward_pretrain(self, inp):
        r"""

        @param inp:
        @return:
        """
        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size * self.resolution, self.d)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(query=feature_map, support=self.categories_matrix, alpha=alpha,
                                         beta=beta)  # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size, self.resolution, self.num_categories).mean(1)  # batch_size,num_cat

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction

    def forward(self, inp):
        r"""
        前向传播, 提取特征 -> 缩放? -> softmax计算概率
        @param inp:
        @return:
        """
        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, support
