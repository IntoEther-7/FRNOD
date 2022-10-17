# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 9:03
import torch
from PIL import Image
from torchvision.transforms import transforms


def pre_process(support: list, query: list, query_anns: list,
                support_n: list = None,
                support_transforms=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((320, 320))]),
                query_transforms=transforms.Compose([transforms.Resize(600)])):
    r"""
    图像处理, 转换成tensor, s_c, s_n为tensor[shot, channel, 320, 320], q_c为[tensor, tensor, ...], q_anns格式没变化
    :param support: 支持图, [PIL.Image]
    :param query: 查询图,
    :param query_anns: 标注
    :param support_transforms:
    :param query_transforms:
    :param support_n:
    :return: 如果有s_n, 则返回s_c, s_n, q_c, q_anns, 否则返回s_c, q_c, q_anns
    """
    s_c = transform_support(support, support_transforms)
    q_c = transform_query(query, query_transforms)
    q_anns = transform_anns(query_anns)
    if support_n:
        s_n = transform_support(support_n, support_transforms)
        return s_c, s_n, q_c, q_anns
    else:
        return s_c, q_c, q_anns


def transform_support(support, transforms):
    support_tensors = []
    t = transforms
    for img in support:
        img = t(img)
        support_tensors.append(img)
    support_tensor = torch.stack(support_tensors, dim=0)
    return support_tensor


def transform_query(query, transforms):
    query_tensors = []
    t = transforms
    for img in query:
        img = t(img)
        query_tensors.append(img)

    return query_tensors


def transform_anns(query_anns):
    return query_anns


if __name__ == '__main__':
    from utils.data.dataset import FsodDataset

    fsod = FsodDataset(root='../../datasets/fsod/', annFile='../../datasets/fsod/annotations/fsod_train.json',
                       support_shot=5,
                       query_shot=5, seed=114514)
    support, query, query_anns = fsod[10]
    transform_anns(query_anns)
