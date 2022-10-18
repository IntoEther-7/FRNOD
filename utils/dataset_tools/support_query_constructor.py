# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-16 14:29
import os.path

import PIL.Image
import torch
from pycocotools import coco
import random
from torchvision.transforms import transforms


# def n_way_k_shot(root, dataset: coco.COCO, catId: int, way: int, support_shot: int = 2,
#                  query_shot: int = 5):
#     sample_range = random.sample(dataset.cats.keys(), way + 1)
#     print(sample_range)
#     if catId in sample_range:
#         sample_range.remove(catId)
#     else:
#         sample_range = sample_range[:way]
#     print(sample_range)
#     pass


def one_way_k_shot(root, dataset: coco.COCO, catId: int, support_shot: int = 2,
                   query_shot: int = 5):
    r"""
    针对某一个种类, 生成k-shot的support和query-shot的query, 并返回support, query, query的标注
    :param dataset: 数据集
    :param catId: 种类id
    :param support_shot: 几个支持集
    :param query_shot: 几个查询图像
    :return: support 已经经过裁剪的support(仅保留了box部分), query 查询图像, query_anns 查询图像的标注数据
    """
    support_annIds, query_imgIds = k_shot(dataset=dataset,
                                          catId=catId,
                                          support_shot=support_shot,
                                          query_shot=query_shot)

    support = []
    for support_annId in support_annIds:
        ann = dataset.anns[support_annId]
        bbox = ann['bbox']
        imgId = ann['image_id']
        imgInfo = dataset.loadImgs(ids=[imgId])[0]
        imgPath = os.path.join(root, 'images', imgInfo['file_name'])
        img = crop_support(imgPath, bbox)
        support.append(img)

    query = []
    query_anns = []  # type:list
    for imgInfo in dataset.loadImgs(ids=query_imgIds):
        imgPath = os.path.join(root, 'images', imgInfo['file_name'])
        img = PIL.Image.open(imgPath).convert('RGB')
        query.append(img)
        annIds = dataset.getAnnIds(imgIds=[imgInfo['id']], catIds=catId)
        query_anns.append([dataset.loadAnns(annId) for annId in annIds])

    return support, query, query_anns


def k_shot(dataset: coco.COCO, catId: int, support_shot: int = 2, query_shot: int = 5):
    r"""
    对某一个类别, 生成支持集和查询集
    :param dataset: 利用cocoAPI生成的数据集
    :param catId: 类别的ID
    :param support_shot: 需要几个support实例, k-shot
    :param query_shot: 需要几张查询图像
    :return: 返回support_annIds(支持实例的ID -> 标注annID), query_imgIds(查询图像的ID)
    """
    # 该类别的所有图像
    imgIds_cat = dataset.getImgIds(catIds=[catId])
    # 选取其中部分图像
    imgIds_sample = random.sample(imgIds_cat, support_shot + query_shot)
    # 分离成支持图像和查询图像
    support_imgIds = imgIds_sample[:support_shot]
    query_imgIds = imgIds_sample[support_shot:]
    # 支持图像的ann
    support_annIds_all = dataset.getAnnIds(imgIds=support_imgIds, catIds=[catId])
    support_annIds = random.sample(support_annIds_all, support_shot)
    # 查询图像的ann
    query_annIDs = dataset.getAnnIds(imgIds=query_imgIds, catIds=[catId])

    return support_annIds, query_imgIds


def crop_support(imgPath, bbox, is_show=False):
    r"""
    根据bbox, 裁剪实例
    :param imgPath: 图像路径
    :param bbox: bbox
    :param is_show: 是否显示裁剪结果
    :return: 裁剪的图像 img_crop (PIL.Image)
    """
    x1, y1, w, h = bbox
    img = PIL.Image.open(imgPath).convert('RGB')
    # toTensor = transforms.ToTensor()
    # imgTensor = toTensor(img)  # (channel, hight, width)
    # print(imgTensor.shape)
    # new_tensor = img[:, y1:y1 + h, x1:x1 + w]
    # toPILImage = transforms.ToPILImage()
    img_crop = img.crop([x1, y1, x1 + w, y1 + h])
    if is_show:
        img.show()
        img_crop.show()
    return img_crop  # type: PIL.Image


# if __name__ == '__main__':
#     root = '../../datasets/fsod/'
#     train_json = os.path.join(root, 'annotations/fsod_train.json')
#     test_json = os.path.join(root, 'annotations/fsod_test.json')
#     fsod = coco.COCO(annotation_file=test_json)
#     catId = 20
#     random.seed(114514)
#     n_way_k_shot(root=root, dataset=fsod, catId=catId, way=5)
