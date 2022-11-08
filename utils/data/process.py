# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 9:03
import random

import PIL.Image
import torch
from PIL import Image
from torchvision.transforms import transforms


def pre_process(support: list, query: list, query_anns: list, val: list, val_anns: list, label_ori: list,
                support_transforms=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((320, 320))]),
                query_transforms=transforms.Compose([transforms.ToTensor()]), is_cuda=False, random_sort=False):
    r"""
    图像处理, 转换成tensor, s_c, s_n为tensor[shot, channel, 320, 320], q_c为[tensor, tensor, ...],
    gt_bboxes为[标注列表[每张图像的标注[每个盒子的参数]]],
    labels为[标注列表[每张图像的标签[每个盒子的标签]]]
    :param support: 支持图, [PIL.Image]
    :param query: 查询图,
    :param query_anns: 标注
    :param support_transforms:
    :param query_transforms:
    :param support_n:
    :return: 如果有s_n, 则返回s_c, s_n, q_c, gt_bboxes, labels, 否则返回s_c, q_c, gt_bboxes, labels
    """
    s_c_list = [transform_support(s, support_transforms, is_cuda) for s in support]  # [way * shot, channels, s, s)]
    bg = [get_bg(s, support_transforms, is_cuda) for s in support]  # [way * (shot, channels, s, s)]
    bg_t = torch.stack(bg)
    bg_t = bg_t.mean(1)
    s_c_list.insert(0, bg_t)
    q_c_list = [transform_query(q, query_transforms, is_cuda) for q in query]
    q_anns_list = [transform_anns(query_anns[i], is_cuda, label_ori) for i in range(len(query_anns))]
    q_c_list, q_anns_list = cat_list(q_c_list, q_anns_list, random_sort)
    val_list = [transform_query(v, query_transforms, is_cuda) for v in val]
    val_anns_list = [transform_anns(val_anns[i], is_cuda, label_ori) for i in range(len(val_anns))]
    val_list, val_anns_list = cat_list(val_list, val_anns_list, random_sort)
    return s_c_list, q_c_list, q_anns_list, val_list, val_anns_list


def pre_process_coco(support: list, query: list, query_anns: list, val: list, val_anns: list,
                     support_transforms=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Resize((320, 320))]),
                     query_transforms=transforms.Compose([transforms.ToTensor()]), is_cuda=False, random_sort=False):
    r"""
    图像处理, 转换成tensor, s_c, s_n为tensor[shot, channel, 320, 320], q_c为[tensor, tensor, ...],
    gt_bboxes为[标注列表[每张图像的标注[每个盒子的参数]]],
    labels为[标注列表[每张图像的标签[每个盒子的标签]]]
    :param support: 支持图, [PIL.Image]
    :param query: 查询图,
    :param query_anns: 标注
    :param support_transforms:
    :param query_transforms:
    :param support_n:
    :return: 如果有s_n, 则返回s_c, s_n, q_c, gt_bboxes, labels, 否则返回s_c, q_c, gt_bboxes, labels
    """
    s_c_list = [transform_support(s, support_transforms, is_cuda) for s in support]  # [way * shot, channels, s, s)]
    bg = [get_bg(s, support_transforms, is_cuda) for s in support]  # [way * (shot, channels, s, s)]
    bg_t = torch.stack(bg)
    bg_t = bg_t.mean(1)
    s_c_list.insert(0, bg_t)
    # q_c_list = [transform_query(q, query_transforms, is_cuda) for q in query]
    # q_anns_list = [transform_anns(query_anns[i], is_cuda, i + 1) for i in range(len(query_anns))]
    # q_c_list, q_anns_list = cat_list(q_c_list, q_anns_list, random_sort)
    q_c_list, q_anns_list = cat_list(query, query_anns, random_sort)
    # val_list = [transform_query(v, query_transforms, is_cuda) for v in val]
    # val_anns_list = [transform_anns(val_anns[i], is_cuda, i + 1) for i in range(len(val_anns))]
    # val_list, val_anns_list = cat_list(val_list, val_anns_list, random_sort)
    val_list, val_anns_list = cat_list(val, val_anns, random_sort)
    return s_c_list, q_c_list, q_anns_list, val_list, val_anns_list


def read_single_coco(q, qa, label_ori,
                     query_transforms=transforms.Compose([transforms.ToTensor()]), is_cuda=False):
    q = transform_query([q], query_transforms, is_cuda)
    qa = transform_anns([qa], is_cuda, label_ori)
    return q, qa


def cat_list(list_c_image: list, list_c_ann: list, random_sort):
    r"""
    把[class1[images1, images2, ...],class2[images1, ...]]
    :param list_c_image:
    :return:
    """
    # ------------------数据集处理, 乱序
    l_img = []
    for c_image in list_c_image:
        for img in c_image:
            l_img.append(img)

    l_ann = []
    for c_ann in list_c_ann:
        for ann in c_ann:
            l_ann.append(ann)

    if random_sort:
        index_list = [i for i in range(len(l_img))]
        random.shuffle(index_list)
        l_img = [l_img[i] for i in index_list]
        l_ann = [l_ann[i] for i in index_list]

    return l_img, l_ann


def pre_process_tri(support: dict, query: list, query_anns: list, val, val_anns,
                    support_n: list = None,
                    support_transforms=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Resize((320, 320))]),
                    query_transforms=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize(600)]), is_cuda=False):
    r"""
    图像处理, 转换成tensor, s_c, s_n为tensor[shot, channel, 320, 320], q_c为[tensor, tensor, ...],
    gt_bboxes为[标注列表[每张图像的标注[每个盒子的参数]]],
    labels为[标注列表[每张图像的标签[每个盒子的标签]]]
    :param support: 支持图, [PIL.Image]
    :param query: 查询图,
    :param query_anns: 标注
    :param support_transforms:
    :param query_transforms:
    :param support_n:
    :return: 如果有s_n, 则返回s_c, s_n, q_c, gt_bboxes, labels, 否则返回s_c, q_c, gt_bboxes, labels
    """
    s_c = transform_support(support, support_transforms, is_cuda)
    q_c = transform_query(query, query_transforms, is_cuda)
    q_anns = transform_anns(query_anns, is_cuda, label_ori=True)
    val = transform_query(val, query_transforms, is_cuda)
    val_anns = transform_anns(val_anns, is_cuda, label_ori=True)
    if support_n:
        s_n = transform_support(support_n, support_transforms, is_cuda)
        return s_c, s_n, q_c, q_anns, val, val_anns
    else:
        return s_c, q_c, q_anns, val, val_anns


def transform_support(support_and_ann, transforms, is_cuda):
    support_tensors = []
    t = transforms
    for imgPath, box in support_and_ann:
        img = crop_support(imgPath, box)
        img = t(img)
        support_tensors.append(img)
    support_tensor = torch.stack(support_tensors, dim=0)
    if is_cuda:
        support_tensor = support_tensor.cuda()
    return support_tensor


def get_bg(support_and_ann, support_transforms, is_cuda):
    bg_tensors = []
    t = support_transforms
    for imgPath, box in support_and_ann:
        img = PIL.Image.open(imgPath).convert('RGB')
        img = t(img)
        bg_tensors.append(img)
    bg_tensor = torch.stack(bg_tensors, dim=0)
    if is_cuda:
        bg_tensor = bg_tensor.cuda()

    return bg_tensor


def transform_query(query, transforms, is_cuda):
    query_tensors = []
    t = transforms
    for imgPath in query:
        img = PIL.Image.open(imgPath).convert('RGB')
        img = t(img)
        if is_cuda:
            img = img.cuda()
        query_tensors.append(img)

    return query_tensors


def transform_anns(query_anns, is_cuda, label_ori: list):
    anns = []
    for query_ann in query_anns:  # 每张图片的ann
        # print('---------------')
        img_bboxes = []
        img_labels = []
        for i in query_ann:
            # print(i)
            i = i[0]
            # print(i)
            img_bboxes.append([i['bbox'][0], i['bbox'][1], i['bbox'][0] + i['bbox'][2], i['bbox'][1] + i['bbox'][3]])
            if not label_ori == None:
                img_labels.append(label_ori.index(i['category_id']) + 1)
            else:
                img_labels.append(i['category_id'])
            if is_cuda:
                i.update({'boxes': torch.tensor(img_bboxes).cuda(),
                          'labels': torch.tensor(img_labels).cuda()})
                ann_img = i
            else:
                i.update({'boxes': torch.tensor(img_bboxes).cuda(),
                          'labels': torch.tensor(img_labels).cuda()})
                ann_img = i
        anns.append(ann_img)
    return anns


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


def label2dict(detection, val_anns_ori_single, label_list):
    res = []

    for i in detection:
        boxes = i['boxes'].tolist()
        labels = i['labels'].tolist()
        scores = i['scores'].tolist()
        for j in range(len(labels)):
            x1, y1, x2, y2 = boxes[j]
            w = x2 - x1
            h = y2 - y1
            box = [x1, y1, w, h]
            res.append({'image_id': val_anns_ori_single['image_id'], 'bbox': box, 'score': scores[j],
                        'category_id': label_list[labels[j] - 1]})
            # img_res = []
            # if not len(i['scores'] == 0):
            #     i['scores']
            #     for xyxy in i['boxes'].tolist():
            #         x1, y1, x2, y2 = xyxy
            #         # x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            #         w = x2 - x1
            #         h = y2 - y1
            # if __name__ == '__main__':
            #     from utils.data.dataset import FsodDataset
            #
            #     fsod = FsodDataset(root='../../datasets/fsod/', annFile='../../datasets/fsod/annotations/fsod_test.json',
            #                        support_shot=5,
            #                        query_shot=5, seed=114514)
            #     support, query, query_anns = fsod[10]
            #     anns = transform_anns(query_anns, is_cuda=False)
    return res
