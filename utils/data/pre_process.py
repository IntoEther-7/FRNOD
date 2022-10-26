# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 9:03
import PIL.Image
import torch
from PIL import Image
from torchvision.transforms import transforms


def pre_process(support: dict, query: list, query_anns: list, val, val_anns,
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
    q_anns = transform_anns(query_anns, is_cuda, is_zero=True)
    val = transform_query(val, query_transforms, is_cuda)
    val_anns = transform_anns(val_anns, is_cuda, is_zero=True)
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


def transform_anns(query_anns, is_cuda, is_zero):
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
            if is_zero:
                img_labels.append(0)
            else:
                img_labels.append(i['category_id'])
            if is_cuda:
                ann_img = {'boxes': torch.tensor(img_bboxes).cuda(), 'labels': torch.tensor(img_labels).cuda()}
            else:
                ann_img = {'boxes': torch.tensor(img_bboxes), 'labels': torch.tensor(img_labels)}
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


def label2dict(detection, val_anns_ori_single):
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
            res.append({'image_id': val_anns_ori_single[0][0]['image_id'], 'bbox': box, 'score': scores[j],
                        'category_id': val_anns_ori_single[0][0]['category_id']})
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
