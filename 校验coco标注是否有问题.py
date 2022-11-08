# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-07 20:57
import json
import os
import random
from copy import deepcopy
from pprint import pprint

import torch
from PIL import Image
from PIL.ImageDraw import ImageDraw
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import roi_align, MultiScaleRoIAlign
from tqdm import tqdm

from models.FROD import FROD
from models.backbone.Conv_4 import BackBone
from models.backbone.ResNet import resnet18, resnet50
from models.change.box_predictor import FRPredictor
from models.change.box_head import FRTwoMLPHead
from utils.data.dataset import FsodDataset
from utils.data.process import pre_process_tri, label2dict, pre_process, pre_process_coco, read_single_coco
from pycocotools.cocoeval import COCOeval
from fastprogress.fastprogress import master_bar, progress_bar
from time import sleep

if __name__ == '__main__':
    random.seed(114514)
    is_cuda = True
    way = 5
    num_classes = 6
    support_shot = 5
    query_shot = 5
    # 超参
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    batch_size_per_image = 256
    positive_fraction = 0.5
    rpn_positive_fraction = 0.5
    rpn_pre_nms_top_n = {'training': 12000, 'testing': 6000}
    rpn_post_nms_top_n = {'training': 2000, 'testing': 500}
    # rpn_pre_nms_top_n = {'training': 6000, 'testing': 6000}
    # rpn_post_nms_top_n = {'training': 20, 'testing': 20}
    roi_size = (7, 7)
    resolution = roi_size[0] * roi_size[1]
    rpn_nms_thresh = 0.7
    scale = 1.
    representation_size = 512
    roi_pooler = MultiScaleRoIAlign(['0'], output_size=roi_size, sampling_ratio=2)
    root = 'datasets/coco'
    train_json = 'datasets/coco/annotations/instances_train2017.json'
    test_json = 'datasets/coco/annotations/instances_val2017.json'
    fsod = FsodDataset(root, train_json, support_shot=support_shot, dataset_img_path='train2017')

    # channels = 64
    # channels = 256
    # channels = 512
    # backbone = BackBone(num_channel=channels)
    backbone = resnet18(pretrained=True, progress=True, frozen=False)
    # backbone = resnet50(pretrained=True, progress=True, frozen=False)
    channels = backbone.out_channels
    s_scale = backbone.s_scale
    # support_size = (roi_size[0] * 16, roi_size[1] * 16)
    support_size = (roi_size[0] * s_scale, roi_size[1] * s_scale)

    if not (os.path.exists('weights_fsod/results')):
        os.makedirs('weights_fsod/results')

    torch.set_printoptions(sci_mode=False)

    cat_list = deepcopy(list(fsod.coco.cats.keys()))
    random.shuffle(cat_list)
    num_mission = len(cat_list) // way
    epoch = 40
    fine_epoch = int(epoch * 0.7)
    lr_list = [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002]
    epoch_lr = []

    for e in range(0, epoch):  # 4 * 60000 * 8
        for i in range(num_mission):
            if i + 1 < 3:
                continue
            catIds = cat_list[i * way:(i + 1) * way]
            print('catIds:', catIds)
            s_c_list_ori, q_c_list_ori, q_anns_list_ori, val_list_ori, val_anns_list_ori \
                = fsod.n_way_k_shot(catIds)
            s_c_list, q_c_list, q_anns_list, val_list, val_anns_list \
                = pre_process_coco(s_c_list_ori, q_c_list_ori,
                                   q_anns_list_ori,
                                   val_list_ori, val_anns_list_ori,
                                   support_transforms=transforms.Compose(
                                       [transforms.ToTensor(),
                                        transforms.Resize(support_size)]),
                                   query_transforms=transforms.Compose(
                                       [transforms.ToTensor()]),
                                   is_cuda=is_cuda, random_sort=True)
            pbar = tqdm(range(len(q_c_list)))
            # for index in pbar:
            #     if index == 158:
            #         print()
            #     q = q_c_list[index]
            #     target = q_anns_list[index]
            #     q, target = read_single_coco(q, target, label_ori=catIds, query_transforms=transforms.Compose(
            #         [transforms.ToTensor()]), is_cuda=is_cuda)
            #     for t in target:
            #         for tensor in t['boxes']:
            #             x1, y1, x2, y2 = tensor
            #             if x2 - x1 <= 0.1 or y2 - y1 <= 0.1:
            #                 tqdm.write(str(t))
            index = 158
            q = q_c_list[index]
            target = q_anns_list[index]
            for tar in target:
                bbox = tar[0]['bbox']
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
            q, target = read_single_coco(q, target, label_ori=catIds,
                                         query_transforms=transforms.Compose([transforms.ToTensor()]), is_cuda=is_cuda)
            for t in target:
                for tensor in t['boxes']:
                    x1, y1, x2, y2 = tensor
                    if x2 - x1 <= 0.1 or y2 - y1 <= 0.1:
                        tqdm.write(str(t))
