# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 16:29
import json
import os.path
import random
import sys
from copy import deepcopy
from pprint import pprint

import torch
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import roi_align, MultiScaleRoIAlign
from tqdm import tqdm

from torchvision.models.resnet import ResNet

from models.FROD import FROD
from models.backbone.Conv_4 import BackBone
from models.backbone.ResNet import resnet18, resnet50
from models.change.box_predictor import FRPredictor
from models.change.box_head import FRTwoMLPHead
from utils.data.dataset import FsodDataset
from utils.data.process import pre_process_tri, label2dict, pre_process, pre_process_coco, read_single_coco
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    # torch.cuda.set_device(0)
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

    # channels = 64
    # channels = 256
    # channels = 512
    # backbone = BackBone(num_channel=channels)
    backbone = resnet50(pretrained=True, progress=True, frozen=False)
    # backbone = resnet50(pretrained=True, progress=True, frozen=False)
    channels = backbone.out_channels
    s_scale = backbone.s_scale
    # support_size = (roi_size[0] * 16, roi_size[1] * 16)
    support_size = (roi_size[0] * s_scale, roi_size[1] * s_scale)

    anchor_generator = AnchorGenerator(sizes=((64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(['0'], output_size=roi_size, sampling_ratio=2)
    root = 'datasets/coco'
    train_json = 'datasets/coco/annotations/instances_train2017.json'
    test_json = 'datasets/coco/annotations/instances_val2017.json'
    fsod = FsodDataset(root, train_json, support_shot=support_shot, dataset_img_path='train2017')
    if not (os.path.exists('weights_coco_r50/results')):
        os.makedirs('weights_coco_r50/results')

    torch.set_printoptions(sci_mode=False)

    # support = torch.randn([num_classes, channels, roi_size[0], roi_size[1]])
    # support = torch.randn([num_classes, resolution, channels])
    # box_head = FRTwoMLPHead(f_channels=channels * resolution, representation_size=representation_size)
    # box_predictor = FRPredictor(f_channels=representation_size, num_classes=num_classes,
    #                             support=support, catIds=[1, 2], Woodubry=True,
    #                             resolution=resolution, channels=channels, scale=scale)
    # box_predictor = FastRCNNPredictor(f_channels=3, num_classes=None)
    model = FROD(way=way, shot=support_shot, representation_size=representation_size, roi_size=roi_size,
                 resolution=resolution,
                 channels=channels,
                 scale=scale,
                 backbone=backbone,
                 num_classes=num_classes,
                 min_size=600,
                 max_size=1000,
                 image_mean=[0., 0., 0.],  # [0.48898793804461593, 0.45319346269085636, 0.40628443137676473]
                 image_std=[1., 1., 1.],  # [0.2889130963312614, 0.28236272244671895, 0.29298000781217653]
                 rpn_anchor_generator=anchor_generator,
                 rpn_head=None,
                 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n['training'],
                 rpn_pre_nms_top_n_test=rpn_pre_nms_top_n['testing'],
                 rpn_post_nms_top_n_train=rpn_post_nms_top_n['training'],
                 rpn_post_nms_top_n_test=rpn_post_nms_top_n['testing'],
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256,
                 rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.5,
                 box_roi_pool=roi_pooler,
                 box_head=None,
                 box_predictor=None,
                 box_score_thresh=0.05,  # 0.9 for visualization
                 box_nms_thresh=0.7,
                 box_detections_per_img=100,  # coco要求
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=128,
                 box_positive_fraction=0.5,
                 bbox_reg_weights=(10., 10., 5., 5.))

    if is_cuda:
        model.cuda()

    cat_list = deepcopy(list(fsod.coco.cats.keys()))
    random.shuffle(cat_list)
    num_mission = len(cat_list) // way
    epoch = 20
    fine_epoch = int(epoch * 0.7)
    lr_list = [0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002]
    epoch_lr = []

    # ----------------------------------------
    if not os.path.exists('weights_coco_r50/results/loss_per_image.json'):
        with open('weights_coco_r50/results/loss_per_image.json', 'w') as f:
            json.dump({}, f)
            print('创建loss_per_image.json')
    if not os.path.exists('weights_coco_r50/results/loss_per_image_val.json'):
        with open('weights_coco_r50/results/loss_per_image_val.json', 'w') as f:
            json.dump({}, f)
            print('创建loss_per_image_val.json')

    # ----------------------------------------------------------------------------------------
    # continue_weight = 'weights_coco_r50/frnod5_40.pth'
    # print('!!!!!!!!!!!!!从{}继续训练!!!!!!!!!!!!!'.format(continue_weight))
    # weight = torch.load(continue_weight)
    # model.load_state_dict(weight['models'])
    continue_epoch = 5
    continue_mission = 41
    continue_epoch_done = True
    continue_mission_done = True
    # ----------------------------------------------------------------------------------------

    for e in range(0, epoch):  # 4 * 60000 * 8
        if continue_epoch is not None and continue_epoch_done is False:
            if e < continue_epoch - 1:
                tqdm.write('skip epoch {}'.format(e + 1))
                continue
            elif e == continue_epoch - 1:
                continue_epoch_done = True
        if e < 15:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9, weight_decay=0.0005)
        # if e < 20:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=lr_list[e // 2], momentum=0.9, weight_decay=0.0005)
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr_list[e // 4], momentum=0.9, weight_decay=0.0005)
        loss_avg_list = []
        # eval_results = []
        val_loss_avg_list = []
        continue_flag = False  # 用于判断是否跳过这张图
        for i in range(num_mission):
            if continue_mission is not None and continue_mission_done is False:
                if i < continue_mission - 1:
                    tqdm.write('skip epoch {} mission {}'.format(e + 1, i + 1))
                    continue
                elif i == continue_mission - 1:
                    continue_mission_done = True
            print('--------------------mission: {} / {}--------------------'.format(i + 1, len(cat_list) // way))
            # print('load data----------------')
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
            # print(s_c_list, q_c_list, q_anns_list, val_list, val_anns_list)
            # 训练----------------------------------------------------------
            model.train()
            # print('train--------------------')
            loss_list_tmp = []
            pbar = tqdm(range(len(q_c_list)))
            for index in pbar:
                q = q_c_list[index]
                target = q_anns_list[index]
                for tar in target:
                    bbox = tar[0]['bbox']
                    if bbox[2] <= 0.1 or bbox[3] <= 0.1:
                        tqdm.write('id为{}的标注存在问题, 对应image_id为{}, 跳过此张图像'.format(tar[0]['id'], tar[0]['image_id']))
                        continue_flag = True
                if continue_flag is True:
                    continue_flag = False
                    continue

                q, target = read_single_coco(q, target, label_ori=catIds, query_transforms=transforms.Compose(
                    [transforms.ToTensor()]), is_cuda=is_cuda)
                result = model.forward(s_c_list, query_images=q, targets=target)
                loss = result['loss_classifier'] + result['loss_box_reg'] \
                       + result['loss_objectness'] + result['loss_rpn_box_reg'] \
                       + result['loss_aux'] + result['loss_attention']
                pbar.set_postfix(
                    {'epoch': '{:3}/{:3}'.format(e + 1, epoch), 'mission': '{:3}/{:3}'.format(i + 1, num_mission),
                     'catIds': catIds,
                     '模式': 'trian', '损失': "%.6f" % float(loss)})
                if torch.isnan(loss).any():
                    print('梯度炸了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    sys.exit(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('loss:    ', loss)
                loss_list_tmp.append(float(loss))

            # print('将本次任务的loss写入weights_coco_r50/results/loss_per_image.json')
            with open('weights_coco_r50/results/loss_per_image.json', 'r') as f:
                content = json.load(f)
            with open('weights_coco_r50/results/loss_per_image.json', 'w') as f:
                content.update({'loss_epoch{}_mission_{}'.format(e + 1, i + 1): loss_list_tmp})
                json.dump(content, f)

            print('将本次任务的权重写入weights_coco_r50/frnod{}_{}.pth'.format(e + 1, i + 1))
            torch.save({'models': model.state_dict()}, 'weights_coco_r50/frnod{}_{}.pth'.format(e + 1, i + 1))

            loss_avg = float(torch.Tensor(loss_list_tmp).mean(0))
            loss_avg_list.append(loss_avg)
            print('loss_avg:', loss_avg)

            # 验证----------------------------------------------
            # print('validation---------------')

            loss_list_tmp = []
            pbar = tqdm(range(len(val_list)))

            for index in pbar:
                v = val_list[index]
                target = val_anns_list[index]
                for tar in target:
                    bbox = tar[0]['bbox']
                    if bbox[2] <= 0.1 or bbox[3] <= 0.1:
                        tqdm.write('id为{}的标注存在问题, 对应image_id为{}, 跳过此张图像'.format(tar[0]['id'], tar[0]['image_id']))
                        continue_flag = True
                if continue_flag is True:
                    continue_flag = False
                    continue
                v, target = read_single_coco(v, target, catIds, query_transforms=transforms.Compose(
                    [transforms.ToTensor()]), is_cuda=is_cuda)
                result = model.forward(s_c_list, query_images=v, targets=target)
                loss = result['loss_classifier'] + result['loss_box_reg'] \
                       + result['loss_objectness'] + result['loss_rpn_box_reg'] \
                       + result['loss_aux'] + result['loss_attention']
                pbar.set_postfix(
                    {'epoch': '{:3}/{:3}'.format(e + 1, epoch), 'mission': '{:3}/{:3}'.format(i + 1, num_mission),
                     'catIds': catIds,
                     '模式': 'test', '损失': "%.6f" % float(loss)})
                # pbar.set_postfix_str("loss: %.6f" % float(loss))
                if torch.isnan(loss).any():
                    print('梯度炸了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    sys.exit(0)
                # pprint(detection)
                # pprint(target_ori)
                # 一个任务的val_loss列表
                loss_list_tmp.append(float(loss))

            # print('将本次任务的loss写入weights_coco_r50/results/loss_per_image_val.json')
            with open('weights_coco_r50/results/loss_per_image_val.json', 'r') as f:
                content = json.load(f)
            with open('weights_coco_r50/results/loss_per_image_val.json', 'w') as f:
                content.update({'loss_epoch{}_mission_{}_val'.format(e + 1, i + 1): loss_list_tmp})
                json.dump(content, f)

            loss_avg = float(torch.Tensor(loss_list_tmp).mean(0))
            val_loss_avg_list.append(loss_avg)
            print('val_loss_avg:', loss_avg)

        with open('weights_coco_r50/results/loss_bepoch{}.json'.format(e + 1), 'w') as f:
            json.dump({'loss_avg_list': loss_avg_list, 'val_loss_avg_list': val_loss_avg_list}, f)
