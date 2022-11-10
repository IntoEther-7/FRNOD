# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 16:29
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
from time import sleep

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    random.seed(114514)
    is_cuda = False
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
    backbone = BackBone(64)
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
                 box_nms_thresh=0.3,
                 box_detections_per_img=100,  # coco要求
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=128,
                 box_positive_fraction=0.5,
                 bbox_reg_weights=(10., 10., 5., 5.))

    if is_cuda:
        model.cuda()

    # ----------------------------------------------------------------------------------------
    weight_path = 'weights_coco_c4/frnod2_7.pth'
    weight = torch.load(weight_path)
    model.load_state_dict(weight['models'])
    print('使用%s' % weight_path)
    # ----------------------------------------------------------------------------------------

    # loss_list = []
    # loss_avg_list = []
    eval_results = []
    cat_list = deepcopy(list(fsod.coco.cats.keys()))
    random.shuffle(cat_list)
    mission = len(cat_list) // way
    # fine_epoch = int(num_mission * 0.7)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    for i in range(mission):
        # if i == fine_epoch:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
        print('--------------------epoch: {} / {}--------------------'.format(i + 1, len(cat_list) // way))
        print('load data----------------')
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

        print('validation---------------')
        continue_flag = False

        model.eval()
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
            detection = model.forward(s_c_list, query_images=v, targets=target)
            # pprint(detection)
            # pprint(target_ori)
            target = target[0]
            detection_list = label2dict(detection, target, catIds)
            eval_results.extend(detection_list)

            imgPath = os.path.join(root, 'train2017', fsod.coco.loadImgs(target['image_id'])[0]['file_name'])
            savePath = os.path.join(root, 'predictions_c4', fsod.coco.loadImgs(target['image_id'])[0]['file_name'])
            saveFolder = os.path.split(savePath)[0]
            img = Image.open(imgPath).convert('RGB')
            img_draw = ImageDraw(img)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            for d in detection_list:
                x, y, w, h = d['bbox']
                img_draw.rectangle([(x, y), (x + w, y + h)], fill=None, outline='red', width=1)
            for box in target['boxes']:
                x1, y1, x2, y2 = box.tolist()
                img_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline='blue', width=1)
            img.save(savePath)

    del model, backbone
    # torch.save(loss_avg_list, 'weights/loss_avg_list.json')
    # with open('weights/loss.json', 'w') as f:
    #     json.dump({'loss_list': loss_list, 'loss_avg_list': loss_avg_list}, f)

    with open('datasets/coco/annotations/coco_prediction_c4.json', 'w') as f:
        json.dump(eval_results, f)
    # 验证
    gt_path = 'datasets/coco/annotations/instances_train2017.json'  # 存放真实标签的路径
    dt_path = "datasets/coco/annotations/coco_prediction_c4.json"  # 存放检测结果的路径
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")  #
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
