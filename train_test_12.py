# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 16:29
import json
import random
from pprint import pprint

import torch
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import roi_align, MultiScaleRoIAlign
from tqdm import tqdm

from models.FROD import FROD
from models.backbone.Conv_4 import BackBone
from models.backbone.ResNet import resnet12
from models.change.box_predictor import FRPredictor
from models.change.box_head import FRTwoMLPHead
from utils.data.dataset import FsodDataset
from utils.data.process import pre_process_tri, label2dict, pre_process
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    random.seed(114514)
    is_cuda = True
    way = 5
    num_classes = 6
    support_shot = 5
    query_shot = 5
    # 超参
    rpn_fg_iou_thresh = 0.5
    rpn_bg_iou_thresh = 0.5
    batch_size_per_image = 256
    positive_fraction = 0.5
    channels = 64
    # channels = 256
    rpn_positive_fraction = 0.7
    rpn_pre_nms_top_n = {'training': 12000, 'testing': 12000}
    rpn_post_nms_top_n = {'training': 2000, 'testing': 2000}
    # rpn_pre_nms_top_n = {'training': 6000, 'testing': 6000}
    # rpn_post_nms_top_n = {'training': 20, 'testing': 20}
    roi_size = (7, 7)
    support_size = (roi_size[0] * 16, roi_size[1] * 16)
    resolution = roi_size[0] * roi_size[1]
    nms_thresh = 0.7
    scale = 1.
    representation_size = 1024

    backbone = BackBone(num_channel=channels)
    # backbone = resnet12(out_channels=channels, drop_rate=0.3)
    anchor_generator = AnchorGenerator(sizes=((64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(['0'], output_size=roi_size, sampling_ratio=2)
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, train_json, support_shot=support_shot, val_shot=query_shot)

    # support = torch.randn([num_classes, channels, roi_size[0], roi_size[1]])
    # support = torch.randn([num_classes, resolution, channels])
    # box_head = FRTwoMLPHead(f_channels=channels * resolution, representation_size=representation_size)
    # box_predictor = FRPredictor(f_channels=representation_size, num_classes=num_classes,
    #                             support=support, catIds=[1, 2], Woodubry=True,
    #                             resolution=resolution, channels=channels, scale=scale)
    # box_predictor = FastRCNNPredictor(f_channels=3, num_classes=None)
    model = FROD(shot=support_shot, representation_size=representation_size, roi_size=roi_size,
                 resolution=resolution,
                 channels=channels,
                 scale=scale,
                 backbone=backbone,
                 num_classes=num_classes,
                 min_size=600,
                 max_size=1000,
                 image_mean=[0.48898793804461593, 0.45319346269085636, 0.40628443137676473],
                 image_std=[0.2889130963312614, 0.28236272244671895, 0.29298000781217653],
                 rpn_anchor_generator=anchor_generator,
                 rpn_head=None,
                 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n['training'],
                 rpn_pre_nms_top_n_test=rpn_pre_nms_top_n['testing'],
                 rpn_post_nms_top_n_train=rpn_post_nms_top_n['training'],
                 rpn_post_nms_top_n_test=rpn_post_nms_top_n['testing'],
                 rpn_nms_thresh=nms_thresh,
                 rpn_fg_iou_thresh=rpn_fg_iou_thresh,
                 rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                 rpn_batch_size_per_image=batch_size_per_image,
                 rpn_positive_fraction=rpn_positive_fraction,
                 rpn_score_thresh=0.0,
                 box_roi_pool=roi_pooler,
                 box_head=None,
                 box_predictor=None,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,  # coco要求
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=100,
                 box_positive_fraction=0.25,
                 bbox_reg_weights=None)

    if is_cuda:
        model.cuda()

    cat_list = [i for i in range(1, 801)]
    random.shuffle(cat_list)
    num_epoch = len(cat_list) // way
    big_epoch = 10
    fine_epoch = int(big_epoch * 0.7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    all_loss_list = []
    all_loss_avg_list = []

    # ----------------------------------------------------------------------------------------
    weight = torch.load('weights/frnod4_160.pth')
    model.load_state_dict(weight['models'])
    # ----------------------------------------------------------------------------------------

    for e in range(4, 10):
        print('----------------------------------epoch: {} / {}-----------------------------------'.format(e + 1,
                                                                                                               big_epoch))
        if e == fine_epoch:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
        loss_list = []
        loss_avg_list = []
        # eval_results = []
        val_loss_list = []
        val_loss_avg_list = []
        for i in range(num_epoch):

            print('--------------------epoch: {} / {}--------------------'.format(i + 1, len(cat_list) // way))
            print('load data----------------')
            catIds = cat_list[i * way:(i + 1) * way]
            print('catIds:', catIds)
            s_c_list_ori, q_c_list_ori, q_anns_list_ori, val_list_ori, val_anns_list_ori \
                = fsod.n_way_k_shot(catIds)
            s_c_list, q_c_list, q_anns_list, val_list, val_anns_list \
                = pre_process(s_c_list_ori, q_c_list_ori,
                              q_anns_list_ori,
                              val_list_ori, val_anns_list_ori,
                              support_transforms=transforms.Compose(
                                  [transforms.ToTensor(),
                                   transforms.Resize(support_size)]),
                              query_transforms=transforms.Compose(
                                  [transforms.ToTensor()]),
                              is_cuda=is_cuda)
            # print(s_c_list, q_c_list, q_anns_list, val_list, val_anns_list)
            # 训练----------------------------------------------------------
            model.train()
            print('train--------------------')
            loss_list_tmp = []
            for c_index in range(len(q_c_list)):
                qs = q_c_list[c_index]
                qs_anns = q_anns_list[c_index]
                for index in tqdm(range(len(qs)), desc='第{} / {}个类别'.format(c_index + 1, len(q_c_list))):
                    q = qs[index]
                    target = qs_anns[index]
                    result = model.forward(s_c_list, query_images=[q], targets=[target])
                    loss = result['loss_classifier'] + result['loss_box_reg'] \
                           + result['loss_objectness'] + result['loss_rpn_box_reg']
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print('loss:    ', loss)
                    loss_list_tmp.append(float(loss))
                    if (i + 1) % 10 == 0:
                        torch.save({'models': model.state_dict()}, 'weights/降低学习率/frnod{}_{}.pth'.format(e + 1, i + 1))

            loss_list.extend(loss_list_tmp)
            loss_avg = float(torch.Tensor(loss_list_tmp).mean(0))
            loss_avg_list.append(loss_avg)
            print('loss_avg:', loss_avg)
            all_loss_list.extend(loss_list)
            all_loss_avg_list.extend(loss_avg_list)

            # 验证----------------------------------------------
            print('validation---------------')

            loss_list_tmp = []
            for c_index in range(len(val_list)):
                vals = val_list[c_index]
                vals_anns = val_anns_list[c_index]
                vals_anns_ori = val_anns_list_ori[c_index]
                for index in tqdm(range(len(vals)), desc='第{} / {}个类别'.format(c_index + 1, len(val_list))):
                    v = vals[index]
                    target = vals_anns[index]
                    target_ori = vals_anns_ori[index]
                    result = model.forward(s_c_list, query_images=[v], targets=[target])
                    loss = result['loss_classifier'] + result['loss_box_reg'] \
                           + result['loss_objectness'] + result['loss_rpn_box_reg']
                    # pprint(detection)
                    # pprint(target_ori)
                    # 一个任务的val_loss列表
                    loss_list_tmp.append(float(loss))

            val_loss_list.extend([loss_list_tmp])
            loss_avg = float(torch.Tensor(loss_list_tmp).mean(0))
            val_loss_avg_list.append(loss_avg)
            print('val_loss_avg:', loss_avg)

            # tmp_detection_list = []
            # model.eval()
            # for c_index in range(len(val_list)):
            #     vals = val_list[c_index]
            #     vals_anns = val_anns_list[c_index]
            #     vals_anns_ori = val_anns_list_ori[c_index]
            #     for index in tqdm(range(len(vals)), desc='第{} / {}个类别'.format(c_index + 1, len(val_list))):
            #         v = vals[index]
            #         target = vals_anns[index]
            #         target_ori = vals_anns_ori[index]
            #         detection = model.forward(s_c_list, query_images=[v], targets=[target])
            #         # pprint(detection)
            #         # pprint(target_ori)
            #         detection_list = label2dict(detection, target_ori, catIds)
            #         tmp_detection_list.extend(detection_list)
            #         eval_results.extend(detection_list)


        with open('weights/results/降低学习率/loss_bepoch{}.json'.format(e + 1), 'w') as f:
            json.dump({'loss_list': loss_list, 'loss_avg_list': loss_avg_list, 'val_loss_list': val_loss_list,
                       'val_loss_avg_list': val_loss_avg_list}, f)

        # with open('weights/results/降低学习率/fsod_prediction_bepoch{}.json'.format(e + 1), 'w') as f:
        #     json.dump(eval_results, f)

    with open('weights/results/降低学习率/loss_all.json', 'w') as f:
        json.dump({'loss_list': all_loss_list, 'loss_avg_list': all_loss_avg_list}, f)

