# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-22 16:29
import json
import random

import torch
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import roi_align, MultiScaleRoIAlign
from tqdm import tqdm

from models.FROD import FROD
from models.backbone.Conv_4 import BackBone
from models.change.box_predictor import FRPredictor
from models.change.box_head import FRTwoMLPHead
from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process, label2dict
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    random.seed(114514)
    is_cuda = True
    way = 2
    num_classes = 3
    support_shot = 5
    query_shot = 5
    # 超参
    fg_iou_thresh = 0.5
    bg_iou_thresh = 0.5
    batch_size_per_image = 64
    positive_fraction = 0.5
    channels = 64
    pre_nms_top_n = {'training': 300, 'testing': 150}
    post_nms_top_n = {'training': 100, 'testing': 50}
    roi_size = (5, 5)
    support_size = (roi_size[0] * 16, roi_size[1] * 16)
    resolution = roi_size[0] * roi_size[1]
    nms_thresh = 0.5
    detections_per_img = 30
    scale = 1.
    representation_size = 1024

    backbone = BackBone(channels)
    anchor_generator = AnchorGenerator(sizes=((64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(['0'], output_size=roi_size, sampling_ratio=2)
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, train_json, support_shot=support_shot, val_shot=query_shot)

    # support = torch.randn([num_classes, channels, roi_size[0], roi_size[1]])
    # support = torch.randn([num_classes, resolution, channels])
    # box_head = FRTwoMLPHead(in_channels=channels * resolution, representation_size=representation_size)
    # box_predictor = FRPredictor(in_channels=representation_size, num_classes=num_classes,
    #                             support=support, catIds=[1, 2], Woodubry=True,
    #                             resolution=resolution, channels=channels, scale=scale)
    # box_predictor = FastRCNNPredictor(in_channels=3, num_classes=None)
    model = FROD(shot=support_shot, representation_size=representation_size,
                 resolution=resolution,
                 channels=channels,
                 scale=scale,
                 backbone=backbone,
                 box_fg_iou_thresh=0.5,
                 rpn_positive_fraction=0.7,
                 rpn_fg_iou_thresh=0.5,
                 rpn_bg_iou_thresh=0.5,
                 rpn_batch_size_per_image=256,
                 rpn_nms_thresh=0.7,
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_test=20,
                 rpn_pre_nms_top_n_train=2000,
                 rpn_post_nms_top_n_train=50,
                 num_classes=way,
                 rpn_anchor_generator=anchor_generator,
                 box_roi_pool=roi_pooler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    if is_cuda:
        model.cuda()

    loss_avg_list = []
    eval_results = []
    for i in range(1, 801):
        if i == 601:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0002)
        model.train()
        print('--------------------epoch:   {}--------------------'.format(i))
        s_c, s_n, q_c_ori, q_anns_ori, val_ori, val_anns_ori = fsod.triTuple(catId=i)
        s_c, s_n, q_c, q_anns, val, val_anns \
            = pre_process(support=s_c, support_n=s_n,
                          query=q_c_ori, query_anns=q_anns_ori,
                          val=val_ori, val_anns=val_anns_ori,
                          support_transforms=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Resize(support_size)]),
                          query_transforms=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Resize(600)]), is_cuda=is_cuda)
        loss_list = []
        print('train--------------------')
        for j in tqdm(range(len(q_c))):
            q_c_single = [q_c[j]]
            q_c_anns_single = [q_anns[j]]
            q_c_anns_ori_single = [q_anns_ori[j]]
            result = model.forward([s_c, s_n], query_images=q_c_single, targets=q_c_anns_single)

            # print(result)
            loss = result['loss_classifier'] + result['loss_box_reg'] + result['loss_objectness']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('loss:    ', loss)
            loss_list.append(loss)
            if i % 10 == 0:
                torch.save({'models': model.state_dict()}, 'weights/frnod{}.pth'.format(i))

        loss_avg = torch.Tensor(loss_list).mean(0)
        print('loss_avg:', loss_avg)

        print('val--------------------')
        model.eval()
        for j in tqdm(range(len(val))):
            detection_list = []
            val_single = [val[j]]
            val_anns_single = [val_anns[j]]
            val_anns_ori_single = val_anns_ori[j]
            detection = model.forward([s_c, s_n], query_images=val_single, targets=val_anns_single)
            # print([(i['labels'], i['scores']) for i in detection], val_anns_ori)
            detection_list.extend(label2dict(detection, val_anns_ori_single))
            # print('预测结果个数:', len(detection_list))
            eval_results.extend(detection_list)

    torch.save(loss_list, 'weights/loss_list.json')
    with open('datasets/fsod/annotations/fsod_prediction.json', 'w') as f:
        json.dump(eval_results, f)


    # 验证
    gt_path = "datasets/fsod/annotations/fsod_train.json"  # 存放真实标签的路径
    dt_path = "datasets/fsod/annotations/fsod_prediction.json"    # 存放检测结果的路径
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")                                             #
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()