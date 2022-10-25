# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-24 14:35
from copy import deepcopy

import torch
from torchvision import transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import roi_align, MultiScaleRoIAlign

from models.FROD import FROD
from models.backbone.Conv_4 import BackBone
from models.change.box_predictor import FRPredictor
from models.change.box_head import FRTwoMLPHead
from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    is_cuda = False
    way = 2
    num_classes = 2
    support_shot = 5
    query_shot = 10
    # 超参
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 100
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
    anchor_generator = AnchorGenerator()
    roi_pooler = MultiScaleRoIAlign(['0'], output_size=roi_size, sampling_ratio=2)
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, test_json, support_shot=support_shot, val_shot=query_shot)

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
                 num_classes=way,
                 rpn_anchor_generator=anchor_generator,
                 box_roi_pool=roi_pooler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    weight = torch.load('weights/frnod800.pth')
    model.load_state_dict(weight['models'])

    if is_cuda:
        model.cuda()

    loss_list = []
    for i in range(1, 201):
        model.train()
        print('--------------------epoch:   {}--------------------'.format(i))
        s_c, s_n, q_c_ori, q_anns_ori, val_ori, val_anns_ori = fsod.triTuple(catId=i)
        detection_json = deepcopy(val_anns_ori)
        s_c, s_n, q_c, q_anns, val, val_anns \
            = pre_process(support=s_c, support_n=s_n,
                          query=q_c_ori, query_anns=q_anns_ori,
                          val=val_ori, val_anns=val_anns_ori,
                          support_transforms=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Resize(support_size)]),
                          query_transforms=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Resize(600)]), is_cuda=is_cuda)
        result = model.forward([s_c, s_n], query_images=q_c, targets=q_anns)
        print(result)
        print(q_anns)
        loss = result['loss_classifier'] + result['loss_box_reg'] + result['loss_objectness']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:    ', loss)
        loss_list.append(loss)
        if i % 10 == 0:
            torch.save({'models': model.state_dict()}, 'weights_test/frnod{}.pth'.format(i))

        model.eval()
        detection = model.forward([s_c, s_n], query_images=val, targets=val_anns)
        print(detection)



    torch.save(loss_list, 'weights/loss_list.json')
