# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-18 10:24
import torch
from torchvision.models.detection import transform
from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process
from models.FRNOD import FRNOD

if __name__ == '__main__':
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, test_json, support_shot=2, query_shot=5)
    s_c, s_n, q_c_list, q_anns = fsod.triTuple(catId=1)
    s_c, s_n, q_c_list, gt_bboxes, labels = pre_process(s_c, q_c_list, q_anns, s_n)
    print('s_c      : ', s_c)
    print('s_n      : ', s_n)
    print('q_c_list : ', q_c_list)
    print('gt_bboxes: ', gt_bboxes)
    print('labels   : ', labels)
    frnod = FRNOD(way=2, shot=2, backbone_name='resnet12', num_categories=200, status='train')
    for q_c in q_c_list:
        frnod.forward_train(s_c, s_n, q_c.unsqueeze(0), bboxes_gt=gt_bboxes, labels=labels)
