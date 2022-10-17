# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
from utils.data.dataset import FsodDataset
from utils.data.pre_process import pre_process
from models.FRNOD import FRNOD
from torchvision.models.detection.rpn import RegionProposalNetwork

if __name__ == '__main__':
    root = 'datasets/fsod'
    train_json = 'datasets/fsod/annotations/fsod_train.json'
    test_json = 'datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, test_json, support_shot=2, query_shot=1)
    s_c, s_n, q_c, q_anns = fsod.triTuple(catId=1)
    s_c, s_n, q_c, q_anns = pre_process(s_c, q_c, q_anns, s_n)
    print(s_c)
    print(s_n)
    print(q_c)
    print(q_anns)
    frnod = FRNOD(way=2, shot=2, backbone_name='resnet12', num_categories=200, status='train')
    frnod.forward_train()
