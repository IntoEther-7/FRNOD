# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 14:10
import torch
from torchvision import transforms

from utils.data.dataset import FsodDataset
from utils.data.process import pre_process
from PIL.Image import Image

root = 'datasets/fsod'
train_json = 'datasets/fsod/annotations/fsod_train.json'
test_json = 'datasets/fsod/annotations/fsod_test.json'
fsod = FsodDataset(root, train_json, support_shot=5)
way = 5
cat_list = [i for i in range(1, 801)]
num_mission = len(cat_list) // way
support_size = (224, 224)
is_cuda = False
for i in range(num_mission):
    print('--------------------mission: {} / {}--------------------'.format(i + 1, len(cat_list) // way))
    catIds = cat_list[i * way:(i + 1) * way]
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
                      is_cuda=is_cuda, random_sort=True)
    t = transforms.ToPILImage()
    for index in range(len(q_c_list)):
        img: Image = t(q_c_list[index])
        boxes = q_anns_list[index]['boxes'].tolist()
        labels = q_anns_list[index]['labels'].tolist()

        for bl_index in range(len(boxes)):
            tmp = img.crop([int(xyxy) for xyxy in boxes[bl_index]])
            tmp.show(title=labels[bl_index])
