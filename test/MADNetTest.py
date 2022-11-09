# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-08 15:08
import torch
from PIL.ImageDraw import ImageDraw
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.models.detection.image_list import ImageList

from models.MADNet import MADNet, PixelAttention, ChannelAttention


class TestMADNet(MADNet):
    def __init__(self, in_channels, reduction=16):
        super(TestMADNet, self).__init__()
        self.in_channels = in_channels
        self.PA = PixelAttention(in_channels)
        self.CA = ChannelAttention(in_channels, reduction=reduction)

    def _generate_mask(self, image: ImageList, target, index):
        toImg = transforms.ToPILImage()
        img: Image.Image = toImg(image.tensors[index])
        img.save('ori.png')
        mask = torch.zeros(image.tensors[index].shape[1:]).cuda()
        boxes = target[index]['boxes']
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[int(y1):int(y2), int(x1):int(x2)] = 1.0
        mask_img = toImg(mask)
        mask_img.save('mask.png')
        return mask

    # https://pic4.zhimg.com/v2-5c44fbf2e0add1153571b0fc7c2a7673_r.jpg


# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channels, size_1, size_2):
#         self.gap = nn.AdaptiveAvgPool2d([1, 1])
#         self.fc1 = nn.Linear()

if __name__ == '__main__':
    madnet = MADNet(3, 1)
    print()

    from utils.data.dataset import FsodDataset
    from utils.data.process import pre_process_coco, read_single_coco
    from tqdm import tqdm
    from PIL import Image

    root = '../datasets/fsod'
    train_json = '../datasets/fsod/annotations/fsod_train.json'
    test_json = '../datasets/fsod/annotations/fsod_test.json'
    fsod = FsodDataset(root, train_json, support_shot=1, dataset_img_path='images')
    s_c_list_ori, q_c_list_ori, q_anns_list_ori, val_list_ori, val_anns_list_ori \
        = fsod.n_way_k_shot([1, 2, 3, 4, 5])
    s_c_list, q_c_list, q_anns_list, val_list, val_anns_list \
        = pre_process_coco(s_c_list_ori, q_c_list_ori,
                           q_anns_list_ori,
                           val_list_ori, val_anns_list_ori,
                           support_transforms=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Resize((320, 320))]),
                           query_transforms=transforms.Compose(
                               [transforms.ToTensor()]),
                           is_cuda=False, random_sort=True)
    pbar = tqdm(range(len(q_c_list)))
    continue_flag = False
    for index in pbar:
        q = q_c_list[index]
        target = q_anns_list[index]
        img = Image.open(q)
        img_draw = ImageDraw
        for tar in target:
            bbox = tar[0]['bbox']
            if bbox[2] <= 0.1 or bbox[3] <= 0.1:
                tqdm.write('id为{}的标注存在问题, 对应image_id为{}, 跳过此张图像'.format(tar[0]['id'], tar[0]['image_id']))
                continue_flag = True
        if continue_flag is True:
            continue_flag = False
            continue

        q, target = read_single_coco(q, target, label_ori=[1, 2, 3, 4, 5], query_transforms=transforms.Compose(
            [transforms.ToTensor()]), is_cuda=False)
        s = torch.ones([5, 3, 100, 100])
        image = ImageList(q, [q[0].shape[1:]])
        y = madnet.forward(s, q[0].unsqueeze(0), image=image, target=target)
        # 2019000014348
