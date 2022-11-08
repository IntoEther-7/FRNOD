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


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.out_channels = 1024
        # 定义分支1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 224, kernel_size=(1, 3), stride=1, padding=1),
            nn.Conv2d(224, 256, kernel_size=(3, 1), stride=1, padding=0)
        )
        # 定义分支2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 192, kernel_size=(5, 1), stride=1, padding=2),
            nn.Conv2d(192, 224, kernel_size=(1, 5), stride=1, padding=0),
            nn.Conv2d(224, 256, kernel_size=(7, 1), stride=1, padding=3),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=0)
        )
        # 定义分支3
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        )
        # 定义分支4
        self.branch4 = nn.Conv2d(in_channels, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        r"""

        :param x:
        :return: (way, 1024, s1, s2)
        """
        # 计算分支1
        branch1_out = self.branch1(x)
        # 计算分支2
        branch2_out = self.branch2(x)
        # 计算分支3
        branch3_out = self.branch3(x)
        # 计算分支4
        branch4_out = self.branch4(x)

        # 拼接四个不同分支得到的通道，作为输出
        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.inception = Inception(in_channels)
        self.conv = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.inception(x)
        saliency_map = self.conv(x)
        return saliency_map


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(-1, self.in_channels)
        x = F.relu(self.fc1(x), inplace=True)
        channel_attention = torch.sigmoid(self.fc2(x))
        return channel_attention.unsqueeze(2).unsqueeze(3)


class MADNet(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MADNet, self).__init__()
        self.in_channels = in_channels
        self.PA = PixelAttention(in_channels)
        self.CA = ChannelAttention(in_channels, reduction=reduction)

    def forward(self, support, query, image=None, target=None, index=None):
        channel_attention = self.CA.forward(support)
        pixel_attention = F.softmax(self.PA.forward(query), dim=1)
        fg_attention = pixel_attention[:, :1, :, :]
        out = query * channel_attention * fg_attention
        loss_attention = None
        if self.training:
            mask = self._generate_mask(image, target, index)
            loss_attention = self._compute_attention_loss(mask, fg_attention)
        return out, loss_attention

    def _compute_attention_loss(self, mask: Tensor, fg_attention: Tensor):
        t = transforms.Resize(fg_attention.shape[2:])
        mask = t(mask.unsqueeze(0).unsqueeze(0)).to(fg_attention.device)
        loss_attention = F.binary_cross_entropy_with_logits(fg_attention, mask)
        return loss_attention

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
