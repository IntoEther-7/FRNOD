# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-17 15:56
import json
import os.path
import random
import time

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.data.dataset import FsodDataset
from PIL import Image


def get_image_list(root, img_path_list: list, dataset: FsodDataset):
    # print('统计图像路径{}'.format(root))
    for img_info in tqdm(dataset.coco.imgs.values()):
        img_path = os.path.join(root, 'images', img_info['file_name'])
        img_path_list.append(img_path)
    return img_path_list
    # return random.sample(img_path_list, 10)


def get_mean_and_std(img_path_list: list):
    tqdm.write('计算图像像素均值')
    time.sleep(0.001)
    pixel_num = 0
    pixel_sum = np.zeros((3,))
    for img_path in tqdm(img_path_list):
        img_np = np.array(Image.open(img_path).convert('RGB')) / 255.  # 高, 宽, 通道 (719, 1024, 3)
        # 各通道加和整幅图的像素
        pixel_sum += np.sum(img_np, axis=(0, 1))
        # 这个图片的像素个数
        pixel_num += img_np.shape[0] * img_np.shape[1]
        # if len(img_np.shape) == 3:
        #     if img_np.shape[2] == 3:
        #         r, g, b = np.split(img_np, 3, axis=2)
        #         r_img_sum = np.sum(r)
        #         g_img_sum = np.sum(g)
        #         b_img_sum = np.sum(b)
        #         r_list.append(r_img_sum)
        #         g_list.append(g_img_sum)
        #         b_list.append(b_img_sum)
        #         pixel_num_img = img_np.shape[0] * img_np.shape[1]
        #         pixel_num_list.append(pixel_num_img)
        #     elif img_np.shape[2] == 4:
        #         r, g, b, _ = np.split(img_np, 4, axis=2)
        #         r_img_sum = np.sum(r)
        #         g_img_sum = np.sum(g)
        #         b_img_sum = np.sum(b)
        #         r_list.append(r_img_sum)
        #         g_list.append(g_img_sum)
        #         b_list.append(b_img_sum)
        #         pixel_num_img = img_np.shape[0] * img_np.shape[1]
        #         pixel_num_list.append(pixel_num_img)

    mean = pixel_sum / pixel_num
    tqdm.write('计算图像像素标准差')
    time.sleep(0.001)

    # 各通道与均值差, 再平方累积和
    pixel_pow2 = np.zeros((3,))
    for img_path in tqdm(img_path_list):
        img_np = np.array(Image.open(img_path).convert('RGB')) / 255.  # 高, 宽, 通道 (719, 1024, 3)
        # 均值差, 平方, 再加和
        pixel_pow2 += np.sum(np.power((img_np - mean), 2), axis=(0, 1))
    # 除以像素个数, 再开方
    std = np.sqrt(pixel_pow2 / pixel_num)
    # r_pow2, g_pow2, b_pow2 = 0, 0, 0
    # for img_path in tqdm(img_path_list):
    #     img_np = np.array(Image.open(img_path).convert('RGB'))  # 高, 宽, 通道 (719, 1024, 3)
    #     r, g, b = np.split(img_np, 3, axis=2)
    #     r_pow2 += np.sum(np.power(r - r_mean, 2))
    #     g_pow2 += np.sum(np.power(g - g_mean, 2))
    #     b_pow2 += np.sum(np.power(b - b_mean, 2))
    # if len(img_np.shape) == 3:
    #     if img_np.shape[2] == 3:
    #         r, g, b = np.split(img_np, 3, axis=2)
    #         r_pow2 += np.sum(np.power(r - r_mean, 2))
    #         g_pow2 += np.sum(np.power(g - r_mean, 2))
    #         b_pow2 += np.sum(np.power(b - r_mean, 2))
    #     elif img_np.shape[2] == 4:
    #         r, g, b, _ = np.split(img_np, 4, axis=2)
    #         r_pow2 += np.sum(np.power(r - r_mean, 2))
    #         g_pow2 += np.sum(np.power(g - r_mean, 2))
    #         b_pow2 += np.sum(np.power(b - r_mean, 2))

    return mean, std


# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


if __name__ == '__main__':
    root = '../../datasets/fsod'
    train_json = '../../datasets/fsod/annotations/fsod_train.json'
    test_json = '../../datasets/fsod/annotations/fsod_test.json'
    fsod_train = FsodDataset(root, train_json, support_shot=2, init=False)
    fsod_test = FsodDataset(root, test_json, support_shot=2, init=False)

    # __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    # COCO数据集的均值和方差为：
    #     mean_vals = [0.471, 0.448, 0.408]
    #     std_vals = [0.234, 0.239, 0.242]
    # ImageNet数据集的均值和方差为：
    #     mean_vals = [0.485, 0.456, 0.406]
    #     std_vals = [0.229, 0.224, 0.225]
    for i in range(1000):
        img_path_list = []
        img_path_list = get_image_list(root, img_path_list, fsod_train)
        img_path_list = get_image_list(root, img_path_list, fsod_test)
        tqdm.write('----------{}----------'.format(i))
        time.sleep(0.001)
        mean, std = get_mean_and_std(img_path_list)
        time.sleep(0.001)
        tqdm.write('mean: ' + str(mean))
        tqdm.write('std: ' + str(std))
        tqdm.write('写入JSON文件')
        time.sleep(0.001)
        with open('config/fsod-{}.json'.format(i), 'w') as f:
            dataset_config = {'root': '../../datasets/fsod',
                              'train_json': '../../datasets/fsod/annotations/fsod_train.json',
                              'test_json': '../../datasets/fsod/annotations/fsod_test.json',
                              'mean': mean.tolist(),
                              'std': std.tolist()}
            json.dump(dataset_config, f)
