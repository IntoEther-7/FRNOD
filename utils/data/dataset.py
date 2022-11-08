# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-16 16:34
import copy
import random
import time

from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from utils.dataset_tools.support_query_constructor import one_way_k_shot


# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

class FsodDataset(Dataset):
    def __init__(self, root, annFile, way=None, support_shot=2, img_transform=None,
                 target_transform=None,
                 seed=None, init=True, quick_test=False, dataset_img_path='images'):
        super(FsodDataset, self).__init__()
        self.num_mission = None
        self.mission = None
        self.sample_list = None
        self.root = root
        self.coco = COCO(annFile)
        time.sleep(0.001)
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.support_shot = support_shot
        self.quick_test = quick_test
        if seed:
            random.seed(seed)

        # 生成support和query
        self.support_list = {}
        self.query_list = {}
        self.query_anns_list = {}
        self.val_list = {}
        self.val_anns_list = {}
        if init:
            print('正在为每个类别生成support和query')
            for catId, cat in tqdm(self.coco.cats.items()):
                support, query, query_anns, val, val_anns = one_way_k_shot(root=self.root, dataset=self.coco,
                                                                           dataset_img_path=dataset_img_path,
                                                                           catId=catId,
                                                                           support_shot=self.support_shot,
                                                                           quick_test=self.quick_test)
                self.support_list.update({catId: support})
                self.query_list.update({catId: query})
                self.query_anns_list.update({catId: query_anns})
                self.val_list.update({catId: val})
                self.val_anns_list.update({catId: val_anns})

            if way:
                self.n_way_k_shot(way)

    def categories(self):
        return self.coco.cats

    def __len__(self):
        num_imgs = len(self.coco.imgs)
        num_anns = len(self.coco.anns)
        num_cats = len(self.coco.cats)
        return "该数据集有 %d 个类别, %d 张图像, %d 个标注\n选取了" % (num_cats, num_imgs, num_anns)

    def __getitem__(self, catId) -> (list, list, list):
        r"""
        返回catId的生成数据, 1 way k shot
        :param catId: 类别index
        :return: type: list -> support, qurey, qurey_anns
        """
        support = self.support_list[catId]
        qurey = self.query_list[catId]
        qurey_anns = self.query_anns_list[catId]
        return support, qurey, qurey_anns

    def triTuple(self, catId) -> (list, list, list, list, list, list):
        r"""
        生成三元组, (q_c, s_c, s_n), 其中sc和qc同类, 其类index为catId, sn为其他类, 随机抽取
        :param catId: c类
        :return: s_c, s_n, q_c, q_anns
        """
        sample_range = random.sample(self.coco.cats.keys(), 2)
        print('sample_range:', sample_range)
        print('catId:', catId)
        if catId in sample_range:
            sample_range.remove(catId)
        sample_index = sample_range[0]
        print('sample_index:', sample_index)
        s_c = self.support_list[catId]
        s_n = self.support_list[sample_index]
        q_c = self.query_list[catId]
        q_anns = self.query_anns_list[catId]
        val = self.val_list[catId]
        val_anns = self.val_anns_list[catId]
        return s_c, s_n, q_c, q_anns, val, val_anns

    def n_way_k_shot(self, catIds):
        s_c_list = [self.support_list[catId] for catId in catIds]
        q_c_list = [self.query_list[catId] for catId in catIds]
        q_anns_list = [self.query_anns_list[catId] for catId in catIds]
        val_list = [self.val_list[catId] for catId in catIds]
        val_anns_list = [self.val_anns_list[catId] for catId in catIds]
        return s_c_list, q_c_list, q_anns_list, val_list, val_anns_list

    def get_n_way_k_shot(self, mission_id):
        catIds = self.mission[mission_id]
        s_c_list = []
        q_c_list = []
        q_anns_list = []
        for catId in catIds:
            s_c, s_n, q_c, q_anns = self.__getitem__(catId)
            s_c_list.append(s_c)
            q_c_list.append(q_c)
            q_anns_list.append(q_anns)
        return s_c_list, q_c_list, q_anns_list
