# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-16 16:34
import random
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from utils.dataset_tools.support_query_constructor import one_way_k_shot


class FsodDataseet(Dataset):
    def __init__(self, root, annFile, support_shot=2, query_shot=5, img_transform=None, target_transform=None,
                 seed=None):
        super(FsodDataseet, self).__init__()
        self.root = root
        self.coco = COCO(annFile)
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.support_shot = support_shot
        self.query_shot = query_shot
        if seed:
            random.seed(seed)
        self.support_list = []
        self.query_list = []
        self.query_anns_list = []
        print('正在为每个类别生成support和query')
        for catId, cat in tqdm(self.coco.cats.items()):
            support, query, query_anns = one_way_k_shot(root=self.root, dataset=self.coco, catId=catId,
                                                        support_shot=self.support_shot,
                                                        query_shot=self.query_shot)
            self.support_list.append(support)
            self.query_list.append(query)
            self.query_anns_list.append(query_anns)

    def categories(self):
        return self.coco.cats

    def __len__(self):
        num_imgs = len(self.coco.imgs)
        num_anns = len(self.coco.anns)
        num_cats = len(self.coco.cats)
        return "该数据集有 %d 个类别, %d 张图像, %d 个标注\n选取了" % (num_cats, num_imgs, num_anns)

    def __getitem__(self, catId):
        support = self.support_list[catId]
        qurey = self.query_list[catId]
        qurey_anns = self.query_anns_list[catId]
        return support, qurey, qurey_anns
