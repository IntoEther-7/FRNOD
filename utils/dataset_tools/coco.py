# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-15 15:26
import random

import cv2
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import transforms


def read_coco_detection(root='G:\\Code\\FRNOD\\', dataset_root='datasets/fsod/images',
                        annFile='datasets/fsod/annotations/fsod_train.json',
                        img_transform=transforms.ToTensor(), target_transform=None,
                        transforms=None):
    return CocoDetection(root=root + dataset_root, annFile=root + annFile, transform=img_transform,
                         target_transform=target_transform,
                         transforms=transforms)


def show_dataset(dataloader, show_anns=False, random_sort=True, start_index=0, show_num=10):
    if random_sort:
        show_list = random.sample(range(len(dataloader)), show_num)
    else:
        show_list = [i for i in range(start_index, start_index + show_num)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in show_list:
        img, labels = dataloader.__getitem__(i)

        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if show_anns:
            bboxes = []
            ids = []

            for label in labels:
                bboxes.append(xywh2xyxy(label['bbox']))
                ids.append(label['category_id'])
            for box, id_ in zip(bboxes, ids):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), thickness=5)
                cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                            thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imshow('test', img)
        if ord('q') == cv2.waitKey(0):
            break

    cv2.destroyAllWindows()


def xywh2xyxy(xywh: list):
    x, y, w, h = xywh
    xyxy = [x, y, x + w, y + h]
    return xyxy


if __name__ == '__main__':
    fsod_dataset = read_coco_detection()
    show_dataset(fsod_dataset, show_anns=True)
