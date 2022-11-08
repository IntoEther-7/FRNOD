# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-01 0:26
import json
import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt

# 每个mission的loss
from tqdm import tqdm

loss_avg_list = []
loss_avg_list_val = []
loss_avg_epoch = []
loss_avg_epoch_val = []

root = 'weights_fsod_r50/results'
# root = 'weights_fsod/results'
# root = 'weights_fsod_conv4/results'
with open(os.path.join(root, 'loss_per_image.json'), 'r') as f:
    d: dict = json.load(f)
    if len(d) >= 160:
        epoch_full = len(d) // 160
        last_epoch = len(d) % 160
        for e in tqdm(range(epoch_full)):
            for m in range(160):
                loss_this_mission = float(torch.Tensor(d['loss_epoch{}_mission_{}'.format(e + 1, m + 1)]).mean())
                loss_avg_list.append(loss_this_mission)
            loss_avg_epoch.append(np.array(loss_avg_list).mean())
        for m in range(last_epoch):
            loss_this_mission = float(torch.Tensor(d['loss_epoch{}_mission_{}'.format(epoch_full + 1, m + 1)]).mean())
            loss_avg_list.append(loss_this_mission)
        loss_avg_epoch.append(np.array(loss_avg_list).mean())
    else:
        for m in tqdm(range(len(d))):
            loss_this_mission = float(torch.Tensor(d['loss_epoch1_mission_{}'.format(m + 1)]).mean())
            loss_avg_list.append(loss_this_mission)
        loss_avg_epoch.append(np.array(loss_avg_list).mean())

    # for i in d.values():
    #     loss_avg_list.extend(i)

with open(os.path.join(root, 'loss_per_image_val.json')) as f:
    d: dict = json.load(f)
    if len(d) >= 160:
        epoch_full = len(d) // 160
        last_epoch = len(d) % 160
        for e in tqdm(range(epoch_full)):
            for m in range(160):
                loss_this_mission = float(torch.Tensor(d['loss_epoch{}_mission_{}_val'.format(e + 1, m + 1)]).mean())
                loss_avg_list_val.append(loss_this_mission)
            loss_avg_epoch_val.append(np.array(loss_avg_list_val).mean())
        for m in range(last_epoch):
            loss_this_mission = float(
                torch.Tensor(d['loss_epoch{}_mission_{}_val'.format(epoch_full + 1, m + 1)]).mean())
            loss_avg_list_val.append(loss_this_mission)
        loss_avg_epoch_val.append(np.array(loss_avg_list_val).mean())
    else:
        for m in tqdm(range(len(d))):
            loss_this_mission = float(torch.Tensor(d['loss_epoch1_mission_{}_val'.format(m + 1)]).mean())
            loss_avg_list_val.append(loss_this_mission)
        loss_avg_epoch_val.append(np.array(loss_avg_list_val).mean())

lal = np.array(loss_avg_list)
lalv = np.array(loss_avg_list_val)
fig = plt.figure(figsize=(12, 6.75), dpi=320.)
plt.plot([x + 1 for x in range(lal.shape[0])], lal, linewidth=0.5)
plt.plot([x + 1 for x in range(lalv.shape[0])], lalv, linewidth=0.5)
plt.legend(['loss', 'loss_val'])
plt.draw()
plt.savefig(os.path.join(root, 'loss_per_mission.png'))
plt.show()
plt.close(fig)

lae = np.array(loss_avg_epoch)
laev = np.array(loss_avg_epoch_val)
fig = plt.figure(figsize=(6, 3.375), dpi=320.)
plt.plot([x + 1 for x in range(lae.shape[0])], lae, linewidth=1)
plt.plot([x + 1 for x in range(laev.shape[0])], laev, linewidth=1)
plt.legend(['loss', 'loss_val'])
plt.draw()
plt.savefig(os.path.join(root, 'loss_per_epoch.png'))
plt.show()
plt.close(fig)
