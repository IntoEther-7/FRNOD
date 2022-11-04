# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-01 0:26
import json
import numpy as np
from matplotlib import pyplot as plt

# 每张图的loss
loss_list = []
with open('weights/results/loss_per_image.json', 'r') as f:
    d: dict = json.load(f)
    if len(d) >= 160:
        epoch_full = len(d) // 160
        last_epoch = len(d) % 160
        for e in range(epoch_full):
            for m in range(160):
                loss_list.extend(d['loss_epoch{}_mission_{}'.format(e + 1, m + 1)])
        for m in range(last_epoch):
            loss_list.extend(d['loss_epoch{}_mission_{}'.format(epoch_full + 1, m + 1)])
    else:
        for m in range(len(d)):
            loss_list.extend(d['loss_epoch1_mission_{}'.format(m + 1)])
    # for i in d.values():
    #     loss_list.extend(i)

ll = np.array(loss_list)

fig = plt.figure()
plt.plot(ll)
plt.show()
plt.close(fig)
