# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-28 10:35
import json
import numpy as np
from matplotlib import pyplot as plt

loss_list = []
loss_avg_list = []
loss_bepoch = []
for i in range(1, 2):
    with open('weights/results/loss_bepoch{}.json'.format(i), 'r') as f:
        d = json.load(f)
        # loss_list.extend(d['loss_list'])
        loss_avg_list.extend(d['loss_avg_list'])
        loss_bepoch.append(np.mean(d['loss_avg_list']))

ll = np.array(loss_list)
lal = np.array(loss_avg_list)
lbe = np.array(loss_bepoch)

fig = plt.figure()
plt.plot(ll)
plt.show()
plt.close(fig)

fig = plt.figure()
plt.plot(lal)
plt.show()
plt.close(fig)

fig = plt.figure()
plt.plot(lbe)
plt.show()
plt.close(fig)
