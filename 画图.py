# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-10-28 10:35
import json
import numpy as np
from matplotlib import pyplot as plt

loss_list = []
loss_avg_list = []
with open('weights/loss.json', 'r') as f:
    d = json.load(f)
    loss_list = d['loss_list']
    loss_avg_list = d['loss_avg_list']

ll = np.array(loss_list)
lal = np.array(loss_avg_list)

plt.figure()
plt.plot(lal)
plt.show()