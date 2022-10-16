import numpy as np

#--------------------------------------------#
#   生成基础的先验框
#--------------------------------------------#
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    r"""
    根据参数生成基础的先验框, 也就是创造anchor, 但是没有放到图上, 用于后期使用,
    生成(len(ratios) * len(anchor_scales)个框,
    每个框有四个参数, 故anchor_base的size为((len(ratios) * len(anchor_scales), 4),
    即(框数, 4)
    @param base_size: 基础框大小
    @param ratios: 长宽比
    @param anchor_scales: anchor缩放
    @return: 基础锚框, 用于后期放到图中使用, size: (框数, 4)
    """
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

#--------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
#--------------------------------------------#
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    r"""
    将基础先验框拓展到所有特征点上
    @param anchor_base: 基础先验框, size: (框数, 4)
    @param feat_stride: anchor采样的步长
    @param height: 特征图的高
    @param width: 特征图的宽
    @return: 所有特征点上的先验框矩阵, size: (K, A, 4)(K个采样点, 每个采样点A个锚框, 每个锚框的坐标)
    """
    #---------------------------------#
    #   计算网格中心点
    #---------------------------------#
    #     一个参数时，参数值为终点值，起点取默认值0，步长取默认值1。
    #     两个参数时，第一个参数为起点值，第二个参数为终点，步长取默认值1。
    #     三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长，其中步长支持小数。
    # 初始化x坐标
    shift_x             = np.arange(0, width * feat_stride, feat_stride)
    # 初始化y坐标
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    # 进行组合, 获得(x, y)对
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y)
    # np.ravel()将np转换为一维矩阵, 返回视图, 且修改np.ravel(), np相应的值也会受到影响
    # np.flatten()返回一维矩阵的拷贝, 修改时np不会受到影响
    # np.stack()将四个矩阵拼接起来
    # 为什么是(x,y,x,y)? anchor矩阵size(框数, 4), 对应其坐标
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

    #---------------------------------#
    #   每个网格点上的9个先验框
    #---------------------------------#
    A       = anchor_base.shape[0] # 基础先验框个数
    K       = shift.shape[0] # 采样坐标点的个数
    # 广播机制, anchor size: (K, A, 4), (每个采样点, 每个基础框, 坐标)
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    #---------------------------------#
    #   所有的先验框
    #---------------------------------#
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print("-----------------------\nbase_anchors: \n", nine_anchors)

    height, width, feat_stride  = 38,38,16
    anchors_all                 = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))

    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths  = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]

    # for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
    for i in range(len(anchors_all)):
        a = np.random.rand()
        if a > 0.99:
            rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
            ax.add_patch(rect)
    plt.show()
