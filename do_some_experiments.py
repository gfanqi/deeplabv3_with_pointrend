import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2

ori_img = cv2.imread('./images/dog.jpg')
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
H, W = ori_img.shape
# print(img.shape)
# 进行自己的处理
# ...
array = np.load('./core/test.npy')[2]


def plot(pred):
    fig = plt.figure()
    # 创建3d图形的两种方式
    # ax = Axes3D(fig)
    pred = np.squeeze(pred)
    H, W = pred.shape[:2]
    ax = fig.add_subplot(111, )
    # X, Y value
    X = np.arange(0, W, 1)
    Y = np.arange(0, H, 1)
    X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
    # R = np.sqrt(X ** 2 + Y ** 2)
    # height value
    # Z = np.sin(R)
    print(pred.shape)
    Z = pred
    # rstride:行之间的跨度  cstride:列之间的跨度
    # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
    # vmax和vmin  颜色的最大值和最小值
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
    # offset : 表示等高线图投射到指定页面的某个刻度
    # ax.contourf(X, Y, Z, zdir='z', offset=0)
    # 设置图像z轴的显示范围，x、y轴设置方式相同
    # ax.set_zlim(0, 2)
    # plt.contourf(X,Y,Z)
    plt.contour(X, Y, Z, [0, 0.1, 0.3,  0.5,  0.7,  0.9,
                          1.1,  1.3,  1.5, 1.7,  1.9, 2], cmap=plt.get_cmap('rainbow'))
    plt.show()
    plt.legend()
    # plt.savefig('temp.png')


plot(array)
