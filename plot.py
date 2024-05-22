import math

import matplotlib.pyplot as plt

# x轴数据
from numpy import arange, exp, log, sin, cos


def cosannle(epoch, epochs):
    return ((1 - cos(epoch * math.pi / epochs)) / 2) * (0.1 - 1) + 1


def plot(plt, x, y, label):
    plt.plot(x, y, label=label)  # 绘制折线图
    return plt


def plot2d(plt):
    x = arange(1, 100, 1)
    plt.figure()

    plt = plot(plt, x, cosannle(x, epochs=100), label='log2')

    plt.legend(loc='upper left')
    plt.title('Simple Line Plot')  # 设置标题
    plt.xlabel('X-axis')  # 设置x轴标签
    plt.ylabel('Y-axis')  # 设置y轴标签
    plt.show()  # 显示图形


if __name__ == '__main__':
    plot2d(plt)
