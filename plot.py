import math

import matplotlib.pyplot as plt

# x轴数据
from numpy import arange, exp, log, sin, cos


def align_new(x, beta):
    return beta * x / ((beta + 1) - x)


def align_org(x, beta):
    return pow(x, beta)


def cosannle(epoch, epochs):
    return ((1 - cos(epoch * math.pi / epochs)) / 2) * (0.1 - 1) + 1


def plot(plt, x, y, label):
    plt.plot(x, y, label=label)  # 绘制折线图
    return plt


def plot2d(plt):
    x = arange(0, 1, 0.01)
    plt.figure()

    plt = plot(plt, x, align_new(x, beta=0.095), label='new_align')
    plt = plot(plt, x, align_org(x, beta=6), label='org_align')
    # plt = plot(plt, x, cosannle(x, epochs=100), label='log2')

    plt.legend(loc='upper left')
    plt.title('Simple Line Plot')  # 设置标题
    plt.xlabel('X-axis')  # 设置x轴标签
    plt.ylabel('Y-axis')  # 设置y轴标签
    plt.show()  # 显示图形


if __name__ == '__main__':
    plot2d(plt)

