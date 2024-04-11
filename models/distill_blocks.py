import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from models.conv import DWConv, Conv


class GaussBlur(nn.Module):
    def __init__(self, channels, sigma=0.5, kernel_size=3):
        super(GaussBlur, self).__init__()
        self.blurConv = self.get_gaussian_kernel(kernel_size=kernel_size,
                                                 sigma=sigma,
                                                 channels=channels)

    def get_gaussian_kernel(self, kernel_size, sigma, channels):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                    groups=channels,
                                    bias=False, padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def forward(self, x):
        return self.blurConv(x)

class TranROI(nn.Module):

    def __init__(self, dim):
        super(TranROI, self).__init__()
        self.a = DWConv(dim, dim, k=3, s=1)
        self.v = nn.Identity()
        self.linear = Conv(dim, dim, k=1, s=1, act=False)

    def forward(self, tea, stu):
        tea_mask = self.a(tea)
        return tea_mask * self.v(stu)
