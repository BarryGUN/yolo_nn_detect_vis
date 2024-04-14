import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from models.conv import DWConv, Conv


class TranROI(nn.Module):

    def __init__(self, dim):
        super(TranROI, self).__init__()
        self.a = DWConv(dim, dim, k=3, s=1)
        self.v = nn.Identity()
        self.linear = Conv(dim, dim, k=1, s=1, act=False)

    def forward(self, tea, stu):
        tea_mask = self.a(tea)
        return tea_mask * self.v(stu)


class MatROI(nn.Module):

    def __init__(self, dim):
        super(MatROI, self).__init__()
        self.a = Conv(dim, dim, k=3, s=1)
        self.v = nn.Identity()
        self.linear = Conv(dim, dim, k=1, s=1, act=False)

    def forward(self, tea, stu):
        tea_mask = self.a(tea)
        return torch.matmul(tea_mask, self.v(stu))
