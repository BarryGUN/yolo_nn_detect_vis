import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from models.conv import DWConv, Conv


class ROI(nn.Module):

    def __init__(self, roi_size=3):
        super(ROI, self).__init__()
        self.roi_size = roi_size

    def forward(self, x, roi, stride):
        return roi_align(input=x,
                         boxes=roi,
                         output_size=(self.roi_size, self.roi_size),
                         spatial_scale=stride)



