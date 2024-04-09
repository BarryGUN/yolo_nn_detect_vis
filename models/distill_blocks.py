import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.ops import  roi_pool

from models.conv import DWConv, Conv


class SE(nn.Module):
    def __init__(self, channel, ration=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ration, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c ,h, w --> b, c, 1, 1
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])

        return x * fc


class Generation(nn.Module):
    def __init__(self, channel, ration=16):
        super(Generation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ration, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c ,h, w --> b, c, 1, 1
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


# class ROI(nn.Module):
#     def __init__(self, output_size):
#         super(ROI, self).__init__()
#         self.output_size = output_size
#
#     def forward(self, feature_map, rois):
#         """
#         feature_map: input feature map，shape(B, C, H, W)
#         rois: interesting region，shape(N, 4)，N is the number of regions，each region expressed by coordinates locate in top-left corner and bottom-right corner
#         """
#         batch_size, channels, _, _ = feature_map.shape
#         _, _, H, W = rois.shape
#         output = roi_pool(input=feature_map, )
#
#         return


class TranROI(nn.Module):

    def __init__(self, dim):
        super(TranROI, self).__init__()
        self.a = DWConv(dim, dim, k=3, s=1)
        self.v = nn.Identity()
        self.linear = Conv(dim, dim, k=1, s=1, act=False)

    def forward(self, tea, stu):
        tea_mask = self.a(tea)
        return tea_mask * self.v(stu)
