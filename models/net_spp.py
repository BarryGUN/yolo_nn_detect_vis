import warnings

import torch
from torch import nn as nn

from models.blocks import SRepBottleneck
from models.conv import SRepConv, CBL, Conv


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # self.m = SoftPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPCSPC(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SimSPPCSPC(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super(SimSPPCSPC, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv1(x))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return torch.cat((y1, y2), dim=1)


class C2fSimSPP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super(C2fSimSPP, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        # self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y1 = self.cv5(torch.cat([y[-1]] + [m(y[-1]) for m in self.m], 1))
        return torch.cat((y1, y[0]), dim=1)


class SimSPPHCSPC(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=(5, 9, 13, 17)):
        super(SimSPPHCSPC, self).__init__()
        c_ = c2 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv((len(k) + 1) * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv1(x))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return torch.cat((y1, y2), dim=1)


class RepSPPCSPC(SimSPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        self.cv4 = SRepConv(c_, c_, kernel_size=3, stride=1)

        self.cv6 = SRepConv(c_, c_, kernel_size=3, stride=1)


class RepSPPCSPCL(RepSPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c1, c_, 1, 1)
        self.cv5 = CBL(4 * c_, c_, 1, 1)
        self.cv7 = CBL(2 * c_, c2, 1, 1)


class RepSPPCSP(SimSPPCSPC):
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        self.cv4 = nn.Identity()
        self.cv6 = SRepConv(c_, c_, 3, 1)


class RepSPPCSPC2(SimSPPCSPC):
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        self.cv7 = SRepConv(c_, c_, kernel_size=3, stride=1)

    def forward(self, x):
        x1 = self.cv4(self.cv1(x))
        y1 = self.cv7(self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1))))
        y2 = self.cv2(x)
        return torch.cat((y1, y2), dim=1)


class RepSPPCSPSB(SimSPPCSPC):
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        # self.cv6 = SRepBottleneck(c_, c_, shortcut=False, e=1.1)
        self.cv6 = SRepBottleneck(c_, c_, shortcut=False, e=1)


class RepSPPCSPSC(RepSPPCSPSB):
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        # self.cv6 = SRepBottleneck(c_, c_, shortcut=False, e=1.1)
        self.cv6 = nn.Sequential(
            SRepConv(c_, c_, stride=1, kernel_size=3),
            SRepBottleneck(c_, c_, shortcut=False, e=1)
        )


class RepSPPCSPAC(SimSPPCSPC):
    def __init__(self, c1, c2, n=1, k=(5, 9, 13)):
        super().__init__(c1, c2, n, k=k)
        c_ = c2 // 2  # hidden channels
        # self.cv6 = SRepBottleneck(c_, c_, shortcut=False, e=1.1)
        self.cv6 = nn.Sequential(
            nn.ModuleList(
                SRepConv(c_, c_, stride=1, kernel_size=3)
                for _ in range(n))

        )



class SPPFCSP(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2,n=1, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = SRepConv(c1, c_, 1, 1)
        self.cv2 = SRepConv(c1, c_, 1, 1)
        self.cv3 = SRepConv(c_ * 4, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = SRepConv(c_, c_, 3, 1)
        self.cv5 = SRepConv(c_ * 2, c2, 1, 1)
        # self.m = SoftPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x_1 = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x_1)
            y2 = self.m(y1)
            return self.cv5(torch.cat((self.cv4(self.cv3(torch.cat((x_1, y1, y2, self.m(y2)), 1))), self.cv2(x)), 1))