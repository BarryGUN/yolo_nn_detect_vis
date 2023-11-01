import torch
from torch import nn as nn

from models.conv import SRepConv, Conv
from models.blocks import SRepBottleneck, FastRepV3Block, C3fBlock, C3sBlock, \
    Bottleneck,  RTMDetBottleneck, C3tBlock, RTMDet2Bottleneck, RepNeXtBottleneck


class BottleneckLan(nn.Module):

    def __init__(self, in_channel, out_channel, n, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(out_channel * e)

        self.conv_in_1 = Conv(in_channel, c_, k=1, s=1, p=0)
        self.conv_in_2 = Conv(in_channel, c_, k=1, s=1, p=0)
        self.blocks = nn.ModuleList(
            Bottleneck(c_, c_, shortcut=shortcut, e=1)
            for _ in range(n)
        )

        self.conv_out = Conv((2 + n) * c_, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        x_cat = [self.conv_in_1(x), self.conv_in_2(x)]
        for block in self.blocks:
            x_cat.append(block(x_cat[-1]))
        x = self.conv_out(torch.cat(x_cat, 1))
        return x

class RepLan(nn.Module):

    def __init__(self, in_channel, out_channel, n, e=0.5):
        super().__init__()
        c_ = int(out_channel * e)

        self.conv_in_1 = Conv(in_channel, c_, k=1, s=1, p=0)
        self.conv_in_2 = Conv(in_channel, c_, k=1, s=1, p=0)
        self.blocks = nn.ModuleList(
            FastRepV3Block(c_, c_, n=2)
            for _ in range(n)
        )

        self.conv_out = Conv((2 + n) * c_, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        x_cat = [self.conv_in_1(x), self.conv_in_2(x)]
        for block in self.blocks:
            x_cat.append(block(x_cat[-1]))
        x = self.conv_out(torch.cat(x_cat, 1))
        return x




class BottleneckLanSRep(nn.Module):

    def __init__(self, in_channel, out_channel, n, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(out_channel * e)

        self.conv_in_1 = Conv(in_channel, c_, k=1, s=1, p=0)
        self.conv_in_2 = Conv(in_channel, c_, k=1, s=1, p=0)
        self.blocks = nn.ModuleList(
            SRepBottleneck(c_, c_, shortcut=shortcut, e=1)
            for _ in range(n)
        )

        self.conv_out = Conv((2 + n) * c_, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        x_cat = [self.conv_in_1(x), self.conv_in_2(x)]
        for m in self.blocks:
            x_cat.append(m(x_cat[-1]))
        return self.conv_out(torch.cat(x_cat, 1))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3f(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)
        self.conv_1 = Conv(c1, c_, k=1, s=1)
        self.conv_2 = Conv(c1, c_, k=1, s=1)
        self.conv_3 = Conv(int(c_ / e), c2, k=1, s=1)
        self.block = C3fBlock(c_, n=n)

    def forward(self, x):
        return  self.conv_3(
            torch.cat(
                [self.conv_1(x), self.block(self.conv_2(x))],
                dim=1
            )
        )


class C2FastRep(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * 0.5)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C3fBlock(self.c, n=1) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2x(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * 0.5)  # hidden channels
        self.cv1 = SRepConv(c1, 2 * self.c, 1, 1)
        self.cv2 = SRepConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(SRepConv(self.c, self.c, 3, 1) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))



class C2sFastRep(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * 0.5)  # hidden channels
        self.cv1 = SRepConv(c1, 2 * self.c, 1, 1)
        self.cv2 = SRepConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C3sBlock(self.c, n=1, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))



class C2t(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * 0.5)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C3tBlock(self.c, n=1, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class C2e(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(RTMDetBottleneck(self.c, self.c, shortcut, e=1) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class C2g(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(RTMDet2Bottleneck(self.c, self.c, shortcut, e=1) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class MS2f(nn.Module):

    def __init__(self, c1, c2, n=1, merge=False, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_2 = int(c2 * e)
        self.slices = n + 1
        self.merge = merge
        c_1 = c_2 * self.slices  # hidden channels

        self.cv1 = Conv(c1, c_1, 1, 1)
        self.bottleneck_series = nn.ModuleList(
            (RTMDet2Bottleneck(c_2, c_2, shortcut=False, k=(3, 5, 3), e=1) for _ in range(n))
        )
        self.cv2 = Conv(c_1, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(self.slices, 1))
        for i, m in enumerate(self.bottleneck_series):
            if self.merge:
                y[i + 1] = m(y[i + 1]) + y[i]
            else:
                y[i + 1] = m(y[i + 1])

        return self.cv2(torch.cat(y, dim=1))


class C2RepNeXt(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(RepNeXtBottleneck(self.c, self.c, shortcut, e=1) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


