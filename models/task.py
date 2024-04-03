# YOLONN 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

from models.blocks import RepBlock, QARepBlock, Bottleneck, \
    RepVGGBlock, QARepVGGBlock, CBFuse, CBLinear, ReConvFuse
from models.conv import DWConv, GhostConv, DeformConv2d, ConvTranspose, DWConvTranspose2d
from models.head import NNDetect
from models.net import C2f, C2ELAN, LightC2ELAN
from models.net_spp import SPPF, \
    SPPCSPC, SPP

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLONN root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.general import LOGGER, make_divisible
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, scale_img, time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class BaseModel(nn.Module):
    # YOLOnn base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        i = 0
        distill_feature = []
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            # save inject layer for distill
            if self.distill and i in self.inject_layer:
                distill_feature.append(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if not self.distill:
            return x
        else:
            return x, distill_feature
        # return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        rep_layer_num = 0
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, (RepVGGBlock, QARepVGGBlock)):
                m.switch_to_deploy()
                rep_layer_num += 1
        LOGGER.info(f"{colorstr('Rep Model: ')}deploy mode, total:{rep_layer_num} layers")
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()

        if isinstance(m, NNDetect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            # m.grid = list(map(fn, m.grid))
        return self


class DetectionModel(BaseModel):
    # YOLONN detection model
    def __init__(self, cfg='yolonn-vis.yaml', ch=3, nc=None, scale='n', inject_layer=None,
                 distill=False):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if 'scale' not in self.yaml.keys():
            self.yaml['scale'] = scale

        self.distill = distill
        self.inject_layer = inject_layer
        if self.distill:
            self.inject_layer = inject_layer
            model_res = parse_model(deepcopy(self.yaml),
                                    ch=[ch],
                                    scale=self.yaml['scale'],
                                    out_ch_index=self.inject_layer)  # model, savelist

            self.model, self.save, self.inject_layer_ch = model_res
            # print(type(model_res[0]))
        else:
            self.model, self.save = parse_model(deepcopy(self.yaml),
                                                ch=[ch],
                                                scale=self.yaml['scale'],
                                                out_ch_index=None)  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()

        if isinstance(m, NNDetect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            if self.distill:
                forward = lambda x: self.forward(x)[0][0] if isinstance(m, NNDetect) else self.forward(x)
            else:
                forward = lambda x: self.forward(x)[0] if isinstance(m, NNDetect) else self.forward(x)
            # forward = lambda x: self.forward(x)[0] if isinstance(m, NNDetect) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            if isinstance(m, NNDetect):
                m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLONN augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


Model = DetectionModel  # retain YOLONN 'Model' class for backwards compatibility


def parse_model(d, ch, scale, out_ch_index=None):  # model_dict, input_channels(3)
    # Parse a YOLONN model.yaml dictionary
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'in':>3}{' ':>3}{'out':<10} {'arguments':<30}")
    # anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    # nc, gd, gw, act = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    max_channels = float('inf')
    gd, gw, = 1.0, 1.0
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))

    if scales:
        if scale in ['n', 's', 'm', 'x', 'l']:
            gd, gw, max_channels = scales[scale]
        else:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        gd, gw, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    layers, save, c2, c1_pr = [], [], ch[-1], ch[-1]  # layers, savelist, ch out
    distill_layer_ch = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (nn.Conv2d, Conv, ConvTranspose, GhostConv, MixConv2d, DWConv, DeformConv2d,
                 nn.ConvTranspose2d, DWConvTranspose2d, Focus, QARepBlock, Bottleneck,
                 RepBlock, SPP, SPPF, SPPCSPC, C2f, C2ELAN, LightC2ELAN):
            c1, c2 = ch[f], args[0]
            c1_pr = c1
            if c2 != nc:  # if not output
                c2 = make_divisible(min(c2, max_channels) * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in (C2f, C2ELAN, LightC2ELAN):
                args.insert(2, n)
                n = 1
        elif m in (nn.BatchNorm2d,):
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (CBFuse, ReConvFuse):
            c2 = ch[f[-1]]
            if m is ReConvFuse:
                args.insert(1, c2)
        elif m in (CBLinear,):
            c2 = args[0]
            cho = []
            for out in c2:
                cho.append(make_divisible(min(out, max_channels) * gw, 8))
            c2 = cho
            c1 = ch[f]
            args = [c1, c2, *args[1:]]

        # TODO: channel, gw, gd
        elif m in (NNDetect,):
            args.append([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(
            f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{c1_pr:>3}{" ":>3}{str(c2):<10} {str(args):<30}')  # print
        # LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
        if out_ch_index is not None and i in out_ch_index:
            distill_layer_ch.append(c2)

    if out_ch_index is None:
        return nn.Sequential(*layers), sorted(save)
    else:
        return nn.Sequential(*layers), sorted(save), distill_layer_ch
    # return nn.Sequential(*layers), sorted(save)

