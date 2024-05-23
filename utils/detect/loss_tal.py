"""
Loss functions
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.loss.loss_utils import smooth_BCE, QFocalLoss, CWDLoss, MimicLoss, SCWDLoss
from utils.detect.assigner.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.detect.assigner.tal.assigner import TaskAlignedAssigner, ExpFreeTaskAlignedAssigner
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


# ------sub loss------
class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False, iou='CIoU'):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.iou = iou
        assert iou in ('CIoU', 'DIoU', 'GIoU', 'EIoU', 'SIoU')
        # self.use_fel = use_fel

    def forward(self, pred_dist,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask):
        # iou loss
        # new
        bbox_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou_param_dict = {
            'box1': pred_bboxes[fg_mask],
            'box2': target_bboxes[fg_mask],
            'xywh': False,
            self.iou: True

        }

        iou = eval('bbox_iou(**iou_param_dict)')
        loss_iou = ((1.0 - iou) * bbox_weight).sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            # new
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):

        # new
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


# Damo-YOLO
class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t):
        super(FeatureLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(stu_channel, tea_channel, kernel_size=1, stride=1,
                      padding=0).to(device)
            for stu_channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        # self.feature_loss = CWDLoss()
        # self.feature_loss = MimicLoss()
        self.feature_loss = SCWDLoss()

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        return self.feature_loss(stu_feats, tea_feats)


# ------hyper loss------
class NNDetectionLoss:
    # Compute losses
    def __init__(self, model, use_dfl=True, iou='CIoU', detector='TOOD'):
        device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        m = de_parallel(model).model[-1]  # Detect() module

        # Define criteria
        self.hyp = model.hyp

        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        # Task-Aligned Assigner
        assert detector in ('TOOD', 'ExpFree')
        if detector == 'TOOD':
            self.assigner = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                                num_classes=self.nc,
                                                alpha=float(os.getenv('YOLOA', 0.5)),
                                                beta=float(os.getenv('YOLOB', 6.0)))
        elif detector == 'ExpFree':
            self.assigner = ExpFreeTaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                                num_classes=self.nc,
                                                alpha=float(os.getenv('YOLOA', 0.5)),
                                                beta=float(os.getenv('YOLOB', 1.0)))
        else:
            raise NotImplementedError('Unknown detector ')
        self.bbox_loss = BboxLoss(m.reg_max - 1,
                                  use_dfl=use_dfl,
                                  iou=iou).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)  # v8
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    # 舍弃了anchor
    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_distri, pred_scores = p
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # TAA
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.cls_loss(pred_scores, target_scores.to(dtype),
                                ).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class NNDetectionLossDistillFeature:
    # Compute losses
    def __init__(self,
                 model,
                 use_dfl=True,
                 iou='CIoU',
                 detector='TOOD'
                 ):
        device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3

        m = de_parallel(model).model[-1]  # Detect() module

        # Define criteria
        self.hyp = model.hyp
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        # Task-Aligned Assigner
        assert detector in ('TOOD', 'ExpFree')
        if detector == 'TOOD':
            self.assigner = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                                num_classes=self.nc,
                                                alpha=float(os.getenv('YOLOA', 0.5)),
                                                beta=float(os.getenv('YOLOB', 6.0)))
        elif detector == 'ExpFree':
            self.assigner = ExpFreeTaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                                       num_classes=self.nc,
                                                       alpha=float(os.getenv('YOLOA', 0.5)),
                                                       beta=float(os.getenv('YOLOB', 6.0)))
        else:
            raise NotImplementedError('Unknown detector ')
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl, iou=iou).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)  # v8
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __get_detect_loss__(self, p, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_distri, pred_scores = p
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # TAA
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.cls_loss(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        return loss, batch_size

    def __call__(self,
                 pred=None,
                 targets=None,
                 distill_loss=0):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        loss_stu, batch_size = self.__get_detect_loss__(pred, targets)

        loss[3] = distill_loss

        for idx in range(3):
            loss[idx] = loss_stu[idx]

        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, distill)
