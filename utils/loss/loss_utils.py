import torch
from torch import nn as nn
from torch.nn import functional as F


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    # def __init__(self, sigmoid=True, gamma=1.5, alpha=0.25):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.loss_fcn =  F.binary_cross_entropy_with_logits
        self.gamma = gamma
        # self.alpha = alpha

        # self.reduction = loss_fcn.reduction
        # self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, label, score):
        # loss = self.loss_fcn(pred, true, reduction='none')

        score, _ = torch.max(score, dim=-1)
        pred_prob = torch.sigmoid(pred)
        scale_factor = pred_prob
        zerolabel = scale_factor.new_zeros(pred.shape)
        loss = self.loss_fcn(pred, zerolabel,reduction='none') * scale_factor.pow(self.gamma)
        bg_class_ind = pred.size(-1)
        # pos = ((label >= 0) &
        #        (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
        # pos_label = label[pos].long()
        # positives are supervised by bbox quality (IoU) score
        # scale_factor = score[pos] - pred_prob[pos, pos_label]
        # for b in range(pred.size(0)):  # Loop over batch dimension
        for b, (pred_prob_elem, pred_elem, score_elem) in enumerate(zip(pred_prob, pred, score)):
            # Get the positions of the valid labels (labels that are >= 0 and < bg_class_ind)
            pos = ((label[b] >= 0) & (label[b] < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
            pos_label = label[b][pos].long()

            # Supervise positives by bbox quality (IoU) score
            scale_factor = score_elem[pos, pos_label] - pred_prob_elem[pos, pos_label]
            loss[b][pos, pos_label] = self.loss_fcn(pred_elem[pos, pos_label], score_elem[pos],
                                                    reduction='none') * scale_factor.abs().pow(self.gamma)
        # loss[pos,
        #      pos_label] = self.loss_fcn(pred[pos, pos_label], score[pos],
        #                        reduction='none') * scale_factor.abs().pow(self.gamma)

        # alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        # modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        # loss *= alpha_factor * modulating_factor

        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:  # 'none'
        return loss


class VariFocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                           self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss


class MimicLoss(nn.Module):
    def __init__(self, ):
        super(MimicLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss


class SCWDLoss(nn.Module):
    """
    s->n
    tr1: c_gain=1.0, pix_gain=0.25   3
    tr2: c_gain=1.0, pix_gain=0.125  1 √
    tr3: c_gain=1.5, pix_gain=0.25   2

    m->s
    tr1: c_gain=1.0, pix_gain=0.025   2
    tr2: c_gain=1.0, pix_gain=0.125  1 √
    tr3: c_gain=1.0, pix_gain=0.0

    """

    def __init__(self, tau=1.0, c_gain=1.0, pix_gain=0.125):
        super(SCWDLoss, self).__init__()
        self.tau = tau
        self.c_gain = c_gain
        self.pix_gain = pix_gain

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            t = t.view(-1, W * H)
            s = s.view(-1, W * H)
            cost = 0
            if self.c_gain != 0:
                # channel wise
                softmax_pred_T = F.softmax(t / self.tau,
                                           dim=1)  # [N*C, H*W]
                logsoftmax = torch.nn.LogSoftmax(dim=1)
                cost = self.c_gain * torch.sum(
                    softmax_pred_T * logsoftmax(t / self.tau) -
                    softmax_pred_T * logsoftmax(s / self.tau)) * (
                               self.tau ** 2)

            if self.pix_gain != 0:
                # spatial wise
                softmax_pred_T = F.softmax(t / self.tau,
                                           dim=0)  # [N*C, H*W]
                logsoftmax = torch.nn.LogSoftmax(dim=0)
                cost += self.pix_gain * torch.sum(
                    softmax_pred_T * logsoftmax(t / self.tau) -
                    softmax_pred_T * logsoftmax(s / self.tau)) * (
                                self.tau ** 2)
            if self.pix_gain == 0 or self.pix_gain == 0:
                losses.append(cost / (C * N))
            else:
                losses.append(cost / (C * N * 2))

        return sum(losses)


