from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass



# =======================================================================
import sys
sys.path.append('/content/drive/MyDrive/Wound_tissue_segmentation/wound_lib/segmentation_models_pytorch/losses')

from typing import Optional
from functools import partial

import torch
from torch.nn.modules.loss import _Loss
from _functional import focal_loss_with_logits
from constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["FocalLoss"]


class FocalLoss(_Loss, base.Loss):
    def __init__(
        self,
        mode: str = 'binary',
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long() # mkd commented

                # cls_y_true = y_true[:, cls, ...] # mkd added
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss



# MCC loss
class MCCLoss(_Loss, base.Loss):
    def __init__(self, eps: float = 1e-5):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.
        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class
        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss
        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)
        Returns:
            torch.Tensor: loss value (1 - mcc)
        """

        bs = y_true.shape[0]

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss
        
        

# Soft BCE loss
__all__ = ["SoftBCEWithLogitsLoss"]


class SoftBCEWithLogitsLoss(nn.Module):

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss
        
# =================================================================
import torch.nn as nn
import torch.nn.functional as FF

__all__ = ["DynamicCEAndSCELoss"]

class DynamicCEAndSCELoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(DynamicCEAndSCELoss, self).__init__()
        self.alpha = 1.0
        self.beta = 0.5
        self.using_weight = True
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.ignore_index = ignore_index
        
        self.__name__ = "DynamicCEAndSCELoss"


    def forward(self, pred, labels):
        nclass = pred.size(1)
        
        # print('=====================nClasses', nclass)
        
        if self.using_weight:
            weights = torch.max(torch.softmax(pred, dim=1), dim=1, keepdim=True).values
            weights[weights>0.8] = 1.0
            weights = torch.clamp(weights, min=1e-10, max=1.0).detach()
        else:
            weights = 0.5

        # CCE
        labels = torch.argmax(labels, dim=1) # mkd added, reversing one hot
        
        not_ignore_mask = labels.ne(self.ignore_index).float()
        # print('================================', pred.shape, labels.shape, not_ignore_mask.shape)
        ce = self.cross_entropy(pred, labels)
        # print('=====================CE', ce.shape)
        ce = torch.mean(ce * not_ignore_mask * (1-weights))
        # RCE
        pred = FF.softmax(pred, dim=1)
        label_one_hot = FF.one_hot(torch.clamp(labels, min=0, max=nclass - 1), nclass).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_one_hot = label_one_hot.permute([0, 3, 1, 2]).contiguous()

        rce = -1 * pred * torch.log(label_one_hot) # mkd changed

        rce = torch.sum(rce, dim=1)
        rce = torch.mean(rce * not_ignore_mask * weights)

        # logging.info((self.alpha * ce, self.beta * rce))
        # Loss
        loss = self.alpha * ce + self.beta * rce
        
        # print('**********, loss:', loss)
        
        return loss

    # def forward(self, preds, target):
    #     return sum([self._forward(preds[i], target) for i in range(len(preds))])

   

# ==================================================================================
class WeightedCELoss(torch.nn.Module):
    def __init__(self, 
        weight = None,
        ignore_index = -100,
        reduction = 'mean'
        ):
        super(WeightedCELoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

        self.__name__ = "WeightedCELoss"


    def forward(self, pred, labels):
        return self.cross_entropy(pred, labels.argmax(dim=1)) # *** assuming labels and pred are one-hot encoded ***