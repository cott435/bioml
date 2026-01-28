import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        # targets should be float in [0,1]
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )

        pt = torch.exp(-bce_loss)  # pt = probability of true class
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean() if mask is None else loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum() if mask is None else torch.mean(loss.sum(-1) / mask.sum(-1))
        return loss