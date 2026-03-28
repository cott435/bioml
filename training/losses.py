import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self, reduction='mean', pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        bce_loss = self.bce(logits, targets)
        loss = bce_loss * mask if mask is not None else bce_loss

        if self.reduction == 'mean':
            return loss.mean() if mask is None else loss.sum() / mask.sum()
        elif self.reduction == 'targets':
            return loss.sum() if mask is None else torch.mean(loss.sum(-1) / targets.sum(-1))
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def step(self):
        pass


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none', smoothing=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, logits, targets, mask=None):
        targets = targets.clip(0, 0.9) if self.smoothing else targets
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )

        pt = torch.exp(-bce_loss)  # pt = probability of true class
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        loss = loss * mask if mask is not None else loss

        if self.reduction == 'mean':
            return loss.mean() if mask is None else loss.sum() / mask.sum()
        elif self.reduction == 'targets':
            return loss.sum() if mask is None else torch.mean(loss.sum(-1) / targets.sum(-1))
        return loss

    def step(self):
        pass

    def get_current_params(self):
        return {}

    @staticmethod
    def reduce_loss(loss, targets, mask, reduce=True):
        pos_mask = targets * mask
        neg_mask = (1 - targets) * mask
        pos_loss_sum = (loss * pos_mask).sum(dim=-1)
        neg_loss_sum = (loss * neg_mask).sum(dim=-1)
        pos_count = pos_mask.sum(dim=-1) + 1e-8
        neg_count = neg_mask.sum(dim=-1) + 1e-8
        pos_avg = pos_loss_sum / pos_count
        neg_avg = neg_loss_sum / neg_count
        if reduce:
            return pos_avg + neg_avg
        else:
            return pos_avg, neg_avg


class DynamicBinaryFocalLoss(BinaryFocalLoss):
    def __init__(self,
                 total_steps,
                 start_alpha=0.25, end_alpha=0.25,
                 start_gamma=0.0, end_gamma=2.0,
                 start_ratio=0.1, end_ratio=0.8,
                 reduction='mean'):
        """
        Inherits from BinaryFocalLoss to dynamically schedule alpha and gamma.

        Args:
            start_alpha: High initial weight for the rare positive class.
            end_alpha: Lower final weight once gamma takes over.
            start_gamma: Starts at 0.0 (Standard BCE) to establish variance.
            end_gamma: Scales up to focus on hard examples.
        """
        # Initialize parent with starting values
        super().__init__(alpha=start_alpha, gamma=start_gamma, reduction=reduction)

        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.start_step = start_ratio * total_steps
        self.end_step = end_ratio * total_steps
        self.current_step = 0

    def step(self):
        """
        Updates alpha and gamma based on the current training step.
        Call this once per batch in your training loop.
        """
        if self.current_step < self.start_step:
            self.current_step += 1
        elif self.current_step < self.end_step:
            self.current_step += 1
            progress = (self.current_step - self.start_step) / (self.end_step - self.start_step)

            self.alpha = self.start_alpha + progress * (self.end_alpha - self.start_alpha)
            self.gamma = self.start_gamma + progress * (self.end_gamma - self.start_gamma)

    def get_current_params(self):
        """Helper to log parameters during training."""
        return {'alpha': self.alpha, 'gamma': self.gamma}

