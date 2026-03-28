from __future__ import annotations

import torch

from .schedulers import get_cosine_scheduler


class TokenOptimizer:
    """Builds and stores optimizer, scheduler, and parameter groups for token models."""

    def __init__(
            self,
            model,
            total_steps: int,
            lr_restarts: bool = False,
            in_lr_ratio: float = 5,
            out_lr_ratio: float = 10,
            base_lr: float = 1e-4,
            warmup_len: float = 0.1,
            weight_decay: float = 0.01,
    ):
        self.model = model
        self.total_steps = total_steps
        self.out_lr_ratio = out_lr_ratio
        self.base_lr = base_lr

        param_groups = [
            {"params": model.stack.parameters(), "lr": base_lr},
            {"params": model.in_proj.parameters(), "lr": base_lr * in_lr_ratio},
            {"params": model.out_proj.parameters(), "lr": base_lr * out_lr_ratio},
        ]
        if model.inp_norm is not None:
            param_groups.append({"params": model.inp_norm.parameters(), "lr": base_lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            self.total_steps,
            warmup_len=warmup_len,
            use_restarts=lr_restarts,
        )

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
