import torch
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, SequentialLR, LinearLR
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        num_epochs: int,
        steps_per_epoch: int,
        max_lr: float,
        min_lr: float = 1e-8,
):
    """
    Factory function to get a learning rate scheduler.

    Args:
        optimizer: The optimizer to wrap.
        scheduler_type: 'one_cycle' or 'cosine'.
        num_epochs: Total number of training epochs.
        steps_per_epoch: Number of batches per epoch.
        max_lr: The peak learning rate (used for OneCycle and initial Cosine).
        min_lr: Minimum learning rate for cosine decay.
    """
    total_steps = num_epochs * steps_per_epoch

    # ---------------------------------------------------------
    # 1. OneCycleLR
    # ---------------------------------------------------------
    # OneCycle handles its own warmup/cooldown internally.
    # We ignore 'warmup_epochs' here because OneCycle uses 'pct_start'.
    if scheduler_type == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps+num_epochs,
            pct_start=0.3,  # Standard: 30% warmup, 70% cooldown
            div_factor=25.0,  # Initial LR = max_lr / 25
            final_div_factor=1000.0  # Final LR = max_lr / (25 * 1000)
        )

    # ---------------------------------------------------------
    # 2. Cosine Annealing (with optional Warmup)
    # ---------------------------------------------------------
    elif 'cosine' in scheduler_type:
        # Main Cosine Scheduler
        # If we have warmup, the cosine part runs for (total - warmup) steps
        warmup_steps = 3 * steps_per_epoch if 'warmup' in scheduler_type else 0

        decay_steps = total_steps - warmup_steps

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=min_lr
        )

        # If no warmup is requested, just return the standard cosine
        if warmup_steps <= 0:
            return cosine_scheduler

        # If warmup IS requested, we chain Linear + Cosine using SequentialLR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.001,  # Start at 0.1% of max_lr
            end_factor=1.0,  # Reach 100% of max_lr
            total_iters=warmup_steps
        )

        # SequentialLR runs schedulers[0] for milestones[0] steps, then switches
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Supported: 'one_cycle', 'cosine'")


def get_cosine_scheduler(
        optimizer: Optimizer,
        total_steps: int,
        use_warmup: bool = True,
        warmup_len: float = None,
        use_restarts: bool = False,
        num_cycles: float = 2.0,
        min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Creates a learning rate scheduler for Cosine Annealing.
    Includes flags to toggle linear warmup and warm restarts.

    Args:
        optimizer: The PyTorch optimizer (e.g., AdamW).
        total_steps: Total number of training steps (epochs * batches per epoch).
        use_warmup: If True, gradually increases LR from 0 to max over warmup_steps.
        warmup_len: Number of steps dedicated to the warmup phase.
        use_restarts: If True, implements hard restarts back to max LR.
        num_cycles: Number of hard restart cycles if use_restarts is True.
        min_lr_ratio: The floor the LR will decay to (e.g., 0.1 means 10% of base LR).
    """

    warmup_len = int(total_steps * warmup_len) if use_warmup else 0

    def lr_lambda(current_step: int):
        # Phase 1: Linear Warmup
        if current_step < warmup_len:
            return float(current_step) / float(max(1, warmup_len))

        # Phase 2: Cosine Annealing
        progress = float(current_step - warmup_len) / float(max(1, total_steps - warmup_len))

        # End of training constraint (forces it to stay at the minimum floor)
        if progress >= 1.0:
            return min_lr_ratio

        # Calculate cycle position
        if use_restarts:
            # Multiplies progress by cycles. Modulo 1.0 isolates the decimal to loop the curve
            cycle_progress = (progress * num_cycles) % 1.0
        else:
            # Standard single decay curve
            cycle_progress = progress

        # Cosine math: 0.5 * (1 + cos(pi * x)) smoothly decays from 1.0 to 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

        # Apply the minimum learning rate floor and return the multiplier
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
