import torch
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, SequentialLR, LinearLR


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
            total_steps=total_steps,
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