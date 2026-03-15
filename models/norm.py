import torch
import torch.nn as nn


class MaskedInstanceNorm1d(nn.Module):
    """
    Instance Normalization for 1D sequences that properly handles padding/masks.

    Args:
        num_features: number of channels/features (C)
        eps: small value added to variance for numerical stability
        momentum: for running mean/var (usually not used in pure instance norm)
        affine: whether to use learnable scale and bias
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, compress_tails=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.compress_tails = compress_tails

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, mask=None):
        """
        x:    (B, C, L)   - batch, channels, sequence length
        mask: (B, L)      - boolean or float mask (True/padding = 1, keep = 0)
                          or None → assume no padding / full sequences

        Returns normalized tensor of same shape
        """
        B, C, L = x.shape

        if mask is None or B==1:
            mean = x.mean(dim=2, keepdim=True)  # (B, C, 1)
            var = x.var(dim=2, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Masked version - only normalize over non-padded positions
            if mask.dtype != torch.bool:
                mask = mask.bool()  # convert float mask (0/1) to boolean

            # Expand mask to (B, C, L)
            mask = mask.unsqueeze(1).expand(-1, C, -1)  # (B, C, L)

            # Count valid elements per sequence/channel
            valid_count = mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, C, 1)

            # Masked mean
            masked_x = x * mask.float()  # zero out padded
            sum_x = masked_x.sum(dim=2, keepdim=True)  # (B, C, 1)
            mean = sum_x / valid_count

            # Masked variance
            diff_sq = (x - mean) ** 2 * mask.float()
            sum_diff_sq = diff_sq.sum(dim=2, keepdim=True)
            var = sum_diff_sq / valid_count

            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.compress_tails:
            x_norm = exponential_tail_compress(x_norm, threshold=3.0, alpha=1.5)
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        return x_norm

def exponential_tail_compress(x: torch.Tensor, threshold=3.0, alpha=1.0):
    abs_x = x.abs()
    sign = x.sign()

    excess = abs_x - threshold
    compressed_excess = alpha * (1 - torch.exp(-excess / alpha))

    new_mag = torch.where(
        abs_x <= threshold,
        abs_x,
        threshold + compressed_excess
    )

    return sign * new_mag

import torch
import torch.nn as nn

class MaskedBatchNorm1d(nn.Module):
    """
    1D BatchNorm that:
      1. Ignores masked/padded tokens when computing batch stats.
      2. Uses the SAME stats for every sub-batch in your accumulation loop
         (so you get true batch-norm behaviour while still doing grad accumulation).
      3. Updates running stats exactly like nn.BatchNorm1d.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        # These are set once per optimizer step (outside the accumulation loop)
        self.batch_mean = None
        self.batch_var = None

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1.0)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def set_batch_stats(self, batch_list):
        """
        Call this ONCE per optimizer step, BEFORE _accumulate_grads / _get_grads.
        batch_list = list of (x, mask, Y) tuples exactly as your loader yields.
        """
        if not self.training or not batch_list:
            return

        device = batch_list[0][0].device
        dtype = batch_list[0][0].dtype

        sum_x = torch.zeros(self.num_features, dtype=dtype, device=device)
        sum_x2 = torch.zeros(self.num_features, dtype=dtype, device=device)
        total_count = torch.tensor(0.0, dtype=dtype, device=device)

        for x, _, mask in batch_list:
            if x.numel() == 0:
                continue
            B, L, D = x.shape
            assert D == self.num_features

            x_flat = x.reshape(-1, D)
            mask_flat = mask.reshape(-1).to(dtype)          # bool → 0/1 float, or 0/1 already
            mask_float = mask_flat.unsqueeze(-1)            # (N, 1)

            sum_x += (x_flat * mask_float).sum(dim=0)
            sum_x2 += ((x_flat ** 2) * mask_float).sum(dim=0)
            total_count += mask_flat.sum()

        if total_count.item() == 0:
            self.batch_mean = torch.zeros(self.num_features, device=self.weight.device)
            self.batch_var = torch.ones(self.num_features, device=self.weight.device)
        else:
            self.batch_mean = (sum_x / total_count).to(self.weight.device)
            self.batch_var = (sum_x2 / total_count).to(self.weight.device) - (self.batch_mean ** 2)

        # Update running stats (exactly like nn.BatchNorm1d)
        if self.track_running_stats:
            self.num_batches_tracked += 1
            m = self.momentum if self.momentum is not None else 1.0 / self.num_batches_tracked.item()
            self.running_mean = (1 - m) * self.running_mean + m * self.batch_mean
            self.running_var = (1 - m) * self.running_var + m * self.batch_var

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Called inside your model forward (on each sub-batch).
        x shape: (B, seq_len, dim) or any shape ending with `num_features`.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.num_features)

        if self.training and self.batch_mean is not None:
            mean = self.batch_mean
            var = self.batch_var
        else:
            # eval or first step fallback
            mean = self.running_mean if self.running_mean is not None else x_flat.mean(dim=0)
            var = self.running_var if self.running_var is not None else x_flat.var(dim=0, unbiased=False)

        x_norm = (x_flat - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm.view(orig_shape)

