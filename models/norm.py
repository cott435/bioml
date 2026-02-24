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
