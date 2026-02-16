from torch import nn as nn, Tensor
import torch
from timm.layers import DropPath

class ConvLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXt1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        expansion_ratio=4,
        dropout=0.1,
        activation='relu',
        batch_norm=False,
        dilation=1,
        drop_path=0.0,
        layer_scale_init_value=0.0
    ):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        activation_fn = nn.ReLU if activation == 'relu' else nn.GELU

        self.dw = nn.Conv1d(dim,dim,  # depthwise
            kernel_size=kernel_size,
            padding=dilation*(kernel_size//2),
            groups=dim,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(dim) if batch_norm else ConvLayerNorm(dim)

        self.pw = nn.Sequential(*[  # pointwise
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            activation_fn(),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        ])
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None):  # (B, C, L)
        x = x * mask if mask is not None else x
        res = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw(x)
        x = self.gamma * x
        return res + self.drop_path(x)


class Conv1dInvBottleNeck(nn.Module):
    def __init__(self, dim, expansion_ratio=4, kernel_size=3, dilation=1, layer_scale_init_value=0.0,
                 activation='relu', dropout=0.1, batch_norm=False, drop_path=0.0):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        activation_fn = nn.ReLU if activation == 'relu' else nn.GELU
        norm_fn = nn.BatchNorm1d if batch_norm else ConvLayerNorm

        self.expand = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            norm_fn(hidden_dim),
            activation_fn(),
        )
        self.dw = nn.Sequential(  # depthwise
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size//2), groups=hidden_dim),
            norm_fn(hidden_dim),
            activation_fn(),
        )
        self.reduce = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            norm_fn(dim),
            nn.Dropout(dropout)
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None):  # (B, C, L)
        res = x
        x = self.expand(x)
        x = x * mask if mask is not None else x
        x = self.dw(x)
        x = self.reduce(x)
        x = self.gamma * x
        return res + self.drop_path(x)


class PreActResNet1DBlock(nn.Module):
    """
    Pre-activation ResNet block (He et al., 2016).
    Best for Deep Networks and Fine-tuning on Transformer embeddings.
    Structure: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    """

    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # Note: We use padding to maintain sequence length
        padding = dilation * (kernel_size // 2)

        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.GELU()  # GELU aligns better with ESM
        self.conv1 = nn.Conv1d(dim, dim, kernel_size,
                               padding=padding, dilation=dilation)

        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size,
                               padding=padding, dilation=dilation)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x

        # Block 1
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        # Block 2
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if mask is not None:
            out = out * mask  # Apply mask to the branch ONLY

        return residual + out

if __name__ == "__main__":
    x = torch.randn(100, 300, 224)
    dp = DropPath(0.2)
    l = nn.Linear(224, 224)
    xx = l(x)
    final = x+dp(xx)

    d=1

