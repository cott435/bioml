from torch import nn as nn, Tensor
import torch
from timm.layers import DropPath


def _activation_cls(name: str):
    if name == 'relu':
        return nn.ReLU
    if name == 'gelu':
        return nn.GELU
    if name == 'silu':
        return nn.SiLU
    raise ValueError(f"Unknown activation: {name}")


class ConvLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(1, 1, dim) * 1e-2)
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        gx = (x.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = self.gamma * (x * nx) + self.beta + x
        if mask is not None:
            out = out * mask
        return out


class ConvFFN(nn.Module):
    """
    TokenActivation-style conv feed-forward block.
    Expects/returns (B, L, C).
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
        activation='gelu',
        norm=False,
        **kwargs,
    ):
        super().__init__()
        activation_fn = _activation_cls(activation)
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x * mask if mask is not None else x
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class ResConvFFN(nn.Module):
    """
    Residual version of ConvFeedForward1DBlock.
    Expects/returns (B, L, C).
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
        activation='gelu',
        norm=True,
        drop_path=0.0,
        layer_scale_init_value = 0.00,
        **kwargs,
    ):
        super().__init__()
        self.ff = ConvFFN(
            dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
            activation=activation,
            norm=norm,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1)), requires_grad=True)
            if layer_scale_init_value > 0 else None)

    def forward(self, x, mask=None):
        block = self.ff(x, mask=mask)
        if self.gamma is not None:
            block = self.gamma * block
        return x + self.drop_path(block)


class ConvNeXt1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        expansion_ratio=4,
        dropout=0.1,
        activation='gelu',
        dilation=1,
        drop_path=0.0,
        layer_scale_init_value=0.00,
        **kwargs,
    ):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        activation_fn = _activation_cls(activation)

        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            groups=dim,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim)

        self.exp = nn.Linear(dim, hidden_dim)
        self.activation = activation_fn()
        self.grn = GRN(hidden_dim)
        self.cmp = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1)), requires_grad=True)
            if layer_scale_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None):  # (B, L, C)
        x = x * mask if mask is not None else x
        res = x
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.exp(x)
        x = self.activation(x)
        x = self.grn(x, mask=mask)
        x = self.cmp(x)
        x = self.dropout(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x * mask if mask is not None else x
        return res + self.drop_path(x)


class Conv1dInvBottleNeck(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=4,
        kernel_size=3,
        dilation=1,
        layer_scale_init_value=0.0,
        activation='relu',
        dropout=0.1,
        batch_norm=False,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        activation_fn = _activation_cls(activation)
        norm_fn = nn.BatchNorm1d if batch_norm else ConvLayerNorm

        self.expand = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            norm_fn(hidden_dim),
            activation_fn(),
        )
        self.dw = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size // 2),
                groups=hidden_dim,
            ),
            norm_fn(hidden_dim),
            activation_fn(),
        )
        self.reduce = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            norm_fn(dim),
            nn.Dropout(dropout),
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None):  # (B, L, C)
        res = x
        x = x.transpose(1, 2)
        x = self.expand(x)
        if mask is not None:
            x = x * mask.transpose(1, 2)
        x = self.dw(x)
        x = self.reduce(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return res + self.drop_path(x)


class PreActResNet1DBlock(nn.Module):
    """
    Pre-activation ResNet block.
    Expects/returns (B, L, C).
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
        activation='gelu',
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        activation_fn = _activation_cls(activation)

        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = activation_fn()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)

        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = activation_fn()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        residual = x
        out = x.transpose(1, 2)

        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out).transpose(1, 2)

        if mask is not None:
            out = out * mask

        return residual + self.drop_path(out)


