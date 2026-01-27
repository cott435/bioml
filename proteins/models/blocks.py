from torch import nn as nn, Tensor


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
        batch_norm=True,
        dilation=1
    ):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        activation_fn = nn.ReLU if activation == 'relu' else nn.GELU

        self.dwconv = nn.Conv1d(dim,dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(dim) if batch_norm else ConvLayerNorm(dim)

        self.pw_block = nn.Sequential(*[
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            activation_fn(),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        ])

    def forward(self, x):  # (B, C, L)
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw_block(x)
        return x + residual


class Conv1dInvBottleNeck(nn.Module):
    def __init__(self, dim, expansion_ratio=4, kernel_size=3, dilation=1,
                 activation='relu', dropout=0.1, batch_norm=True):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        activation_fn = nn.ReLU if activation == 'relu' else nn.GELU
        norm_fn = nn.BatchNorm1d if batch_norm else ConvLayerNorm

        self.block = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            norm_fn(hidden_dim),
            activation_fn(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation, padding=kernel_size//2, groups=hidden_dim),
            norm_fn(hidden_dim),
            activation_fn(),

            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            norm_fn(dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # (B, C, L)
        return x + self.block(x)


