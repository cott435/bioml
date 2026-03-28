import torch
import torch.nn as nn
from .blocks import (
    Conv1dInvBottleNeck,
    ConvNeXt1DBlock,
    ConvFFN,
    ResConvFFN,
    PreActResNet1DBlock,
)
from timm.layers import trunc_normal_
from .norm import MaskedInstanceNorm1d, MaskedBatchNorm1d

blocks = {
    'Conv1dInvBottleNeck': Conv1dInvBottleNeck,
    'ConvNeXt1DBlock': ConvNeXt1DBlock,
    'ConvFFN': ConvFFN,
    'ResConvFFN': ResConvFFN,
    'PreActResNet1DBlock': PreActResNet1DBlock,
}

norms = {
    'batch': MaskedBatchNorm1d,
    'instance': MaskedInstanceNorm1d}

class Stack1D(nn.Module):

    def __init__(self, dim, layers=1, expansion_ratio=4, kernel_size=3,
                 activation='gelu', dropout=0.1, block_type='ConvNeXt1DBlock',
                 final_norm=False, drop_path_rate=0.0, **kwargs):
        super().__init__()
        if isinstance(block_type, str):
            assert block_type in blocks
        block = blocks[block_type] if isinstance(block_type, str) else block_type
        dilations = [2 ** (i % 5) for i in range(layers)]  # restarts after 16
        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]

        self.stack = nn.ModuleList([
            block(dim, expansion_ratio=expansion_ratio, kernel_size=kernel_size, dilation=dilations[i],
                  dropout=dropout, activation=activation, drop_path=dp_rate[i])
            for i in range(layers)
        ])
        self.norm = nn.LayerNorm(dim) if final_norm else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            #feats = m.in_features if hasattr(m, 'in_features') else m.in_channels
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        for layer in self.stack:
            x = layer(x, mask=mask)
        return self.norm(x)


class TokenActivationHead(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=None, final_bias=0,
                 feature_dropout=None, token_dropout=None, inp_norm='instance', **kwargs):
        super().__init__()
        assert inp_norm is None or inp_norm in norms

        self.inp_norm = norms[inp_norm](in_dim) if inp_norm is not None else None
        self.feature_dropout = nn.Dropout1d(feature_dropout) if feature_dropout else nn.Identity()
        self.token_dropout = nn.Dropout1d(token_dropout) if token_dropout else nn.Identity()
        self.in_proj = nn.Linear(in_dim, hidden_dim) if hidden_dim else nn.Identity()
        self.proj_norm = nn.LayerNorm(hidden_dim)
        hidden_dim = hidden_dim or in_dim

        self.stack = Stack1D(hidden_dim, **kwargs)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self._init_weights()
        # just subtract final bias and keep param bias at 0 to avoid weight decay issues
        self.final_bias = final_bias

    def _init_weights(self):
        if isinstance(self.in_proj, nn.Linear):
            nn.init.kaiming_normal_(self.in_proj.weight, mode='fan_in', nonlinearity='linear')
            if self.in_proj.bias is not None:
                nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        #nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.out_proj.bias)


    def forward_(self, embeds, mask=None, sigmoid=False):
        x = self.inp_norm(embeds.transpose(1, 2), mask=mask).transpose(1, 2) if self.inp_norm is not None else embeds
        x_norm = x
        x_proj = self.in_proj(x)

        x = self.token_dropout(x)
        x = self.feature_dropout(x.transpose(1, 2)).transpose(1, 2)
        x = self.in_proj(x)


        mask = mask.unsqueeze(-1) if mask is not None else None
        x = self.stack(x, mask=mask)
        x = self.out_proj(x).squeeze(-1) + self.final_bias
        return torch.sigmoid(x) if sigmoid else x

    def forward(self, embeds, mask=None, sigmoid=False):
        x = self.inp_norm(embeds.transpose(1, 2), mask=mask).transpose(1, 2) if self.inp_norm is not None else embeds
        x = self.token_dropout(x)
        x = self.feature_dropout(x.transpose(1, 2)).transpose(1, 2)
        x = self.in_proj(x)
        x = self.proj_norm(x)
        mask = mask.unsqueeze(-1) if mask is not None else None
        x = self.stack(x, mask=mask)
        x = self.out_proj(x).squeeze(-1) + self.final_bias
        return torch.sigmoid(x) if sigmoid else x

class SequenceInteractionHead(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_layers: int = 3,
            expansion_ratio: int = 4,
            kernel_size: int = 3,
            dropout: float = 0.1,
            output_dim: int | None = None,
            final_norm: bool = True,
            matmul_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        # Shared stack of blocks — applied to both sequences
        blocks = []
        current_dim = embed_dim
        for i in range(num_layers):
            blocks.append(Conv1dInvBottleNeck(
                current_dim,
                expansion_ratio=expansion_ratio,
                kernel_size=kernel_size,
                dropout=dropout,
            ))


        self.transform = nn.Sequential(*blocks)

        self.final_norm = nn.LayerNorm(self.output_dim) if final_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, self.output_dim, bias=False) if self.output_dim != embed_dim else nn.Identity()
        self.matmul_norm = matmul_norm

    def forward(
            self,
            emb1: torch.Tensor,  # (B, len1, embed_dim)
            emb2: torch.Tensor,  # (B, len2, embed_dim)
    ) -> torch.Tensor:
        """
        Returns:
            interaction: (B, len1, len2) similarity matrix
        """
        # Apply same transformation to both sequences
        t1 = self.transform(emb1)  # (B, len1, embed_dim)
        t2 = self.transform(emb2)  # (B, len2, embed_dim)

        t1 = self.proj(t1)
        t2 = self.proj(t2)
        t1 = self.final_norm(t1)
        t2 = self.final_norm(t2)

        if self.matmul_norm:
            t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-8)
            t2 = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)

        # (B, len1, embed_dim) @ (B, embed_dim, len2) → (B, len1, len2)
        interaction = torch.matmul(t1, t2.transpose(1, 2))

        return interaction


