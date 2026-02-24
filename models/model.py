import torch
import torch.nn as nn
from .blocks import Conv1dInvBottleNeck, ConvNeXt1DBlock, ConvLayerNorm
from timm.layers import trunc_normal_
from .norm import MaskedInstanceNorm1d

blocks = {'Conv1dInvBottleNeck': Conv1dInvBottleNeck, 'ConvNeXt1DBlock': ConvNeXt1DBlock,}

class Conv1dStack(nn.Module):

    def __init__(self, dim, layers=1, expansion_ratio=4, kernel_size=3,
                 activation='relu', dropout=0.1, batch_norm=True, block_type='Conv1dInvBottleNeck',
                 drop_path_rate=0.0, layer_scale_init_value=0.0):
        super().__init__()
        if isinstance(block_type, str):
            assert block_type in blocks
        block = blocks[block_type] if isinstance(block_type, str) else block_type
        dilations = [2 ** (i % 5) for i in range(layers)]  # restarts after 16
        dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]

        self.stack = nn.ModuleList([
            block(dim, expansion_ratio=expansion_ratio, kernel_size=kernel_size, dilation=dilations[i],
                  dropout=dropout, batch_norm=batch_norm, activation=activation, drop_path=dp_rate[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(layers)
        ])
        self.norm = ConvLayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        for layer in self.stack:
            x = layer(x, mask=mask)
        return self.norm(x)


class SequenceActiveSiteHead(nn.Module):

    def __init__(self, in_dim, out_dim=1, layers=1, hidden_dim=None, activation='relu', batch_norm=False,
                 dropout=0.1, block_type='Conv1dInvBottleNeck', kernel_size=5, inp_norm=True, final_bias=0,
                 drop_path_rate=0.0, expansion_ratio=4, layer_scale_init_value=0.0, feature_dropout=None,
                 feature_dropout_first=True, token_dropout=None):
        super().__init__()
        self.inp_norm = MaskedInstanceNorm1d(in_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim) if hidden_dim else nn.Identity()
        hidden_dim = hidden_dim or in_dim
        self.stack = Conv1dStack(hidden_dim, dropout=dropout,
                                 activation=activation, batch_norm=batch_norm, layers=layers,
                                 block_type=block_type, kernel_size=kernel_size,
                                 drop_path_rate=drop_path_rate, expansion_ratio=expansion_ratio,
                                 layer_scale_init_value=layer_scale_init_value)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.feature_dropout_first = feature_dropout_first
        self.feature_dropout = nn.Dropout1d(feature_dropout) if feature_dropout else nn.Identity()
        self.token_dropout = nn.Dropout1d(token_dropout) if token_dropout else nn.Identity()
        self._init_weights(final_bias)

    def _init_weights(self, final_bias):
        nn.init.kaiming_normal_(self.in_proj.weight, mode='fan_in', nonlinearity='linear')
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.constant_(self.out_proj.bias, final_bias)


    def forward(self, embeds, mask=None, sigmoid=False):
        x = self.inp_norm(embeds.transpose(1, 2), mask=mask).transpose(1, 2)
        x = self.token_dropout(x)
        if self.feature_dropout_first:
            x = self.feature_dropout(x.transpose(1, 2)).transpose(1, 2)
        x = self.in_proj(x)
        if not self.feature_dropout_first:
            x = self.feature_dropout(x.transpose(1, 2))
        else:
            x = x.transpose(1, 2)
        mask = mask.unsqueeze(1) if mask is not None else None
        x = self.stack(x, mask=mask).transpose(1, 2).squeeze(-1)
        x = self.out_proj(x).squeeze(-1)
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




