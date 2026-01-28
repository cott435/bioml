import torch
import torch.nn as nn
from .blocks import Conv1dInvBottleNeck, ConvNeXt1DBlock, ConvLayerNorm

blocks = {'Conv1dInvBottleNeck': Conv1dInvBottleNeck, 'ConvNeXt1DBlock': ConvNeXt1DBlock,}

class Conv1dStack(nn.Module):

    def __init__(self, in_dim, out_dim=None, hidden_dim=None, layers=1, expansion_ratio=4, kernel_size=3, dilation=1,
                 activation='relu', dropout=0.1, batch_norm=True, final_bias=True, block_type='Conv1dInvBottleNeck'):
        super().__init__()
        assert block_type in blocks
        self.inp_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1) if hidden_dim else nn.Identity()
        hidden_dim = hidden_dim or in_dim
        block = blocks[block_type]
        self.stack = nn.Sequential(*[
            block(hidden_dim, expansion_ratio=expansion_ratio, kernel_size=kernel_size, dilation=dilation,
                                dropout=dropout, batch_norm=batch_norm, activation=activation)
            for _ in range(layers)
        ])
        self.norm = ConvLayerNorm(hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, out_dim, kernel_size=1, bias=final_bias) if out_dim else nn.Identity()

    def forward(self, x):
        x = self.inp_proj(x)
        x = self.stack(x)
        import matplotlib.pyplot as plt

        normed = self.norm(x)
        normed_out = self.out_proj(normed)
        plt.figure()
        plt.hist(normed_out.cpu().detach().numpy().flatten(), bins=100)
        out = self.out_proj(x)
        plt.figure()
        plt.hist(out.cpu().detach().numpy().flatten(), bins=100)
        return self.out_proj(x)


class SequenceActiveSiteHead(nn.Module):

    def __init__(self, in_dim, out_dim=1, layers=1, hidden_dim=None, activation='relu', batch_norm=True,
                 dropout=0.1, block_type='Conv1dInvBottleNeck', kernel_size=3, dilation=1):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.stack = Conv1dStack(in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                 activation=activation, batch_norm=batch_norm, layers=layers,
                                 block_type=block_type, kernel_size=kernel_size, dilation=dilation)


    def forward(self, embeds, sigmoid=False):
        x = self.stack(embeds.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        return torch.sigmoid(x) if sigmoid else x


class SequenceInteractionHead(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_layers: int = 3,
            expansion_ratio: int = 4,
            kernel_size: int = 3,
            dropout: float = 0.1,
            output_dim: int | None = None,  # optional: project to lower dim before matmul
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




