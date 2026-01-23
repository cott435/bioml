import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expansion_ratio=4, kernel_size=3, stride=1,
                 activation='relu', dropout=0.1, batch_norm=True):
        super().__init__()
        hidden_dim = in_dim * expansion_ratio
        activation_fn = nn.ReLU if activation == 'relu' else nn.GELU
        norm_fn = nn.BatchNorm1d if batch_norm else nn.LayerNorm

        self.block = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1, stride=stride),
            norm_fn(hidden_dim),
            activation_fn(),

            nn.Conv1d(hidden_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            norm_fn(out_dim),
            activation_fn(),

            nn.Conv1d(out_dim, out_dim, kernel_size=1, stride=stride),
            norm_fn(out_dim),
            nn.Dropout(dropout)
        )

        self.residual_proj = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):  # (B, in_dim, len)
        residual = self.residual_proj(x)
        x = self.block(x)
        return x + residual  # (B, out_dim, len)


class SequenceActiveSiteHead(nn.Module):

    def __init__(self, embed_dim, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.block = Conv1dBlock(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim, bias=False)

    def forward(self, embeds, sigmoid=False):
        x = self.block(embeds.transpose(1, 2)).transpose(1, 2)
        x = self.proj(x).squeeze(-1)  # [B, L]
        return torch.sigmoid(x) if sigmoid else x

    def get_active_mask(self, tokens):
        mask = tokens != self.esm.padding_idx
        return mask[self.bos_eos_slice]


class SequenceInteractionHead(nn.Module):
    """
    Transforms two sequences with stacked residual blocks (shared weights),
    then computes pairwise interaction matrix via matmul.

    Input embeddings are expected to come from the same ESM model.
    """

    def __init__(
            self,
            embed_dim: int,
            num_layers: int = 3,
            expansion_ratio: int = 4,
            kernel_size: int = 3,
            stride: int = 1,
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
            blocks.append(Conv1dBlock(
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



