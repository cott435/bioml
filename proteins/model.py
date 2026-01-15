import torch
import torch.nn as nn

class ESMActiveSite(nn.Module):

    def __init__(self, embed_dim, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim/2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(embed_dim/2), out_dim)
        )

    def forward(self, embeds, sigmoid=False):
        logits = self.head(embeds).squeeze(-1)  # [B, L]
        return torch.sigmoid(logits) if sigmoid else logits

    def get_active_mask(self, tokens):
        mask = tokens != self.esm.padding_idx
        return mask[self.bos_eos_slice]


