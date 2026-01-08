import torch
import torch.nn as nn

class ESMActiveSite(nn.Module):

    def __init__(self, esm_model, out_dim):
        super().__init__()
        self.esm = esm_model
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False
        self.out_dim = out_dim
        self.head = nn.Sequential(
            nn.Linear(esm_model.embed_dim, int(esm_model.embed_dim/2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(esm_model.embed_dim/2), out_dim)
        )

    def forward(self, tokens, sigmoid=False):
        results = self.esm(tokens, repr_layers=[self.esm.num_layers])
        embeds = results["representations"][self.esm.num_layers][:, 1:-1, :]  # [B, L, D]
        logits = self.head(embeds).squeeze(-1)  # [B, L]
        return torch.sigmoid(logits) if sigmoid else logits


