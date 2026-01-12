import torch
import torch.nn as nn

class ESMActiveSite(nn.Module):

    def __init__(self, esm_model, out_dim=1):
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
        self.bos_eos_slice = (slice(None), slice(int(self.esm.prepend_bos), -int(self.esm.append_eos)))

    def forward(self, tokens, sigmoid=False):
        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[self.esm.num_layers])
        embeds = results["representations"][self.esm.num_layers][self.bos_eos_slice]  # [B, L, D]
        """embeds = torch.rand(tokens.shape[0], tokens.shape[1], self.esm.embed_dim)[self.bos_eos_slice]"""
        logits = self.head(embeds).squeeze(-1)  # [B, L]
        return torch.sigmoid(logits) if sigmoid else logits

    def get_active_mask(self, tokens):
        mask = tokens != self.esm.padding_idx
        return mask[self.bos_eos_slice]


