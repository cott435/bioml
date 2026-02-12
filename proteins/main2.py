import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiHeadAttention(nn.Module):
    """
    Minimal multi-head attention implementation for clarity.

    Features:
    - Explicit Q/K/V projections
    - Head splitting and recombination
    - Optional causal masking
    - Optional external attention mask

    Input shape:
        x: (batch, seq_len, embed_dim)

    Output shape:
        (batch, seq_len, embed_dim)
    """

    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        # QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        # (B, T, D) -> (B, H, T, Dh)
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # (B, H, T, Dh) -> (B, T, D)
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * Dh)

    def _causal_mask(self, T, t2, device):
        # Upper triangular mask
        mask = torch.triu(torch.ones(T, t2, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x, context=None, attn_mask=None):
        B, T, _ = x.shape

        # Project Q, K, V
        context = x if context is None else context
        t2 = context.shape[1]
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(context))
        V = self._split_heads(self.v_proj(context))

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1)
        scores /= math.sqrt(self.head_dim)

        # Optional causal mask
        if self.causal:
            causal_mask = self._causal_mask(T, t2, x.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Optional external mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        # Weighted sum
        out = attn @ V

        # Merge heads
        out = self._merge_heads(out)

        return self.out_proj(out)

attn = SimpleMultiHeadAttention(256, 8, causal=True)
x = torch.randn(10, 256, 256)
cont = torch.randn(10, 160, 256)
out = attn(x, context=cont)




from pathlib import Path
from data.datasets import ESMCSingleDS, MultiSequenceDS, ESMCMultiDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from training.param_search import OptunaSearch
from training.trainers import EPTrainer
from sklearn.model_selection import GroupKFold
from training.params import ModelParamSpace, TrainerParamSpace


data_name = 'HuRi'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

dataset = ESMCMultiDS(data_name, model_name, save_dir=base_data_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

results_dir = Path.cwd() / 'experiments'
model_param_space = ModelParamSpace()
trainer_param_space = TrainerParamSpace()
op = OptunaSearch(dataset, GroupKFold, SequenceActiveSiteHead, EPTrainer, device=device,
                     base_save_dir=results_dir, study_name='test1',
                     trainer_params=trainer_param_space, model_params=model_param_space, n_splits=10)
op.optimize(20)


from proteins.plotting import hist


