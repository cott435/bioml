import torch
import torch.nn as nn


from .blocks import (
    Conv1dInvBottleNeck,
    ConvNeXt1DBlock,
    ConvFFN,
    ResConvFFN,
    PreActResNet1DBlock,
    MaskedLSTM,
    _activation_cls
)
from timm.layers import trunc_normal_
from .norm import MaskedInstanceNorm1d, MaskedBatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

blocks = {
    'Conv1dInvBottleNeck': Conv1dInvBottleNeck,
    'ConvNeXt1DBlock': ConvNeXt1DBlock,
    'ConvFFN': ConvFFN,
    'ResConvFFN': ResConvFFN,
    'PreActResNet1DBlock': PreActResNet1DBlock,
    'lstm':'lstm'
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

        if block_type == 'lstm':
            self.stack = MaskedLSTM(dim, dropout=dropout, layers=layers)
        else:
            block = blocks[block_type] if isinstance(block_type, str) else block_type
            dilations = [2 ** (i % 5) for i in range(layers)]  # restarts after 16
            dp_rate = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
            self.stack = nn.ModuleList([
                block(dim, expansion_ratio=expansion_ratio, kernel_size=kernel_size, dilation=dilations[i],
                      dropout=dropout, activation=activation, drop_path=dp_rate[i])
                for i in range(layers)
            ])
        self.norm = nn.LayerNorm(dim) if final_norm else nn.Identity()
        if block_type == 'lstm':
            self._init_lstm()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            #feats = m.in_features if hasattr(m, 'in_features') else m.in_channels
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_lstm(self):
        for name, param in self.stack.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x, mask=None):
        if isinstance(self.stack, MaskedLSTM):
            x = self.stack(x, mask=mask)
        else:
            for layer in self.stack:
                x = layer(x, mask=mask)
        return self.norm(x)

class InputProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim, in_proj_norm=False, inp_activation=None,
                 inp_intermediate_dim=None, inp_dropout=0.2, **kwargs):
        super().__init__()
        self.inp_proj = nn.Linear(in_dim, inp_intermediate_dim or hidden_dim)
        if inp_intermediate_dim:
            self.inp_proj2 = nn.Linear(inp_intermediate_dim, hidden_dim)
            self.activation = _activation_cls(inp_activation)() if inp_activation else nn.Identity()
            self.dropout_int = nn.Dropout(inp_dropout) if inp_dropout else nn.Identity()
        else:
            self.inp_proj2 = None
        self.inp_norm = nn.LayerNorm(hidden_dim) if in_proj_norm else nn.Identity()
        self.dropout = nn.Dropout(inp_dropout) if inp_dropout else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        x = self.inp_proj(x)
        if self.inp_proj2 is not None:
            x = self.dropout_int(self.activation(x))
            x = self.inp_proj2(x)
        x = self.inp_norm(x)
        if mask is not None:
            x = x * mask
        return self.dropout(x)


class TokenActivationHead(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=None, final_bias=0, dropout=None,
                 feature_dropout=None, token_dropout=None, inp_norm='instance', **kwargs):
        super().__init__()
        assert inp_norm is None or inp_norm in norms

        self.inp_norm = norms[inp_norm](in_dim) if inp_norm is not None else None
        self.feature_dropout = nn.Dropout1d(feature_dropout) if feature_dropout else nn.Identity()
        self.token_dropout = nn.Dropout1d(token_dropout) if token_dropout else nn.Identity()
        self.in_proj = InputProjection(in_dim, hidden_dim, **kwargs) if hidden_dim else nn.Identity()
        hidden_dim = hidden_dim or in_dim

        self.stack = Stack1D(hidden_dim, dropout=dropout, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)
        # just subtract final bias and keep param bias at 0 to avoid weight decay issues
        self.final_bias = final_bias


    def forward(self, embeds, mask=None, sigmoid=False):
        x = self.inp_norm(embeds.transpose(1, 2), mask=mask).transpose(1, 2) if self.inp_norm is not None else embeds
        x = self.token_dropout(x)
        x = self.feature_dropout(x.transpose(1, 2)).transpose(1, 2)

        mask = mask.unsqueeze(-1) if mask is not None else None
        x = self.in_proj(x, mask=mask)
        x = self.stack(x, mask=mask)

        x = self.dropout(x)
        x = self.out_proj(x).squeeze(-1) + self.final_bias
        return torch.sigmoid(x) if sigmoid else x


class TokenActivationLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, layers=2, dropout=0.3, inp_norm='instance',
                 final_bias=0,feature_dropout=None, token_dropout=None, **kwargs):
        super().__init__()

        assert inp_norm is None or inp_norm in norms

        self.inp_norm = norms[inp_norm](in_dim) if inp_norm is not None else None
        self.feature_dropout = nn.Dropout1d(feature_dropout) if feature_dropout else nn.Identity()
        self.token_dropout = nn.Dropout1d(token_dropout) if token_dropout else nn.Identity()

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.stack = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self._init_weights()
        self.final_bias = final_bias

    def _init_weights(self):
        if isinstance(self.in_proj[0], nn.Linear):
            nn.init.kaiming_normal_(self.in_proj[0].weight, mode='fan_in', nonlinearity='linear')
            if self.in_proj[0].bias is not None:
                nn.init.zeros_(self.in_proj[0].bias)
        for name, param in self.stack.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)


    def forward(self, embeds, mask=None):

        x = self.inp_norm(embeds.transpose(1, 2), mask=mask).transpose(1, 2) if self.inp_norm is not None else embeds
        x = self.token_dropout(x)
        x = self.feature_dropout(x.transpose(1, 2)).transpose(1, 2)
        x = self.in_proj(x)  # (B, S, hidden_dim)

        if mask is not None:
            lengths = mask.sum(dim=1).cpu()

            packed_x = pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
        else:
            packed_x = x

        lstm_out, _ = self.stack(packed_x)  # (B, S, hidden_dim)

        if mask is not None:
            lstm_out, _ = pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=lengths.max()
            )
        lstm_out = self.dropout(lstm_out)
        logits = self.out_proj(lstm_out).squeeze(-1) + self.final_bias  # (B, S)

        if mask is not None:
            logits = logits * mask

        return logits


