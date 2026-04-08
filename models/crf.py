import math
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """
    Linear-chain CRF for batched variable-length sequences.

    Conventions
    -----------
    emissions: (B, T, C)
        Unnormalized emission scores from the upstream model.
    tags: (B, T)
        Integer tag ids in [0, C).
    mask: (B, T) bool
        True where timestep is valid.

    Score of a path y:
        start[y0] + emission[0, y0]
        + sum_t transitions[y_{t-1}, y_t] + emission[t, y_t]
        + end[y_last]

    Notes
    -----
    - transitions[i, j] means score of going FROM tag i TO tag j
    - This module optionally adds a learned/fixed tag_bias to emissions
      before scoring/decoding. This is often useful for class imbalance.
    """

    def __init__(
        self,
        num_tags: int,
        learn_tag_bias: bool = True,
        learn_transitions: bool = True,
        learn_start_end: bool = True,
    ):
        super().__init__()
        self.num_tags = num_tags

        self.tag_bias = nn.Parameter(torch.zeros(num_tags), requires_grad=learn_tag_bias)
        self.transitions = nn.Parameter(
            torch.zeros(num_tags, num_tags), requires_grad=learn_transitions
        )
        self.start_trans = nn.Parameter(
            torch.zeros(num_tags), requires_grad=learn_start_end
        )
        self.end_trans = nn.Parameter(
            torch.zeros(num_tags), requires_grad=learn_start_end
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_tag_bias(
        self,
        bias: Optional[torch.Tensor] = None,
        tag_counts: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        clamp_min: float = 1e-8,
        center: bool = True,
    ):
        """
        Initialize per-tag bias either directly or from counts.

        Parameters
        ----------
        bias:
            Tensor of shape (C,). Direct initialization values.
        tag_counts:
            Tensor of shape (C,) with counts for each tag in the training set.
            If provided, bias <- log(p_k) / temperature.
        temperature:
            Divide log-priors by this. >1 softens, <1 sharpens.
        clamp_min:
            Numerical floor for probabilities.
        center:
            If True, subtract mean so only relative offsets matter.
        """
        if bias is not None and tag_counts is not None:
            raise ValueError("Provide either bias or tag_counts, not both.")

        if bias is not None:
            bias = torch.as_tensor(bias, dtype=self.tag_bias.dtype, device=self.tag_bias.device)
        elif tag_counts is not None:
            counts = torch.as_tensor(
                tag_counts, dtype=self.tag_bias.dtype, device=self.tag_bias.device
            )
            probs = counts / counts.sum()
            probs = probs.clamp_min(clamp_min)
            bias = probs.log() / temperature
        else:
            raise ValueError("Provide bias or tag_counts.")

        if center:
            bias = bias - bias.mean()

        self.tag_bias.copy_(bias)

    @torch.no_grad()
    def init_transitions(
        self,
        transitions: Optional[torch.Tensor] = None,
        start_trans: Optional[torch.Tensor] = None,
        end_trans: Optional[torch.Tensor] = None,
    ):
        """
        Directly initialize transitions/start/end.
        """
        if transitions is not None:
            t = torch.as_tensor(
                transitions, dtype=self.transitions.dtype, device=self.transitions.device
            )
            if t.shape != (self.num_tags, self.num_tags):
                raise ValueError(
                    f"transitions must have shape {(self.num_tags, self.num_tags)}"
                )
            self.transitions.copy_(t)

        if start_trans is not None:
            s = torch.as_tensor(
                start_trans, dtype=self.start_trans.dtype, device=self.start_trans.device
            )
            if s.shape != (self.num_tags,):
                raise ValueError(f"start_trans must have shape {(self.num_tags,)}")
            self.start_trans.copy_(s)

        if end_trans is not None:
            e = torch.as_tensor(
                end_trans, dtype=self.end_trans.dtype, device=self.end_trans.device
            )
            if e.shape != (self.num_tags,):
                raise ValueError(f"end_trans must have shape {(self.num_tags,)}")
            self.end_trans.copy_(e)

    @torch.no_grad()
    def init_binary_block_structure(
        self,
        neg_tag: int = 0,
        pos_tag: int = 1,
        pos_fraction: Optional[float] = None,
        avg_pos_block_len: Optional[float] = None,
        stay_scale: float = 1.0,
        switch_penalty: float = 2.0,
        start_pos_penalty: float = 1.0,
        end_pos_penalty: float = 0.5,
        center_tag_bias: bool = True,
    ):
        """
        Initialize for a binary task with mostly negatives and one short positive block.

        Recommended for:
            - long sequences
            - rare positives
            - contiguous positive region

        Parameters
        ----------
        neg_tag, pos_tag:
            Tag ids for negative and positive.
        pos_fraction:
            Global fraction of positive tokens, e.g. 10/500 = 0.02.
            Used to initialize tag bias.
        avg_pos_block_len:
            Typical positive block length, e.g. 10.
            Used to initialize persistence in the positive state.
        stay_scale:
            Positive reward for staying in same state.
        switch_penalty:
            Penalty magnitude for switching states.
        start_pos_penalty:
            Penalize starting in positive.
        end_pos_penalty:
            Mild penalty for ending in positive.
        center_tag_bias:
            Whether to mean-center tag bias after log-prior init.

        Heuristic behavior
        ------------------
        - negative is common
        - positive starts are somewhat discouraged
        - once in positive, staying positive is rewarded
        - isolated flips are discouraged
        """
        if self.num_tags != 2:
            raise ValueError("init_binary_block_structure is only for num_tags=2.")

        trans = torch.zeros_like(self.transitions)

        # Base structure:
        # 0->0 encouraged
        # 1->1 encouraged, often slightly more if you want block persistence
        # switching discouraged
        trans[neg_tag, neg_tag] = +stay_scale
        trans[pos_tag, pos_tag] = +(stay_scale + 0.5)
        trans[neg_tag, pos_tag] = -switch_penalty
        trans[pos_tag, neg_tag] = -switch_penalty

        self.transitions.copy_(trans)

        start = torch.zeros_like(self.start_trans)
        end = torch.zeros_like(self.end_trans)

        start[pos_tag] = -start_pos_penalty
        end[pos_tag] = -end_pos_penalty

        self.start_trans.copy_(start)
        self.end_trans.copy_(end)

        if pos_fraction is not None:
            pos_fraction = float(pos_fraction)
            pos_fraction = min(max(pos_fraction, 1e-8), 1 - 1e-8)
            probs = torch.tensor(
                [1.0 - pos_fraction, pos_fraction],
                dtype=self.tag_bias.dtype,
                device=self.tag_bias.device,
            )
            bias = probs.log()
            if center_tag_bias:
                bias = bias - bias.mean()
            self.tag_bias.copy_(bias)

        # Optional refinement from expected positive block length:
        # If average block length is L, then once entering positive,
        # probability of staying positive is roughly 1 - 1/L.
        # In logit terms we can strengthen 1->1 over 1->0.
        if avg_pos_block_len is not None and avg_pos_block_len > 1:
            p_stay_pos = 1.0 - (1.0 / float(avg_pos_block_len))
            p_stay_pos = min(max(p_stay_pos, 1e-6), 1 - 1e-6)
            logit_stay = math.log(p_stay_pos / (1 - p_stay_pos))

            # Use this as relative preference of 1->1 over 1->0.
            trans[pos_tag, pos_tag] = +0.5 * logit_stay
            trans[pos_tag, neg_tag] = -0.5 * logit_stay

            # For negative, staying negative should also usually be favored,
            # but not infinitely strongly because a positive block can begin.
            trans[neg_tag, neg_tag] = max(trans[neg_tag, neg_tag].item(), stay_scale)
            trans[neg_tag, pos_tag] = min(trans[neg_tag, pos_tag].item(), -switch_penalty)

            self.transitions.copy_(trans)

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    def _validate(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape (B, T, C)")
        bsz, seq_len, num_tags = emissions.shape
        if num_tags != self.num_tags:
            raise ValueError(
                f"Expected emissions last dim {self.num_tags}, got {num_tags}"
            )
        if mask is None:
            mask = torch.ones(
                bsz, seq_len, dtype=torch.bool, device=emissions.device
            )
        else:
            if mask.shape != (bsz, seq_len):
                raise ValueError(f"mask must have shape {(bsz, seq_len)}")
            mask = mask.to(dtype=torch.bool, device=emissions.device)

        if not mask[:, 0].all():
            raise ValueError("mask[:, 0] must be True for all sequences.")

        return mask

    def add_tag_bias(self, emissions: torch.Tensor) -> torch.Tensor:
        return emissions + self.tag_bias.view(1, 1, -1)

    def sequence_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_tag_bias: bool = True,
    ) -> torch.Tensor:
        """
        Score the provided tag sequence. Returns (B,).
        """
        mask = self._validate(emissions, mask)
        if apply_tag_bias:
            emissions = self.add_tag_bias(emissions)

        bsz, seq_len, _ = emissions.shape
        if tags.shape != (bsz, seq_len):
            raise ValueError(f"tags must have shape {(bsz, seq_len)}")

        score = self.start_trans[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for t in range(1, seq_len):
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]

            trans_score = self.transitions[prev_tags, curr_tags]
            emit_score = emissions[:, t].gather(1, curr_tags.unsqueeze(1)).squeeze(1)

            score = score + (trans_score + emit_score) * mask[:, t]

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_trans[last_tags]
        return score

    def forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_tag_bias: bool = True,
        return_alpha: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute log-partition logZ with masking.

        alpha_t(j) = log total mass of all paths ending at tag j at time t
        """
        mask = self._validate(emissions, mask)
        if apply_tag_bias:
            emissions = self.add_tag_bias(emissions)

        bsz, seq_len, num_tags = emissions.shape

        alpha = self.start_trans.view(1, num_tags) + emissions[:, 0]
        alpha_trace = [alpha] if return_alpha else None

        for t in range(1, seq_len):
            # scores[b, i, j] = alpha[b, i] + trans[i, j]
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            next_alpha = torch.logsumexp(scores, dim=1) + emissions[:, t]

            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)

            if return_alpha:
                alpha_trace.append(alpha)

        logZ = torch.logsumexp(alpha + self.end_trans.view(1, num_tags), dim=1)

        if return_alpha:
            return logZ, torch.stack(alpha_trace, dim=1)
        return logZ, None

    def backward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_tag_bias: bool = True,
        return_beta: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        beta_t(i) = log total suffix mass from tag i at time t onward
        under the convention that beta at last valid position already
        includes end transition.
        """
        mask = self._validate(emissions, mask)
        if apply_tag_bias:
            emissions = self.add_tag_bias(emissions)

        bsz, seq_len, num_tags = emissions.shape

        beta = self.end_trans.view(1, num_tags).expand(bsz, num_tags)
        if not return_beta:
            for t in range(seq_len - 2, -1, -1):
                scores = (
                    self.transitions.unsqueeze(0)
                    + emissions[:, t + 1].unsqueeze(1)
                    + beta.unsqueeze(1)
                )
                next_beta = torch.logsumexp(scores, dim=2)
                beta = torch.where(mask[:, t + 1].unsqueeze(1), next_beta, beta)
            return None

        beta_trace = [None] * seq_len
        beta_trace[-1] = beta

        for t in range(seq_len - 2, -1, -1):
            scores = (
                self.transitions.unsqueeze(0)
                + emissions[:, t + 1].unsqueeze(1)
                + beta.unsqueeze(1)
            )
            next_beta = torch.logsumexp(scores, dim=2)
            beta = torch.where(mask[:, t + 1].unsqueeze(1), next_beta, beta)
            beta_trace[t] = beta

        return torch.stack(beta_trace, dim=1)

    def negative_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        apply_tag_bias: bool = True,
    ) -> torch.Tensor:
        logZ, _ = self.forward_algorithm(
            emissions, mask=mask, apply_tag_bias=apply_tag_bias, return_alpha=False
        )
        gold_score = self.sequence_score(
            emissions, tags, mask=mask, apply_tag_bias=apply_tag_bias
        )
        nll = logZ - gold_score

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        raise ValueError("reduction must be one of: none, sum, mean")

    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_tag_bias: bool = True,
        return_scores: bool = False,
    ) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
        """
        Return best path for each sequence.
        """
        mask = self._validate(emissions, mask)
        if apply_tag_bias:
            emissions = self.add_tag_bias(emissions)

        bsz, seq_len, num_tags = emissions.shape

        delta = self.start_trans.view(1, num_tags) + emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            scores = delta.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_prev_scores, best_prev_tags = scores.max(dim=1)
            next_delta = best_prev_scores + emissions[:, t]

            delta = torch.where(mask[:, t].unsqueeze(1), next_delta, delta)
            backpointers.append(best_prev_tags)

        delta = delta + self.end_trans.view(1, num_tags)
        best_last_scores, best_last_tags = delta.max(dim=1)

        lengths = mask.long().sum(dim=1)
        paths: List[List[int]] = []

        for b in range(bsz):
            L = int(lengths[b].item())
            last_tag = int(best_last_tags[b].item())
            path = [last_tag]

            for t in range(L - 2, -1, -1):
                last_tag = int(backpointers[t][b, last_tag].item())
                path.append(last_tag)

            path.reverse()
            paths.append(path)

        if return_scores:
            return paths, best_last_scores
        return paths, None

    def marginals(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_tag_bias: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            unary_probs: (B, T, C)
            pair_probs:  (B, T-1, C, C)
            alpha:       (B, T, C)
            beta:        (B, T, C)
            logZ:        (B,)
        """
        mask = self._validate(emissions, mask)
        if apply_tag_bias:
            emissions = self.add_tag_bias(emissions)

        logZ, alpha = self.forward_algorithm(
            emissions, mask=mask, apply_tag_bias=False, return_alpha=True
        )
        beta = self.backward_algorithm(
            emissions, mask=mask, apply_tag_bias=False, return_beta=True
        )

        unary_log_probs = alpha + beta - emissions - logZ[:, None, None]
        unary_probs = unary_log_probs.exp()
        unary_probs = unary_probs * mask.unsqueeze(-1)

        bsz, seq_len, num_tags = emissions.shape
        pair_probs = emissions.new_zeros((bsz, max(seq_len - 1, 0), num_tags, num_tags))

        for t in range(seq_len - 1):
            pair_log = (
                alpha[:, t].unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t + 1].unsqueeze(1)
                + beta[:, t + 1].unsqueeze(1)
                - logZ[:, None, None]
            )
            pair_t = pair_log.exp()
            pair_t = pair_t * mask[:, t + 1].view(-1, 1, 1)
            pair_probs[:, t] = pair_t

        return {
            "unary_probs": unary_probs,
            "pair_probs": pair_probs,
            "alpha": alpha,
            "beta": beta,
            "logZ": logZ,
        }


# ----------------------------------------------------------------------
# Recommended helper for your use case
# ----------------------------------------------------------------------

def make_binary_block_crf(
    pos_count: int,
    neg_count: int,
    avg_pos_block_len: float = 10.0,
    stay_scale: float = 1.0,
    switch_penalty: float = 2.0,
    start_pos_penalty: float = 1.5,
    end_pos_penalty: float = 0.5,
    learn_tag_bias: bool = True,
    learn_transitions: bool = True,
    learn_start_end: bool = True,
) -> LinearChainCRF:
    """
    Convenience constructor for your common case:
        class 0 = negative
        class 1 = positive block

    Example:
        crf = make_binary_block_crf(pos_count=10, neg_count=490, avg_pos_block_len=10)
    """
    crf = LinearChainCRF(
        num_tags=2,
        learn_tag_bias=learn_tag_bias,
        learn_transitions=learn_transitions,
        learn_start_end=learn_start_end,
    )

    pos_fraction = pos_count / float(pos_count + neg_count)

    crf.init_binary_block_structure(
        neg_tag=0,
        pos_tag=1,
        pos_fraction=pos_fraction,
        avg_pos_block_len=avg_pos_block_len,
        stay_scale=stay_scale,
        switch_penalty=switch_penalty,
        start_pos_penalty=start_pos_penalty,
        end_pos_penalty=end_pos_penalty,
        center_tag_bias=True,
    )
    return crf


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, C = 3, 12, 2
    lengths = torch.tensor([12, 9, 6])
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

    # Fake model outputs
    emissions = torch.randn(B, T, C)

    # Example labels with a small positive block
    tags = torch.zeros(B, T, dtype=torch.long)
    tags[0, 4:7] = 1
    tags[1, 2:4] = 1
    tags[2, 1:3] = 1

    crf = make_binary_block_crf(
        pos_count=10,
        neg_count=490,
        avg_pos_block_len=10,
        stay_scale=1.0,
        switch_penalty=2.0,
        start_pos_penalty=1.5,
        end_pos_penalty=0.5,
    )

    loss = crf.negative_log_likelihood(emissions, tags, mask=mask)
    paths, scores = crf.viterbi_decode(emissions, mask=mask, return_scores=True)
    marg = crf.marginals(emissions, mask=mask)

    print("loss:", loss.item())
    print("tag_bias:", crf.tag_bias.data)
    print("transitions:\n", crf.transitions.data)
    print("start_trans:", crf.start_trans.data)
    print("end_trans:", crf.end_trans.data)
    print("paths:", paths)
    print("path scores:", scores)
    print("unary_probs shape:", marg["unary_probs"].shape)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ActiveSitePredictor(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=256, lstm_hidden=128, dropout_rate=0.3):
        super().__init__()

        # 1. Feature Extractor (Bottleneck)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 2. Sequence Modeler (BiLSTM)
        # Note: batch_first=True expects (Batch, Seq, Features)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,  # Keep it shallow to prevent overthinking
            bidirectional=True,
            batch_first=True
        )

        # 3. Classifier Head
        # Multiply lstm_hidden by 2 because it is Bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden * 2, 1)
        )

    def forward(self, embeddings, mask):
        """
        embeddings: (B, S, 960)
        mask: (B, S) boolean tensor where True = valid amino acid, False = padding
        """
        # --- 1. Extract Features ---
        # Shape: (B, S, 256)
        x = self.feature_extractor(embeddings)

        # --- 2. Handle Variable Lengths for LSTM ---
        # Calculate actual lengths from the mask
        lengths = mask.sum(dim=1).cpu()

        # Pack the sequence so the LSTM ignores padding
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through BiLSTM
        packed_output, _ = self.bilstm(packed_input)

        # Unpack back to padded tensor
        # lstm_out shape: (B, S, 256)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        # --- 3. Classify ---
        # Shape: (B, S, 1)
        logits = self.classifier(lstm_out)

        # Squeeze to get (B, S)
        logits = logits.squeeze(-1)

        # --- 4. Apply Mask to Logits ---
        # Set padding logits to a massive negative number so they become 0 after sigmoid
        # and don't interfere with your loss calculations
        logits = logits.masked_fill(~mask, -1e9)

        return logits















