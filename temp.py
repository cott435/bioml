
"""
Linear-Chain CRF for 1D Sequence Labeling (from scratch)

This implements a standard linear-chain CRF layer that sits on top of
any backbone (e.g. ConvNext1D) that produces per-position emission scores.

Key concepts:
  - Emissions: scores from your backbone, shape (batch, seq_len, num_tags)
  - Transitions: learned pairwise scores between adjacent tags
  - The CRF loss = log Z(x) - score(x, y*)   (partition fn minus gold score)
  - Decoding uses Viterbi to find the best global tag sequence

The magic: instead of classifying each position independently, the CRF
finds the label sequence that maximizes the JOINT score across the whole
sequence, naturally producing smooth contiguous blocks.
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearChainCRF(nn.Module):
    """
    Linear-chain CRF layer.

    Place this on top of your backbone:
        emissions = backbone(x)          # (batch, seq_len, num_tags)
        loss = crf.loss(emissions, tags)  # scalar
        preds = crf.decode(emissions)     # (batch, seq_len)

    Parameters
    ----------
    num_tags : int
        Number of label classes (2 for binary: negative/positive).
    pad_tag_id : int or None
        If your sequences are padded, pass the tag id used for padding
        so those positions are masked out of the loss.
    """

    def __init__(self, num_tags: int, pad_tag_id: Optional[int] = None):
        super().__init__()
        self.num_tags = num_tags
        self.pad_tag_id = pad_tag_id

        # ---------------------------------------------------------------
        # TRANSITION MATRIX: transitions[i, j] = score for tag_i -> tag_j
        #
        # For your problem (continuous positive blocks), after training
        # you'd expect something like:
        #   transitions = [[high, low ],    # neg->neg is easy, neg->pos is hard
        #                  [low,  high]]    # pos->neg is hard, pos->pos is easy
        #
        # This is what enforces smoothness — the model pays a penalty
        # for every transition between different states.
        # ---------------------------------------------------------------
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # START and END transition scores:
        # These model the probability of starting/ending a sequence
        # with each tag. Learned just like transitions.
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize transitions uniformly in [-0.1, 0.1].
        Small init lets the emissions dominate early in training,
        then transitions refine the structure as training progresses.
        """
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    # ===================================================================
    #  LOSS: Negative log-likelihood = log_partition - gold_score
    # ===================================================================

    def loss(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Compute CRF negative log-likelihood loss.

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
            Raw scores from backbone (NOT softmaxed).
        tags : (batch, seq_len)
            Ground-truth label indices.
        mask : (batch, seq_len) bool tensor, optional
            True for real tokens, False for padding.
            If None, auto-derived from pad_tag_id or assumed all valid.

        Returns
        -------
        loss : scalar tensor
            Mean negative log-likelihood across the batch.

        The math:
            P(y|x) = exp(score(x,y)) / Z(x)
            -log P(y|x) = log Z(x) - score(x,y)

            score(x,y) = sum of emission scores at gold tags
                        + sum of transition scores between adjacent gold tags
                        + start/end transition scores

            Z(x) = sum over ALL possible tag sequences of exp(score(x,y'))
                  = computed efficiently via the forward algorithm
        """
        if mask is None:
            if self.pad_tag_id is not None:
                mask = tags != self.pad_tag_id
            else:
                mask = torch.ones_like(tags, dtype=torch.bool)

        # Transpose to (seq_len, batch, ...) — standard CRF convention
        # makes the dynamic programming loop cleaner
        emissions = emissions.transpose(0, 1)  # (seq_len, batch, num_tags)
        tags = tags.transpose(0, 1)            # (seq_len, batch)
        mask = mask.transpose(0, 1)            # (seq_len, batch)

        gold_score = self._compute_gold_score(emissions, tags, mask)
        log_partition = self._compute_log_partition(emissions, mask)

        # NLL per sequence, then mean across batch
        nll = log_partition - gold_score  # (batch,)
        return nll.mean()

    # ===================================================================
    #  GOLD SCORE: score of the ground-truth tag sequence
    # ===================================================================

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,   # (seq_len, batch, num_tags)
        tags: torch.LongTensor,     # (seq_len, batch)
        mask: torch.BoolTensor,     # (seq_len, batch)
    ) -> torch.Tensor:
        """
        Score of the gold (ground-truth) label sequence.

        score = start_transition[y_0]
              + sum_t( emission[t, y_t] + transition[y_{t-1}, y_t] )
              + end_transition[y_T]

        This is the numerator of P(y|x) — how compatible the gold
        sequence is with the emissions and learned transitions.
        """
        seq_len, batch_size = tags.shape

        # Score for starting with the first gold tag
        # start_transitions[y_0] for each sequence in the batch
        score = self.start_transitions[tags[0]]  # (batch,)

        # Add emission score at position 0 for the gold tag
        # gather picks out emission[0, b, tags[0, b]] for each b
        score += emissions[0].gather(1, tags[0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            # ---------------------------------------------------------
            # For each position t, add:
            #   1) Transition score: transition[y_{t-1}, y_t]
            #   2) Emission score:   emission[t, y_t]
            #
            # Only for positions where mask is True (not padding).
            # ---------------------------------------------------------

            # Transition: look up transitions[prev_tag, current_tag]
            transition_score = self.transitions[tags[t - 1], tags[t]]

            # Emission: look up emissions[t, current_tag]
            emission_score = emissions[t].gather(1, tags[t].unsqueeze(1)).squeeze(1)

            # Mask: only add scores for valid (non-padded) positions
            score += (transition_score + emission_score) * mask[t].float()

        # End transition: score for ending on the last real tag
        # Find the index of the last valid position for each sequence
        # mask.sum(0) gives the length; -1 for 0-indexed
        lengths = mask.sum(dim=0).long()  # (batch,)
        last_tags = tags.gather(0, (lengths - 1).unsqueeze(0)).squeeze(0)  # (batch,)
        score += self.end_transitions[last_tags]

        return score  # (batch,)

    # ===================================================================
    #  LOG PARTITION (Forward Algorithm): log Z(x)
    # ===================================================================

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,   # (seq_len, batch, num_tags)
        mask: torch.BoolTensor,     # (seq_len, batch)
    ) -> torch.Tensor:
        """
        Compute log Z(x) using the forward algorithm.

        This is the key dynamic programming step. We compute:
            alpha[t, j] = log( sum over all sequences y_{0:t} ending in tag j
                               of exp(score(y_{0:t})) )

        Using the recursion:
            alpha[t, j] = logsumexp_i( alpha[t-1, i] + transition[i,j] ) + emission[t, j]

        At the end:
            log Z = logsumexp_j( alpha[T, j] + end_transition[j] )

        logsumexp is used throughout for numerical stability — we're
        working in log-space to avoid overflow from summing exponentials.

        WHY THIS IS TRACTABLE:
        Naively summing over all possible tag sequences is O(num_tags^seq_len).
        The forward algorithm exploits the Markov property (each tag only
        depends on the previous one) to reduce this to O(seq_len * num_tags^2).
        With num_tags=2, this is just O(4 * seq_len) — trivial.
        """
        seq_len, batch_size, num_tags = emissions.shape

        # Initialize: alpha[0, j] = start_transition[j] + emission[0, j]
        # Shape: (batch, num_tags)
        alpha = self.start_transitions.unsqueeze(0) + emissions[0]

        for t in range(1, seq_len):
            # ---------------------------------------------------------
            # The core DP step, visualized for num_tags=2:
            #
            # alpha[t-1] = [a0, a1]  (previous forward scores)
            #
            # We want alpha[t, j] = logsumexp_i(alpha[t-1, i] + trans[i,j]) + emit[t,j]
            #
            # Expand alpha to (batch, num_tags, 1):
            #   [[a0],    <- scores for "came from tag 0"
            #    [a1]]    <- scores for "came from tag 1"
            #
            # Add transitions (num_tags, num_tags):
            #   [[a0 + t[0,0], a0 + t[0,1]],
            #    [a1 + t[1,0], a1 + t[1,1]]]
            #
            # logsumexp over dim=1 (over "came from" dimension):
            #   [logsumexp(a0+t[0,0], a1+t[1,0]),   <- new alpha for tag 0
            #    logsumexp(a0+t[0,1], a1+t[1,1])]   <- new alpha for tag 1
            # ---------------------------------------------------------

            # (batch, num_tags, 1) + (num_tags, num_tags) → (batch, num_tags, num_tags)
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)

            # logsumexp over the "previous tag" dimension
            # → (batch, num_tags)
            new_alpha = torch.logsumexp(scores, dim=1) + emissions[t]

            # For padded positions, keep old alpha (don't update)
            mask_t = mask[t].unsqueeze(1).float()  # (batch, 1)
            alpha = new_alpha * mask_t + alpha * (1.0 - mask_t)

        # Final: add end transitions, then logsumexp over tags
        # log Z = logsumexp_j( alpha[T, j] + end_transition[j] )
        alpha = alpha + self.end_transitions.unsqueeze(0)
        log_partition = torch.logsumexp(alpha, dim=1)  # (batch,)

        return log_partition

    # ===================================================================
    #  VITERBI DECODING: find the best tag sequence
    # ===================================================================

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.LongTensor:
        """
        Find the most likely tag sequence using Viterbi algorithm.

        This is identical to the forward algorithm, but replaces
        logsumexp with max — instead of summing over all paths,
        we keep only the best path.

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
        mask : (batch, seq_len) bool, optional

        Returns
        -------
        best_tags : (batch, seq_len) LongTensor
            The globally optimal tag sequence for each example.
            This is what gives you smooth, contiguous blocks!
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        emissions = emissions.transpose(0, 1)  # (seq_len, batch, num_tags)
        mask = mask.transpose(0, 1)

        seq_len, batch_size, num_tags = emissions.shape

        # viterbi[t, j] = score of best path ending in tag j at position t
        viterbi = self.start_transitions.unsqueeze(0) + emissions[0]  # (batch, num_tags)

        # Backpointers: which previous tag led to the best score at each step
        # This is how we reconstruct the path after the forward pass
        backpointers = []

        for t in range(1, seq_len):
            # (batch, num_tags, 1) + (num_tags, num_tags) → (batch, num_tags, num_tags)
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)

            # MAX instead of logsumexp — keep only the best incoming tag
            best_scores, best_prev_tags = scores.max(dim=1)  # both (batch, num_tags)

            new_viterbi = best_scores + emissions[t]

            # Mask handling
            mask_t = mask[t].unsqueeze(1).float()
            viterbi = new_viterbi * mask_t + viterbi * (1.0 - mask_t)

            backpointers.append(best_prev_tags)

        # Add end transitions
        viterbi += self.end_transitions.unsqueeze(0)

        # Best final tag
        _, best_last_tags = viterbi.max(dim=1)  # (batch,)

        # --------------------------------------------------
        # Backtrack through the pointers to reconstruct path
        # This is like unwinding a linked list from end to start
        # --------------------------------------------------
        lengths = mask.sum(dim=0).long()
        best_path = torch.zeros(seq_len, batch_size, dtype=torch.long, device=emissions.device)

        best_path[0] = best_last_tags  # temporary; we'll shift after reversal

        # Walk backward through backpointers
        for t in range(len(backpointers) - 1, -1, -1):
            # For each sequence, look up: "given the best tag at t+1,
            # what was the best tag at t?"
            # Only update if position t+1 is within the sequence length
            in_bounds = (t + 1) < lengths
            best_last_tags = torch.where(
                in_bounds,
                backpointers[t].gather(1, best_last_tags.unsqueeze(1)).squeeze(1),
                best_last_tags,
            )
            best_path[t] = best_last_tags

        # Fix: assign the actual best final tags at each sequence's last position
        # We need to re-derive since we overwrote during backtracking
        viterbi_final = self.start_transitions.unsqueeze(0) + emissions[0]
        bp_list = []
        for t in range(1, seq_len):
            scores = viterbi_final.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_s, best_p = scores.max(dim=1)
            viterbi_final = best_s + emissions[t]
            mask_t = mask[t].unsqueeze(1).float()
            viterbi_final = viterbi_final * mask_t + viterbi_final * (1.0 - mask_t)
            bp_list.append(best_p)

        viterbi_final += self.end_transitions.unsqueeze(0)
        _, best_last = viterbi_final.max(dim=1)

        # Proper backtracking
        result = torch.zeros(seq_len, batch_size, dtype=torch.long, device=emissions.device)
        for b in range(batch_size):
            L = lengths[b].item()
            result[L - 1, b] = best_last[b]
            for t in range(L - 2, -1, -1):
                result[t, b] = bp_list[t][b, result[t + 1, b]]

        return result.transpose(0, 1)  # (batch, seq_len)

    # ===================================================================
    #  UTILITIES
    # ===================================================================

    def transition_matrix(self) -> torch.Tensor:
        """
        Return the transition matrix as probabilities (for inspection).
        Applies softmax row-wise so each row sums to 1.

        After training, you'd expect for your problem:
            neg->neg: ~0.97   neg->pos: ~0.03
            pos->neg: ~low    pos->pos: ~high
        """
        return torch.softmax(self.transitions, dim=1)

    def __repr__(self):
        return (
            f"LinearChainCRF(num_tags={self.num_tags})\n"
            f"  Transition scores (raw):\n{self.transitions.data}\n"
            f"  Start scores: {self.start_transitions.data}\n"
            f"  End scores:   {self.end_transitions.data}"
        )


# =======================================================================
#  USAGE EXAMPLE
# =======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 100
    num_tags = 2  # 0 = negative, 1 = positive

    # Simulate a backbone producing emission scores
    emissions = torch.randn(batch_size, seq_len, num_tags)

    # Simulate ground-truth tags with contiguous positive blocks (~3% rate)
    tags = torch.zeros(batch_size, seq_len, dtype=torch.long)
    # Insert a small block of positives in each sequence
    for b in range(batch_size):
        start = torch.randint(20, 80, (1,)).item()
        length = torch.randint(2, 5, (1,)).item()  # block of 2-4 positives
        tags[b, start : start + length] = 1

    print(f"Positive rate: {tags.float().mean():.1%}")
    print(f"Example tags[0]: ...{tags[0, 18:35].tolist()}...")

    # Create CRF and compute loss
    crf = LinearChainCRF(num_tags=num_tags)
    print(f"\nInitial CRF:\n{crf}\n")

    loss = crf.loss(emissions, tags)
    print(f"Initial loss: {loss.item():.4f}")

    # Decode (before any training — will be random)
    preds = crf.decode(emissions)
    print(f"Predictions[0]: ...{preds[0, 18:35].tolist()}...")
    print(f"Gold tags[0]:   ...{tags[0, 18:35].tolist()}...")

    # Quick training loop to show it converges
    optimizer = torch.optim.Adam(crf.parameters(), lr=0.05)
    print("\nTraining for 200 steps...")
    for step in range(2000):
        loss = crf.loss(emissions, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            preds = crf.decode(emissions)
            acc = (preds == tags).float().mean()
            print(f"  Step {step+1:3d} | loss={loss.item():.4f} | acc={acc:.4f}")

    print(f"\nFinal CRF:\n{crf}")
    print(f"\nTransition probabilities:\n{crf.transition_matrix()}")

    preds = crf.decode(emissions)
    print(f"\nFinal predictions[0]: ...{preds[0, 18:35].tolist()}...")
    print(f"Gold tags[0]:         ...{tags[0, 18:35].tolist()}...")


from testing.analyze import ProteinEmbeddingAnalyzer
from pathlib import Path
from data import ESMCSingleDS
from testing.plotting import ESMPlotter

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'

dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

smoothed_plotter = ESMPlotter(dataset, 3000, smoothing_window=5)
plotter = ESMPlotter(dataset, 3000)

plotter.plot(1)
smoothed_plotter.plot(1)




















