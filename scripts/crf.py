import torch
from pathlib import Path
from data import ESMCSingleDS
from data.utils import pad_collate_fn
from models.blocks import LinearChainCRF


data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd().parents[0]  / 'data'/ 'data_files'
dataset = ESMCSingleDS(data_name, model_name, save_dir=base_data_dir, max_len=3000)

idxs = [0,1,2,3,4,5,6]
"""
x, y, mask = pad_collate_fn([dataset[i] for i in idxs])


proj = torch.nn.Linear(960, 2)
torch.nn.init.normal_(proj.weight, std=0.01)
crf = LinearChainCRF(2)

x = proj(x)
loss = crf.get_loss(x, y.to(torch.long), mask)

pred = crf.viterbi_decode(x, mask)"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def logsumexp(x, dim):
    return torch.logsumexp(x, dim=dim)

class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_trans = nn.Parameter(torch.zeros(num_tags))
        self.end_trans = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    def forward_algorithm(self, emissions):
        B, T, C = emissions.shape
        alpha = self.start_trans + emissions[:, 0]

        for t in range(1, T):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha = logsumexp(scores, dim=1) + emissions[:, t]

        return logsumexp(alpha + self.end_trans, dim=1)

    def backward_algorithm(self, emissions):
        B, T, C = emissions.shape
        beta = self.end_trans.unsqueeze(0)

        betas = [None] * T
        betas[-1] = beta

        for t in range(T - 2, -1, -1):
            scores = (
                self.transitions.unsqueeze(0)
                + emissions[:, t + 1].unsqueeze(1)
                + beta.unsqueeze(1)
            )
            beta = logsumexp(scores, dim=2)
            betas[t] = beta

        return torch.stack(betas, dim=1)

    def marginals(self, emissions):
        logZ = self.forward_algorithm(emissions)
        beta = self.backward_algorithm(emissions)

        B, T, C = emissions.shape
        alpha = self.start_trans + emissions[:, 0]
        alphas = [alpha]

        for t in range(1, T):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha = logsumexp(scores, dim=1) + emissions[:, t]
            alphas.append(alpha)

        alpha = torch.stack(alphas, dim=1)

        unary = torch.exp(alpha + beta - emissions - logZ[:, None, None])
        return unary

    def viterbi_decode(self, emissions):
        B, T, C = emissions.shape

        delta = self.start_trans + emissions[:, 0]
        backpointers = []

        for t in range(1, T):
            scores = delta.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(dim=1)
            delta = best_scores + emissions[:, t]
            backpointers.append(best_tags)

        delta = delta + self.end_trans
        best_last = delta.argmax(dim=1)

        paths = []
        for b in range(B):
            path = [best_last[b].item()]
            for t in reversed(range(T - 1)):
                path.append(backpointers[t][b, path[-1]].item())
            paths.append(list(reversed(path)))

        return paths

def make_sequence(T=50, block_size=10):
    y = torch.zeros(T, dtype=torch.long)
    start = np.random.randint(5, T - block_size - 5)
    y[start:start+block_size] = 1
    return y

def make_emissions(y, strength=3.0, noise=1.0):
    T = len(y)
    emissions = torch.randn(T, 2) * noise

    for t in range(T):
        emissions[t, y[t]] += strength
        emissions[t, 1 - y[t]] -= strength

    return emissions.unsqueeze(0)


# ----------------------------
# Plotting
# ----------------------------

def plot_case(title, y, emissions, marginals, viterbi):
    y = y.numpy()
    emissions = emissions[0].detach().numpy()
    marginals = marginals[0].detach().numpy()
    viterbi = np.array(viterbi)

    T = len(y)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(y, label="True Y")
    axs[0].set_title("True Sequence")

    axs[1].plot(emissions[:, 1], label="Emission logit (tag=1)")
    axs[1].plot(emissions[:, 0], label="Emission logit (tag=0)")
    axs[1].legend()
    axs[1].set_title("Emissions")

    axs[2].plot(marginals[:, 1], label="p(tag=1)")
    axs[2].set_ylim(0, 1)
    axs[2].set_title("Marginals")

    axs[3].plot(viterbi, label="Viterbi", linestyle="--")
    axs[3].set_title("Decoded Sequence")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Experiments
# ----------------------------

def run_experiment(emission_strength, transition_strength):
    T = 60
    y = make_sequence(T)

    emissions = make_emissions(y, strength=emission_strength)

    crf = LinearChainCRF(2)

    # transitions: prefer staying in same state if strong
    with torch.no_grad():
        crf.transitions[:] = torch.tensor([
            [ transition_strength, -transition_strength],
            [-transition_strength,  transition_strength],
        ])

    marginals = crf.marginals(emissions)
    viterbi = crf.viterbi_decode(emissions)[0]

    return y, emissions, marginals, viterbi


# ----------------------------
# Run scenarios
# ----------------------------

if __name__ == "__main__":
    cases = [
        ("Good emissions, weak transitions", 3.0, 0.5),
        ("Bad emissions, strong transitions", 0.5, 3.0),
        ("Good emissions, strong transitions", 3.0, 3.0),
        ("Bad emissions, weak transitions", 0.5, 0.5),
    ]

    for title, e_strength, t_strength in cases:
        y, emissions, marginals, viterbi = run_experiment(e_strength, t_strength)
        plot_case(title, y, emissions, marginals, viterbi)
