from matplotlib import pyplot as plt
import torch
import numpy as np

def plot_seq_info(sequences, bind_idx, max_seq_len=None):
    seq_len = sequences.apply(lambda x: len(x))
    active_sites = bind_idx.apply(lambda x: len(x))
    if max_seq_len:
        active_sites = active_sites[seq_len < max_seq_len]
        seq_len = seq_len[seq_len < max_seq_len]
    ratio = active_sites / seq_len

    fig, axs = plt.subplots(3, 1)
    axs[0].hist(seq_len, bins=100)
    axs[0].set_title("Hist of Sequence Lengths")
    axs[1].hist(active_sites, bins=100)
    axs[1].set_title("Hist of Active Sites")
    axs[2].hist(ratio, bins=100)
    axs[2].set_title("Hist of Active Site Ratio")
    fig.tight_layout()

    fig, axs = plt.subplots(2, 1)
    axs[0].scatter(seq_len, ratio)
    axs[0].set_ylabel("Active Site Ratio")
    axs[1].scatter(seq_len, active_sites)
    axs[1].set_xlabel("Sequence Length")
    axs[1].set_ylabel("Active Sites")


def hist(x, bin=100):
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32).cpu().detach().numpy()
    else:
        x = np.array(x)
    plt.figure()
    plt.hist(x.flatten(), bins=bin)


def hists(xs, bin=100, name=None, mask=None, tails=False):
    if not isinstance(xs, list):
        xs = [xs]
    size = len(xs) * 4
    fig, axs = plt.subplots(len(xs), 1, figsize=(10, size))

    for i, ax in enumerate(axs.flatten()):
        x = xs[i]
        if mask is not None:
            x = x[mask]
        if isinstance(x, torch.Tensor):
            x = x.to(torch.float32).cpu().detach().numpy()
        else:
            x = np.array(x)
        x = x.flatten()
        ax.hist(x, bins=bin)
        if not tails:
            q1, q3 = np.percentile(x, [1, 99])
            ax.set_xlim(q1, q3)
    fig.tight_layout()
    if name:
        fig.savefig(name)


def save_scatter_logits_loss(
    logits,
    losses,
    labels,
    out_path,
    max_points=200_000,
    point_size=2,
    seed=0,
):


    # Convert to numpy
    x = np.asarray(logits)
    y = np.asarray(losses)
    c = np.asarray(labels)

    if x.ndim > 1:
        x = x[:, 0]

    N = x.shape[0]
    if N > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_points, replace=False)
        x, y, c = x[idx], y[idx], c[idx]

    plt.ioff()  # ensure no interactive redraws

    fig, ax = plt.subplots()
    ax.scatter(
        x,
        y,
        c=c,
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("Logit")
    ax.set_ylabel("Loss")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

