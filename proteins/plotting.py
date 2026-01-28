from matplotlib import pyplot as plt




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

