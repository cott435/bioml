from data.embed import get_token
token = get_token()
from esm.sdk import client

model = client(
    model="esmc-300m-2024-12", url="https://forge.evolutionaryscale.ai", token=token
)

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)


def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output


def batch_embed(
    model: ESM3InferenceClient, inputs: Sequence[ProteinType]
) -> Sequence[LogitsOutput]:
    """Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(embed_sequence, model, protein) for protein in inputs
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

adk_path = Path.cwd().parent /"adk.csv"
df = pd.read_csv(adk_path)
df = df[["org_name", "sequence", "lid_type", "temperature"]]
df = df[df["lid_type"] != "other"]  # drop one structural class for simplicity

outputs = batch_embed(model, df["sequence"].tolist())


import torch

# we'll summarize the embeddings using their mean across the sequence dimension
# which allows us to compare embeddings for sequences of different lengths
all_mean_embeddings = [
    torch.mean(output.hidden_states.float(), dim=-2).squeeze() for output in outputs
]

# now we have a list of tensors of [num_layers, hidden_size]
print("embedding shape [num_layers, hidden_size]:", all_mean_embeddings[0].shape)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

N_KMEANS_CLUSTERS = 3
def plot_embeddings_at_layer(all_mean_embeddings: torch.Tensor, layer_idx: int):
    stacked_mean_embeddings = (
        torch.stack([embedding[layer_idx, :] for embedding in all_mean_embeddings])
        .float()
        .numpy()
    )

    # project all the embeddings to 2D using PCA
    pca = PCA(n_components=2)
    pca.fit(stacked_mean_embeddings)
    projected_mean_embeddings = pca.transform(stacked_mean_embeddings)

    # compute kmeans purity as a measure of how good the clustering is
    kmeans = KMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=0).fit(
        projected_mean_embeddings
    )
    rand_index = adjusted_rand_score(df["lid_type"], kmeans.labels_)

    # plot the clusters
    plt.figure(figsize=(4, 4))
    sns.scatterplot(
        x=projected_mean_embeddings[:, 0],
        y=projected_mean_embeddings[:, 1],
        hue=df["lid_type"],
    )
    plt.title(
        f"PCA of mean embeddings at layer {layer_idx}.\nRand index: {rand_index:.2f}"
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()
plot_embeddings_at_layer(all_mean_embeddings, layer_idx=30)
plot_embeddings_at_layer(all_mean_embeddings, layer_idx=12)

from proteins.data.parse import get_tdc_epitope
data = get_tdc_epitope('IEDB_Jespersen')

outputs = batch_embed(model, data["X"].tolist()[:4])

sequence=data["Sequence"].tolist()[0]
protein = ESMProtein(sequence=sequence)
protein_tensor = model.encode(protein)
output = model.logits(protein_tensor, EMBEDDING_CONFIG)

em = output.embeddings[0]
hs = output.hidden_states[-1,0]


t = outputs[0].hidden_states.to(torch.float32).numpy()

for i in [1,5,10,15,20,25,30]:
    plt.figure()
    plt.hist(t[i].flatten(), bins=50)
    plt.title(i)

