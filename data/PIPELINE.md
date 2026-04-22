# Data Processing Pipeline


1. Build/load processed dataframe
2. Generate FASTA
3. Cluster sequences with CD-HIT
4. Embed sequences with ESMC
5. Train from cached embeddings

The default embedding storage is **LMDB**. Optional storage is **H5**.

## Core Components

- `data/pre_embed.py`
  - `BaseSequenceDS`
  - `SingleSequenceDS`
  - `MultiSequenceDS`
- `data/embed.py`
  - `ESMCBatchEmbedder` (local model)
  - `ESMCForgeEmbedder` (Forge API)
- `data/embedding_store.py`
  - `LMDBEmbeddingStore`
  - `H5EmbeddingStore`
- `data/datasets.py`
  - `ESMCSingleDS`
  - `ESMLMDBDataset`
  - `PackedSequenceDataset`
  - `ESMCPairDS`
- `data/pipeline.py`
  - `SequenceProcessingPipeline`

## Lazy Behavior

The pipeline reuses existing files unless forced:

- Existing parquet (`finalized_*_df.parquet`) is reused
- Existing FASTA is reused
- Existing `.clstr` cluster file is reused
- Existing embeddings are skipped by ID unless `force_embed=True`

## Folder Layout

For dataset name `IEDB_Jespersen` and model `esmc_300m`:

- `data/data_files/IEDB_Jespersen/finalized_50_df.parquet`
- `data/data_files/IEDB_Jespersen/sequences.fasta`
- `data/data_files/IEDB_Jespersen/clustered_50_sequences.clstr`
- `data/data_files/IEDB_Jespersen/esmc_300m/esmc_300m_embeddings.lmdb` (default)
  - or `.../esmc_300m_embeddings.h5`

## End-to-End Usage

```python
from data import SequenceProcessingPipeline

pipe = SequenceProcessingPipeline(
    data_name="IEDB_Jespersen",
    model_name="esmc_300m",
    sequence_kind="single",  # "single" or "multi"
    save_dir="data/data_files",
    cluster_coef=0.5,
)

result = pipe.run(
    df=raw_df,                      # optional if parquet already exists
    column_map=None,                # optional rename map
    embedder_kind="local",          # "local" or "forge"
    storage="lmdb",                 # default; optional "h5"
    include_hidden_states=True,     # store hidden states
    hidden_layers=[0, 10, 20],      # None => store all available
    dtype="float16",                # "float16" or "float32"
    device="cuda",                  # for local embedder
    max_tok_per_batch=5000,
    force_preprocess=False,
    force_embed=False,
)

print(result.embedding_file, result.written)
```

## Training Dataset Construction

### Single-sequence token tasks

```python
train_ds = pipe.build_training_dataset(
    storage="lmdb",
    representation="embeddings",   # "embeddings" | "hidden_states" | "concat"
    hidden_layers=[0, 10, 20],     # used for hidden_states/concat
    max_len=5000,
)
```

`ESMCSingleDS.__getitem__` returns:

- `emb`: shape `[seq_len, embed_dim or expanded_dim]`
- `y`: shape `[seq_len]`

### Pair (two-sequence) tasks

```python
pair_pipe = SequenceProcessingPipeline(
    data_name="HuRI",
    model_name="esmc_300m",
    sequence_kind="multi",
    save_dir="data/data_files",
)

pair_ds = pair_pipe.build_training_dataset(
    storage="lmdb",
    representation="concat",
    hidden_layers=[0, 10, 20],
)
```

`ESMCPairDS.__getitem__` returns:

- `emb1`: shape `[len(seq1), dim]`
- `emb2`: shape `[len(seq2), dim]`
- `y`: scalar float target

## Hidden States

When `include_hidden_states=True`, each record can store:

- `embeddings`: `[seq_len, d]`
- `hidden_states`: `[n_layers_selected, seq_len, d]`
- `hidden_layers`: layer indices that were stored

During loading:

- `representation="embeddings"` uses only embeddings
- `representation="hidden_states"` flattens to `[seq_len, n_layers*d]`
- `representation="concat"` concatenates embeddings + flattened hidden states

## Storage Notes

### LMDB (default)

- Fast random access
- Good for large datasets
- Supports multi-array records via packed NPZ payload
- Backward compatible with older LMDB records containing only embeddings

### H5

- Group-per-sequence layout
- Supports embeddings and hidden states
- Optional compression

## Clustering and Splits

- Clustering is produced by CD-HIT using `cluster_coef`
- Single-sequence grouping uses `cluster`
- Multi-sequence grouping stores `cluster1` and `cluster2`
- Pair splitter (`ClusterPairSplitter`) now expects exactly two cluster columns

## Compatibility

Legacy imports still work:

- `ESMCSingleDS`
- `ESMLMDBDataset`
- `PackedSequenceDataset`
- `SingleSequenceDS`
- `MultiSequenceDS`

New exports:

- `ESMCPairDS`
- `SequenceProcessingPipeline`
- `PipelineResult`

## Recommended Defaults

- `storage="lmdb"`
- `dtype="float16"` for disk efficiency
- `representation="embeddings"` for baseline runs
- switch to `"concat"` once hidden states are part of your models

