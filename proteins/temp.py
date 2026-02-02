from esm.models.esmc import ESMC
from proteins.data.embed import ESMCBatchEmbedder
from tqdm.auto import tqdm
from collections import OrderedDict
from pathlib import Path
import torch
from proteins.data.datasets import ESMCSingleDS, SingleSequenceDS

class Embed(ESMCBatchEmbedder):

    def batch_save(self, sequences: dict, max_tok_per_batch=5000, force=False):
        sorted_sequences = sorted(sequences.items(), key=lambda item: len(item[1]))
        batches = self._batch_tensorize(OrderedDict(sorted_sequences), max_tok_per_batch)
        loop = tqdm(batches, desc="Embedding batches")
        for ids, batch in loop:
            embedding_batch = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in batch]
            merged_embeddings = self._merge_split_embeddings(embedding_batch)


data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd()  / 'data'/ 'data_files'

ssd = SingleSequenceDS(data_name, save_dir=base_data_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
el = Embed(model_name, save_dir=base_data_dir / data_name, device=device)
el.batch_save(ssd.unique_sequences)

d=1



