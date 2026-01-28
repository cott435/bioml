from esm.models.esmc import ESMC
from esm.sdk import client
from esm.sdk.api import LogitsConfig, LogitsOutput, ESMProtein, ESMProteinError, ESMProteinTensor
from typing import Sequence, List, Tuple
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
import h5py
import multiprocessing as mp

"""
Note:
    Local model hidden states outputs L layers worth of hidden_repr
    Forge model hidden states outputs L+1 layers worth of hidden_repr
        Forge appends embeddings (before transformer blocks) to front
        
    For forge model, the final hidden_repr is post-LN while for local the final is pre-LN
"""

model_repr_layers = {
    'esmc_300m': [9, 19, 28],
    'esmc_600m': [9, 19, 29]
}

def get_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv
        load_dotenv()
        return getenv('FORGE_TOKEN')
    return token

class ESMCEmbedder:
    """
    Embeds protein sequences with ESMC model. Any sequence over max_seq_len is split for embedding
    Merging is down by averaging over the seq_overlap
    """
    def __init__(self,
                 model_name='esmc_300m',
                 save_dir=None,
                 max_seq_len=2000,
                 seq_overlap=250,
                 device: torch.device | str='cpu'):
        self.file_path = save_dir / f'{model_name}_embeddings.h5' if save_dir else None
        if save_dir:
            save_dir.mkdir(exist_ok=True)
        self.emb_config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        self.max_seq_len = max_seq_len
        self.seq_overlap = seq_overlap
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._set_model(model_name)
        self.repr_layers = model_repr_layers[model_name]

    def _set_model(self, model_name: str):
        self.model = ESMC.from_pretrained(model_name).to(self.device)

    def _check_current_ids(self) -> set:
        if self.file_path.exists():
            with h5py.File(self.file_path, 'r') as hdf:
                existing_ids = set(hdf.keys())
            return existing_ids
        return set()

    def _get_new_ids(self, sequences: dict, force=False) -> dict:
        if not force:
            existing_ids = self._check_current_ids()
            sequences = {id_: seq for id_, seq in sequences.items() if id_ not in existing_ids}
        return sequences

    def embed_sequence(self, sequence: str) -> LogitsOutput:
        protein_tensors = self._tensorize(sequence)
        outputs = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in protein_tensors]
        return self._merge_split_embeddings(outputs)

    def _tensorize(self, sequence: str) -> List[ESMProteinTensor]:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        protein_tensors = self._split_tensor_sequences(protein_tensor)
        return protein_tensors

    def _split_tensor_sequences(self, protein_tensor: ESMProteinTensor, min_size=40) -> List[ESMProteinTensor]:
        if len(protein_tensor.sequence) < (self.max_seq_len + min_size):
            return [protein_tensor]
        tokens = protein_tensor.sequence
        seqs = []
        start, end = 0, 0
        while start < len(tokens) and end < len(tokens):
            end = min(len(tokens), start + self.max_seq_len)
            chunk = tokens[start:end]
            if len(chunk) < (min_size + self.seq_overlap):
                seqs[-1] = torch.concat([seqs[-1], chunk[self.seq_overlap:]])
                break
            seqs.append(chunk)
            start += self.max_seq_len - self.seq_overlap
        return [ESMProteinTensor(sequence=seq.to(self.device)) for seq in seqs]

    def _merge_split_embeddings(self, embeddings: List[LogitsOutput]) -> LogitsOutput:
        targs = ['embeddings', 'hidden_states']
        if len(embeddings) == 1:
            return embeddings[0]
        unique_lengths = [embeddings[0].embeddings.shape[1]]
        unique_lengths.extend([logits_output.embeddings.shape[1] - self.seq_overlap for logits_output in embeddings[1:]])
        indexes = np.array([0] + unique_lengths).cumsum()
        output = {}
        for attr in targs:
            tensors = [getattr(logits_output, attr) for logits_output in embeddings]
            shape = list(tensors[0].shape)
            shape[-2] = indexes[-1]
            merged = torch.zeros(*shape, dtype=tensors[0].dtype, device=self.device)
            counts = torch.zeros_like(merged)
            for i in range(len(unique_lengths)):
                start=max(0, indexes[i] - self.seq_overlap)
                end = indexes[i+1]
                merged[..., start:end, :] += tensors[i]
                counts[..., start:end, :] += 1
            output[attr] = merged / counts.clamp(min=1)
        return LogitsOutput(**output)


class ESMCBatchEmbedder(ESMCEmbedder):

    """
    During batch save, sequences are sorted by length for efficiency with masking.
    Sequences are split if they are over max_seq_len.
    Batches are shortened if needed to ensure all sequences in a batch have the same number of splits
    """

    def __init__(self,
                 model_name='esmc_300m',
                 save_dir=None,
                 max_seq_len=2000,
                 seq_overlap=250,
                 device: torch.device | str='cpu'):
        super().__init__(model_name=model_name, save_dir=save_dir, max_seq_len=max_seq_len, seq_overlap=seq_overlap, device=device)

    def _batch_tensorize(self, sequences: dict, max_tok_per_batch) -> List[Tuple[List[str], List[_BatchedESMProteinTensor]]]:
        tensor_sequences = {id_: self._tensorize(seq) for id_, seq in sequences.items()}

        batches, current_keys, current_values = [], [],[]
        current_length = None

        for key, tensor_list in tensor_sequences.items():
            length = len(tensor_list)
            current_length = length if current_length is None else current_length
            tok_in_batch = (length - 1)*self.max_seq_len * (len(current_keys) + 1) + ((len(current_keys) + 1) * len(tensor_list[-1]))
            if length != current_length or tok_in_batch > max_tok_per_batch: # Flush batch
                stacked = [
                    _BatchedESMProteinTensor(
                        sequence=stack_variable_length_tensors(ts, constant_value=self.model.tokenizer.pad_token_id)
                    )
                    for ts in zip(*current_values)
                ]
                batches.append((current_keys, stacked))
                current_keys, current_values = [], []
                current_length = length

            current_keys.append(key)
            current_values.append([t.sequence for t in tensor_list])
        if current_keys:
            stacked = [
                _BatchedESMProteinTensor(
                    sequence=stack_variable_length_tensors(ts, constant_value=self.model.tokenizer.pad_token_id)
                )
                for ts in zip(*current_values)
                ]
            batches.append((current_keys, stacked))

        return batches

    def batch_save(self, sequences: dict, max_tok_per_batch=5000, force=False):
        #sequences = self._get_new_ids(sequences, force=force)
        if not sequences:
            return
        sorted_sequences = sorted(sequences.items(), key=lambda item: len(item[1]))
        batches = self._batch_tensorize(OrderedDict(sorted_sequences), max_tok_per_batch)
        loop = tqdm(batches, desc="Embedding batches")
        for ids, batch in loop:
            embedding_batch = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in batch]
            merged_embeddings = self._merge_split_embeddings(embedding_batch)
        with h5py.File(self.file_path, "w" if force else "a") as hdf:
            for ids, batch in loop:
                embedding_batch = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in batch]
                merged_embeddings = self._merge_split_embeddings(embedding_batch)
                for i, id_ in enumerate(ids):
                    sequence_slice = slice(1, len(sequences[id_])+1)
                    emb = merged_embeddings.embeddings[i, sequence_slice].to(torch.float32).cpu().numpy()
                    #hdf.create_dataset(id_, data=emb, compression="gzip", compression_opts=4)
                    # hidden_states = merged_embeddings.hidden_states[:, i, sequence_slice].to(torch.float32)
                    # selected_hidden_states = {layer: hidden_states[layer] for layer in self.repr_layers}


class ESMCForgeEmbedder(ESMCEmbedder):

    def __init__(self, model_name='esmc_300m', token=None, save_dir=None, max_seq_len=2000, seq_overlap=250):
        self.token = get_token(token)
        super().__init__(model_name=model_name, save_dir=save_dir, max_seq_len=max_seq_len, seq_overlap=seq_overlap)
        self.repr_layers = [l + 1 for l in self.repr_layers]

    def _set_model(self, model_name):
        model_name = model_name + '-2024-12' if '-2024-12' not in model_name else model_name
        model_name = model_name.replace('_', '-')
        self.model = client(model_name, url="https://forge.evolutionaryscale.ai", token=self.token)

    def _api_worker(self, input_queue: mp.Queue, output_queue: mp.Queue):
        while True:
            item = input_queue.get()
            if item is None:
                break
            id_, sequence = item
            output = self.embed_sequence(sequence)  # API call
            emb = output.embeddings[0, 1:-1].to(torch.float32).cpu().numpy()
            # hidden_states = output.hidden_states[:, 0, 1:-1].to(torch.float32).cpu().numpy()
            # selected_hidden_states = {layer: hidden_states[layer] for layer in self.repr_layers}
            output_queue.put((id_, emb))

    def _hdf_writer(self, output_queue: mp.Queue, force=False):
        with h5py.File(self.file_path, 'w' if force else 'a') as hdf:
            while True:
                item = output_queue.get()
                if item is None:
                    break
                id_, emb = item
                hdf.create_dataset(id_, data=emb, compression="gzip", compression_opts=4)

    def batch_save(self, sequences: dict, force=False):
        sequences = self._get_new_ids(sequences, force=force)
        if not sequences:
            return
        num_workers = min(8, len(sequences))
        input_queue = mp.Queue()
        output_queue = mp.Queue(maxsize=10)

        api_workers = [
            mp.Process(target=self._api_worker, args=(input_queue, output_queue))
            for _ in range(num_workers)
        ]
        for worker in api_workers:
            worker.start()

        writer = mp.Process(target=self._hdf_writer, args=(output_queue, force))
        writer.start()

        for id_, seq in sequences.items():
            input_queue.put((id_, seq))
        for _ in api_workers:  # place stop in queue
            input_queue.put(None)

        for worker in api_workers:  # wait for embedders to all finish
            worker.join()

        output_queue.put(None)  # stop and wait for writer
        writer.join()


if __name__ == "__main__":
    from pathlib import Path
    import torch
    from proteins.data.parse import *
    from proteins.data.datasets import ESMCSingleDS, SingleSequenceDS

    data_name = 'IEDB_Jespersen'
    model_name = 'esmc_300m'
    base_data_dir = Path.cwd() / 'data_files'

    data = get_tdc_epitope(data_name, file_dir=base_data_dir)
    ssd = SingleSequenceDS(data_name, df=data, save_dir=base_data_dir)

    #forge_embedder = ESMCForgeEmbedder(model_name, save_dir=base_data_dir / data_name)
    #top20 = ssd.data.iloc[:20]
    #forge_embedder.batch_save(dict(zip(top20['ID'], top20['Sequence'])))

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    el = ESMCBatchEmbedder(model_name, save_dir=base_data_dir / data_name, device=device)
    el.batch_save(ssd.unique_sequences)

    dataset = ESMCSingleDS(data_name, model_name, df=data, save_dir=base_data_dir)




