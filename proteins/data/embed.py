from esm.models.esmc import ESMC
from esm.sdk import client
from esm.sdk.api import LogitsConfig, LogitsOutput, ESMProtein, ESMProteinError, ESMProteinTensor
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor
import torch
from .utils import missing_esm_ids
import numpy as np

"""
Note:
    Local model hidden states outputs L layers worth of hidden_repr
    Forge model hidden states outputs L+1 layers worth of hidden_repr
        Forge appends embeddings (before transformer blocks) to front
        
    For forge model, the final hidden_repr is post-LN while for local the final is pre-LN
"""

model_repr_layers = {
    'esmc_300m': [4, 9, 14, 19, 24, 25, 26, 27, 28],
    'esmc_600m': [4, 9, 14, 19, 24, 29, 30, 31, 32, 33, 34]
}

def get_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv
        load_dotenv()
        return getenv('FORGE_TOKEN')
    return token

class ESMCEmbedder:

    def __init__(self, model_name='esmc_300m', save_dir=None,
                 max_seq_len=2000, seq_overlap=250, device='cpu'):
        self.save_dir = save_dir / model_name if save_dir else None
        if save_dir:
            self.save_dir.mkdir(exist_ok=True)
        self.emb_config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        self.max_seq_len = max_seq_len
        self.seq_overlap = seq_overlap
        self.device = device
        self._set_model(model_name)
        self.repr_layers = model_repr_layers[model_name]

    def _set_model(self, model_name: str):
        self.model = ESMC.from_pretrained(model_name)

    def _check_current_data(self, sequences: dict, force=False) -> dict:
        if not force:
            new_ids = missing_esm_ids(sequences, self.save_dir)
            sequences = {id_: seq for id_, seq in sequences.items() if id_ in new_ids}
        return sequences

    def embed_sequence(self, sequence: str) -> LogitsOutput:
        protein_tensors = self._tensorize(sequence)
        outputs = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in protein_tensors]
        return self._merge_split_embeddings(outputs)

    def save_sequence_embedding(self, id_: str, sequence: str) -> None:
        output = self.embed_sequence(sequence)
        torch.save(output.embeddings[0], self.save_dir / f"{id_}_embeddings.pt")
        hidden_states = output.hidden_states[:, 0]
        selected_hidden_states = {layer: hidden_states[layer] for layer in self.repr_layers}
        torch.save(selected_hidden_states, self.save_dir / f"{id_}_hidden_states.pt")

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
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self.max_seq_len)
            chunk = tokens[start:end]
            if len(chunk) < (min_size + self.seq_overlap):
                seqs[-1] = seqs[-1] + chunk[self.seq_overlap:]
                break
            seqs.append(chunk)
            start += self.max_seq_len - self.seq_overlap
        return [ESMProteinTensor(sequence=seq) for seq in seqs]

    def _merge_split_embeddings(self, embeddings: List[LogitsOutput], trim=True) -> LogitsOutput:
        targs = ['embeddings', 'hidden_states']
        if len(embeddings) == 1:
            if not trim:
                return embeddings[0]
            output = {att: getattr(embeddings[0], att)[..., 1:-1, :] for att in targs}
            return LogitsOutput(**output)
        unique_lengths = [embeddings[0].embeddings.shape[1]]
        unique_lengths.extend([logits_output.embeddings.shape[1] - self.seq_overlap for logits_output in embeddings[1:]])
        indexes = np.array([0] + unique_lengths).cumsum()
        output = {}
        for attr in targs:
            tensors = [getattr(logits_output, attr) for logits_output in embeddings]
            shape = list(tensors[0].shape)
            shape[-2] = indexes[-1]
            merged = torch.zeros(*shape, dtype=tensors[0].dtype)
            counts = torch.zeros_like(merged)
            for i in range(len(unique_lengths)):
                start=max(0, indexes[i] - self.seq_overlap)
                end = indexes[i+1]
                merged[..., start:end, :] += tensors[i]
                counts[..., start:end, :] += 1
            full_emb = merged / counts.clamp(min=1)
            output[attr] = full_emb[..., 1:-1, :] if trim else full_emb
        return LogitsOutput(**output)


class ESMCBatchEmbedder(ESMCEmbedder):

    def __init__(self, model_name='esmc_300m', save_dir=None, max_seq_len=2000, seq_overlap=250, device='cpu'):
        super().__init__(model_name=model_name, save_dir=save_dir, max_seq_len=max_seq_len, seq_overlap=seq_overlap, device=device)

    def _batch_tensorize(self, sequences: dict, max_batch_size) -> _BatchedESMProteinTensor:
        sequences = [self._tensorize(seq) for seq in sequences]
        # TODO: pull sequences from protein tensor and extend IDs
        n_full = len(sequences) // max_batch_size
        extra = len(sequences) % max_batch_size > 0
        for i in range(n_full + (1 if extra else 0)):
            batched = stack_variable_length_tensors(sequences[i * max_batch_size:(i + 1) * max_batch_size],
                                                    constant_value=self.model.tokenizer.pad_token_id)
            protein_tensor = _BatchedESMProteinTensor(sequence=batched)

    def batch_save(self, sequences: dict, batch_size=16) -> Sequence[LogitsOutput]:
        # TODO:
        #  Check what ids still need saved
        #  tokenize all sequences and then split them (must also keep track of sub_ids)
        #  Sort sequences and their IDs
        #     Figure out sort method; if plain sort, first half and second half can be far apart and use a lot of memory
        #     Must use method that quickly finalizes split sequences to free memory (sort sequences <2000 and run)
        #     Sort sequences >2000 and split and run alternate batches (2000 first, then next tensors to finalize whole sequences)
        #  Batch and run through model
        #  merge sequences based on sub IDs
        #  Save data

        sequences = self._check_current_data(sequences)
        sorted_sequences = sorted(sequences.items(), key=lambda item: len(item[1]))
        no_split = [(id_, seq) for id_, seq in sorted_sequences if len(seq) <= self.max_seq_len]
        split = [(id_, seq) for id_, seq in sorted_sequences if len(seq) > self.max_seq_len]

        self._batch_tensorize(sequences, batch_size)


class ESMCForgeEmbedder(ESMCEmbedder):

    def __init__(self, model_name='esmc_300m', token=None, save_dir=None, max_seq_len=2000, seq_overlap=250):
        self.token = get_token(token)
        super().__init__(model_name=model_name, save_dir=save_dir, max_seq_len=max_seq_len, seq_overlap=seq_overlap)
        self.repr_layers = [l + 1 for l in self.repr_layers]

    def _set_model(self, model_name):
        model_name = model_name + '-2024-12' if '-2024-12' not in model_name else model_name
        model_name = model_name.replace('_', '-')
        self.model = client(model_name, url="https://forge.evolutionaryscale.ai", token=self.token)

    def batch_save(self, sequences: dict, force=False) -> Sequence[LogitsOutput]:
        sequences = self._check_current_data(sequences, force=force)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.save_sequence_embedding, id_, seq) for id_, seq in sequences.items()
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(ESMProteinError(500, str(e)))
        return results



