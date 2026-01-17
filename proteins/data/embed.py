from esm.models.esmc import ESMC
from esm.sdk import client
from esm.sdk.api import LogitsConfig, LogitsOutput, ESMProtein, ESMProteinError, ESMProteinTensor
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor
import torch
from .utils import missing_esm_ids


def get_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv
        load_dotenv()
        return getenv('FORGE_TOKEN')
    return token

class ESMCEmbedder:

    def __init__(self, model_name='esmc_300m', save_dir=None, force=False):
        self.save_dir = save_dir / model_name if save_dir else None
        if save_dir:
            self.save_dir.mkdir(exist_ok=True)
        self.emb_config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        self.force = force
        self._set_model(model_name)

    def _set_model(self, model_name):
        self.model = ESMC.from_pretrained(model_name)

    def _tensorize(self, sequence: str) -> ESMProteinTensor:
        protein = ESMProtein(sequence=sequence)
        return self.model.encode(protein)

    def _batch_tensorize(self, sequences: dict) -> _BatchedESMProteinTensor:
        batch_size = 16
        sorted_sequences = sorted(sequences.items(), key=lambda item: len(item[1]))
        ids, sequences = map(list, zip(*sorted_sequences))
        sequences = [self._tensorize(seq).sequence for seq in sequences]
        n_full = len(sequences) // batch_size
        extra = len(sequences) % batch_size > 0
        for i in range(n_full + (1 if extra else 0)):
            batched = stack_variable_length_tensors(sequences[i * batch_size:(i + 1) * batch_size],
                                                    constant_value=self.model.tokenizer.pad_token_id)
            protein_tensor = _BatchedESMProteinTensor(sequence=batched)

    def embed_sequence(self, sequence: str) -> LogitsOutput:
        protein_tensor = self._tensorize(sequence)
        output = self.model.logits(protein_tensor, self.emb_config)
        return output

    def save_sequence_embedding(self, id_: str, sequence: str) -> None:
        output = self.embed_sequence(sequence)
        torch.save(output.embeddings[0], self.save_dir / f"{id_}_embeddings.pt")
        torch.save(output.hidden_states[:, 0], self.save_dir / f"{id_}_hidden_states.pt")

    def batch_save(self, sequences: dict) -> Sequence[LogitsOutput]:
        self._batch_tensorize(sequences)





class ESMCForgeEmbedder(ESMCEmbedder):

    def __init__(self, model_name='esmc_300m', token=None, save_dir=None, force=False):
        self.token = get_token(token)
        super().__init__(model_name=model_name, save_dir=save_dir, force=force)

    def _set_model(self, model_name):
        self.model = client(model_name, url="https://forge.evolutionaryscale.ai", token=self.token)

    def batch_save(self, sequences: dict) -> Sequence[LogitsOutput]:
        if not self.force:
            new_ids = missing_esm_ids(sequences, self.save_dir)
            sequences = {id_: seq for id_, seq in sequences.items() if id_ in new_ids}
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



