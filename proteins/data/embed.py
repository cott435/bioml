from esm.models.esmc import ESMC
from esm.sdk import client
from os import path, makedirs
from esm.sdk.api import LogitsConfig, LogitsOutput, ESMProtein, ESMProteinError, ProteinType
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence
from esm.utils.encoding import tokenize_sequence
import torch
from .utils import missing_esm_ids


def get_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv
        load_dotenv()
        return getenv('FORGE_TOKEN')
    return token

class ESMCLocal:

     def __init__(self, model_name):
        self.model = ESMC.from_pretrained(model_name)


class ESMCForge:

    def __init__(self, model_name, token=None, save_dir=None, force=False):
        self.model = client(model_name, url="https://forge.evolutionaryscale.ai", token=get_token(token))
        self.save_dir = path.join(save_dir, model_name)
        makedirs(self.save_dir, exist_ok=True)
        self.emb_config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        self.force = force

    def embed_sequence(self, sequence: str) -> LogitsOutput:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(protein_tensor, self.emb_config)
        return output

    def save_sequence(self, id_: str, sequence: str) -> None:
        output = self.embed_sequence(sequence)
        torch.save(output.embeddings[0], path.join(self.save_dir, f"{id_}_embeddings.pt"))
        torch.save(output.hidden_states[:, 0], path.join(self.save_dir, f"{id_}_hidden_states.pt"))

    def batch_save(self, sequences: dict) -> Sequence[LogitsOutput]:
        if not self.force:
            new_ids = missing_esm_ids(sequences, self.save_dir)
            sequences = {id_: seq for id_, seq in sequences.items() if id_ in new_ids}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.save_sequence, id_, seq) for id_, seq in sequences.items()
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(ESMProteinError(500, str(e)))
        return results



