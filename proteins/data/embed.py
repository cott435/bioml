from esm.models.esmc import ESMC
from esm.sdk import client
from os import path
from esm.sdk.api import LogitsConfig, LogitsOutput, ESMProtein, ESMProteinError, ProteinType
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

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

    def __init__(self, model_name, token=None, save_dir=None):
        self.model = client(model_name, url="https://forge.evolutionaryscale.ai", token=get_token(token))
        self.save_dir = path.join(save_dir, model_name)
        self.emb_config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)

    def embed_sequence(self, sequence: str) -> LogitsOutput:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(protein_tensor, self.emb_config)
        return output

    def batch_save(self, sequences: Sequence[ProteinType], ids: Sequence[str]) -> Sequence[LogitsOutput]:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.embed_sequence, self.model, seq) for seq in sequences
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(ESMProteinError(500, str(e)))
        # TODO add save method here to replace return method
        return results

def save_logits_outputs():
    pass
