import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import matplotlib.pyplot as pl
import py3Dmol
import torch
from biotite.database import rcsb
from esm.sdk import client
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

def get_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv
        load_dotenv()
        return getenv('FORGE_TOKEN')
    return token


model = client('esm3-small-2024-08', url="https://forge.evolutionaryscale.ai", token=get_token())

template_gfp = ESMProtein.from_protein_chain(
    ProteinChain.from_rcsb("1qy3", chain_id="A")
)

protein = ESMProtein(sequence=template_gfp.sequence)

result = model.generate(
    protein,
    GenerationConfig(
        track="sasa",           # ← this is the key: gets SASA + other function tokens
        num_steps=1,                # single pass → fast, deterministic-ish
        temperature=0.0,            # 0 = greedy / highest probability
    )
)

# ── Extract SASA (binned)
# result.function is usually a list or tensor of function tokens per residue
# SASA is one of the function sub-tracks (typically 16 bins)
# You need to map bin index → approximate relative or absolute SASA value

sasa_bins = result.function.sasa   # shape ~ [seq_len], values 0–15
# Or depending on exact SDK version: result.function.data, result.function.sasa_bins, etc.

print("Per-residue SASA bin indices (0–15):", sasa_bins.tolist())

# Optional: rough mapping back to relative SASA (0–1 scale)
# Approximate mid-points of 16 bins (from paper/training distribution)
bin_midpoints = [0.0, 0.031, 0.094, 0.156, 0.219, 0.281, 0.344, 0.406,
                 0.469, 0.531, 0.594, 0.656, 0.719, 0.781, 0.844, 0.95]
relative_sasa = [bin_midpoints[b] for b in sasa_bins]

print("Approximate relative SASA (0=buried, ~1=exposed):", relative_sasa)

