import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))
import torch
from data import get_tdc_epitope, data_to_list, plot_viz
import numpy as np
from model import ESMActiveSite
from training import run_cross_validation
import esm

data = get_tdc_epitope('IEDB_Jespersen', split=False)
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
fasta_path = "tdc_epitope_antigens.fasta"
records = []
for idx, row in data.iterrows():
    seq = row['Sequence']  # Adjust column name if different (e.g., 'X' or 'antigen_sequence')
    antigen_id = row.get('Antigen_ID', f"antigen_{idx}")
    records.append(SeqRecord(Seq(seq), id=antigen_id, description=""))
SeqIO.write(records, fasta_path, "fasta")
identity = 0.4  # 40%; try 0.5 for less aggressive
output_fasta = "tdc_epitope_clustered.fasta"
clstr_file = output_fasta + ".clstr"
import subprocess
subprocess.run([
    "cd-hit", "-i", fasta_path, "-o", output_fasta,
    "-c", str(identity), "-n", "2", "-M", "16000", "-T", "8"
], check=True)

data = data[data['Antigen'].apply(lambda x: len(x) < 5000)]
plot_viz(data['Antigen'], data['Y'])
sequences, labels = data_to_list(data)

esm2, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm2 = esm2.eval()
esm2_batch_converter = esm2_alphabet.get_batch_converter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ESMActiveSite(esm2, 1)
run_cross_validation(model, esm2_batch_converter, sequences, labels, device=device)


