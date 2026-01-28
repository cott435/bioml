from data.parse import *
from pathlib import Path
from proteins.data.datasets import MultiSequenceDS, SingleSequenceDS, ESMCMultiDS, ESMCSingleDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from proteins.data.utils import pad_collate_fn
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score
import numpy as np

data_name = 'IEDB_Jespersen'
model_name = 'esmc_300m'
base_data_dir = Path.cwd() / 'data' / 'data_files'
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
dataset=ESMCSingleDS(data_name, model_name, save_dir=base_data_dir)

base_data_dir = Path.cwd() / 'data' / 'data_files'


def compute_ep_metric(probs, labels, thresh=0.5):
    preds = (probs > thresh).astype(int)
    auprc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return auprc, {"AUPRC": auprc, "MCC": mcc, "F1": f1}

model = SequenceActiveSiteHead(dataset.embed_dim, layers=3, kernel_size=7, batch_norm=True,
                               hidden_dim=203,
                               block_type='Conv1dInvBottleNeck')
sd = torch.load(Path.cwd() / 'experiments'/'model.pth', map_location=device)
model.load_state_dict(sd['model_state_dict'])
model.to(device)
model.eval()
all_labels, all_probs = [], []
val_loader = DataLoader(dataset=dataset, batch_size=15, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

with torch.no_grad():
    for embeds, labels, mask in val_loader:
        embeds, labels, mask = embeds.to(device), labels.to(device), mask.to(device)
        logits = model(embeds)
        probs = torch.sigmoid(logits)
        all_probs.extend(torch.masked_select(probs, mask).cpu().numpy())
        all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
main_score, metrics = compute_ep_metric(all_probs, all_labels)

