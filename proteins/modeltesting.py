from data.parse import *
from pathlib import Path
from proteins.data.datasets import MultiSequenceDS, SingleSequenceDS, ESMCMultiDS, ESMCSingleDS
from proteins.models.model import SequenceActiveSiteHead
import torch
from proteins.data.utils import pad_collate_fn
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score
import numpy as np
from proteins.training.losses import BinaryFocalLoss

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

model = SequenceActiveSiteHead(dataset.embed_dim, layers=1, kernel_size=5, batch_norm=False,
                               hidden_dim=258,
                               block_type='ConvNeXt1DBlock')
sd = torch.load(Path.cwd() / 'experiments'/'model.pth', map_location=device)
model.load_state_dict(sd['model_state_dict'])
model.to(device)
model.eval()
all_labels, all_logits, all_loss = [], [], []
val_loader = DataLoader(dataset=dataset, batch_size=15, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)
loss_crit = BinaryFocalLoss(reduction='none')
with torch.no_grad():
    for embeds, labels, mask in val_loader:
        embeds, labels, mask = embeds.to(device), labels.to(device), mask.to(device)
        logits = model(embeds)
        loss = loss_crit(logits, labels)
        all_logits.extend(torch.masked_select(logits, mask).cpu().numpy())
        all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())
        all_loss.extend(torch.masked_select(loss, mask).cpu().numpy())
all_labels = np.array(all_labels)
all_logits = np.array(all_logits)
all_loss = np.array(all_loss)
all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
main_score, metrics = compute_ep_metric(all_probs, all_labels)


from plotting import save_scatter_logits_loss, plt
save_scatter_logits_loss(all_logits, all_loss, all_labels, 't/ttest', max_points=500000)


plt.figure()
scatter = plt.scatter(all_logits, all_loss, c=all_labels)
plt.xlabel("Logits")
plt.ylabel("Loss")
plt.title("Logits vs Loss colored by label")
plt.colorbar(scatter, label="Label")
plt.show()

import numpy as np

t4 = np.load('/Users/connorott/Downloads/004.npz')
t5 = np.load('/Users/connorott/Downloads/005.npz')
t4 = {f: t4[f] for f in t4.files}
t5 = {f: t5[f] for f in t5.files}

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score

mcc = np.array([matthews_corrcoef(t4['labels'][i], t4['logits'][i]>0.5) for i in range(len(t4['logits']))])
f1 = np.array([f1_score(t4['labels'][i], t4['logits'][i]>0.5) for i in range(len(t4['logits']))])
prc = np.array([average_precision_score(t4['labels'][i], t4['logits'][i]) for i in range(len(t4['logits']))])

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax=axes[0]
ax.plot(mcc, label='MCC')
ax.plot(f1, label='F1')
ax.plot(prc, label='PRC')
ax.legend()

ax=axes[1]
sns.kdeplot(t4['logits'][0][t4['labels'][0] == 0], fill=True, color='red', label='Non-Binding (Sampled)', ax=ax)
sns.kdeplot(t4['logits'][0][t4['labels'][0] == 1], fill=True, color='green', label='Binding (All)', ax=ax)
ax.set_title(f'Logit Separation (Epoch {len(t4['labels']) - 1})')
ax.set_xlabel('Logits')
ax.legend()