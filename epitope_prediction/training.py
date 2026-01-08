from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from data import EpitopeDataset, alphabet_batch_convert
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold

class Trainer:
    """Handles training, validation, and logging for one run (single split or fold)."""
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, epochs=20, loss_weight=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weight, reduction='none')
        self.epochs = epochs
        self.best_auc = 0.0
        self.best_state = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for embeds, labels in self.train_loader:
            embeds, labels = embeds.to(self.device), labels.to(self.device)
            preds = self.model(embeds)  # If passing pre-extracted embeds
            loss = self.criterion(preds, labels[:, 1:-1].to(preds.dtype))
            loss_mask = embeds[:, 1:-1] != 1
            loss = loss * loss_mask
            loss = torch.mean(loss.sum(dim=-1) / loss_mask.sum(dim=-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeds, labels in self.val_loader:
                embeds = embeds.to(self.device)
                preds = self.model.head(embeds)
                all_preds.extend(preds.flatten().cpu().numpy())
                all_labels.extend(labels.flatten().cpu().numpy())
        auc = roc_auc_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, (np.array(all_preds) > 0.5).astype(int), average='binary')
        return auc, f1, precision, recall

    def train(self):
        for epoch in range(self.epochs):
            loss = self.train_epoch()
            auc, f1, prec, rec = self.validate()
            print(f"Epoch {epoch+1}: Loss {loss:.4f} | Val AUC {auc:.4f} F1 {f1:.4f}")
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_state = self.model.state_dict()  # Save best checkpoint
            self.scheduler.step()
        self.model.load_state_dict(self.best_state)  # Load best


def run_cross_validation(model, batch_converter, sequences, labels, n_splits=5, device='cpu'):

    def collate_fn(batch):
        esm2_batch_labels, esm2_batch_strs, esm2_batch_tokens = batch_converter([b[0] for b in batch])
        labels = torch.zeros_like(esm2_batch_tokens)
        start = int(batch_converter.alphabet.prepend_bos)
        for i, l in enumerate([b[1] for b in batch]):
            labels[i, start:len(l) + start] = torch.tensor(l)
        return esm2_batch_tokens, labels

    """Separate function for full CV – this is the clean place for it."""
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    stratify_labels = [len(l)/len(s) for s, l in zip(sequences, labels)]
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        train_seq = [sequences[i] for i in train_idx]
        val_seq = [sequences[i] for i in val_idx]
        train_lab = [labels[i] for i in train_idx]
        val_lab = [labels[i] for i in val_idx]
        loss_weight = np.array([(len(l) - sum(l))/sum(l) for l in train_lab]).mean().item()

        train_ds = EpitopeDataset(train_seq, train_lab)
        val_ds = EpitopeDataset(val_seq, val_lab)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)  # Variable len
        val_loader = DataLoader(val_ds, batch_size=64)

        trainer = Trainer(model, train_loader, val_loader, device=device, loss_weight=torch.tensor(loss_weight).to(device))
        trainer.train()

        final_auc, final_f1, _, _ = trainer.validate()
        fold_results.append({"auc": final_auc, "f1": final_f1})
        print(f"Fold {fold+1} Best Val AUC: {final_auc:.4f}")

    avg_auc = sum(r["auc"] for r in fold_results) / n_splits
    print(f"\nCV Average AUC: {avg_auc:.4f}")

