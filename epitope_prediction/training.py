from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from data import EpitopeDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, average_precision_score


class Trainer:
    """Handles training, validation, and logging for one run (single split or fold)."""
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, epochs=20, max_norm=None,
                 save_dir=None, loss_weight=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        loss_weight = torch.tensor(loss_weight).to(device) if loss_weight is not None else None
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weight, reduction='none')
        self.epochs = epochs
        self.best_metric = -float('inf')
        self.writer = SummaryWriter(log_dir=save_dir) if save_dir else None
        self.total_steps = 0
        self.save_dir = save_dir
        self.max_norm = max_norm if max_norm else float('inf')

    def compute_loss(self, embeds, labels):
        bos_eos_slice = (slice(None), slice(int(self.model.esm.prepend_bos), int(self.model.esm.append_eos)))
        preds = self.model(embeds)
        loss = self.criterion(preds, labels[bos_eos_slice].to(preds.dtype))
        loss_mask = embeds[bos_eos_slice] != 1
        loss = loss * loss_mask
        loss = torch.mean(loss.sum(dim=-1) / loss_mask.sum(dim=-1))
        return loss, preds

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        for embeds, labels in loop:
            embeds, labels = embeds.to(self.device), labels.to(self.device)
            loss, pred = self.compute_loss(embeds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()

            if self.writer:
                self.writer.add_scalar('Loss', loss.item(), self.total_steps)
                self.writer.add_scalar('GradNorm', grad_norm.item(), self.total_steps)
            self.total_steps += 1
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_val_loss = 0

        with torch.no_grad():
            for embeds, labels in self.val_loader:
                embeds, labels = embeds.to(self.device), labels.to(self.device)
                loss, preds = self.compute_loss(embeds, labels)
                total_val_loss += loss.item()

                active_indices = (embeds[:, 1:-1] != 1).bool()
                probs = torch.sigmoid(preds)

                valid_probs = torch.masked_select(probs, active_indices).cpu().numpy()
                valid_labels = torch.masked_select(labels[:, 1:-1].to(preds.dtype), active_indices).cpu().numpy()

                all_probs.extend(valid_probs)
                all_labels.extend(valid_labels)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)

        auprc = average_precision_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        avg_val_loss = total_val_loss / len(self.val_loader)

        # Log Metrics
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            self.writer.add_scalar('Val/AUPRC', auprc, epoch)
            self.writer.add_scalar('Val/MCC', mcc, epoch)
            self.writer.add_scalar('Val/F1', f1, epoch)

        return {
            "loss": avg_val_loss,
            "auprc": auprc,
            "mcc": mcc,
            "f1": f1
        }

    def validate_(self):
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
        print(f"\n=== Training (Logs: {self.save_dir}) ===")
        try:
            for epoch in range(self.epochs):
                self.train_epoch(epoch)
                metrics = self.validate(epoch)
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + 1}: Loss {metrics['loss']:.4f} | AUC {metrics['auc']:.4f} | MCC {metrics['mcc']:.4f} | AUPRC {metrics['auprc']:.4f}")
                if metrics['auprc'] > self.best_metric:
                    self.best_metric = metrics['auprc']
                    self.save_checkpoint("best_model.pth")
                self.scheduler.step()

            if self.save_dir:
                self.load_checkpoint(os.path.join(self.save_dir, "best_model.pth"))
        finally:
            if self.writer:
                self.writer.close()

    def save_checkpoint(self, filename="checkpoint.pth"):
        if not self.save_dir:
            return
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)


def run_cross_validation(model, tokenizer, data, n_splits=5, device='cpu'):

    """Separate function for full CV – this is the clean place for it."""
    skf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, groups=data['cluster_id'])):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        loss_weight = train_data.apply(lambda row: (len(row['Antigen']) - len(row['Y']))/len(row['Y']), axis=1).mean().item()

        train_ds = EpitopeDataset(train_data, x_col='Antigen', y_col='Y')
        val_ds = EpitopeDataset(val_data, x_col='Antigen', y_col='Y')

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=tokenizer.collate_batch)  # Variable len
        val_loader = DataLoader(val_ds, batch_size=64, collate_fn=tokenizer.collate_batch)

        trainer = Trainer(model, train_loader, val_loader, device=device, loss_weight=loss_weight)
        trainer.train()

        final_auc, final_f1, _, _ = trainer.validate()
        fold_results.append({"auc": final_auc, "f1": final_f1})
        print(f"Fold {fold+1} Best Val AUC: {final_auc:.4f}")

    avg_auc = sum(r["auc"] for r in fold_results) / n_splits
    print(f"\nCV Average AUC: {avg_auc:.4f}")

