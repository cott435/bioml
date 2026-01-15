from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from data.utils import pad_collate_fn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score


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

    def compute_loss(self, logits, labels, mask):
        loss = self.criterion(logits, labels.to(logits.dtype))
        loss = torch.mean(loss.sum(dim=-1) / mask.sum(dim=-1))
        return loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for embeds, labels, mask in loop:
            embeds, labels = embeds.to(self.device), labels.to(self.device)
            logits = self.model(embeds)
            loss = self.compute_loss(logits, labels, mask)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.log_metrics({'Loss/Training': loss.item(),'GradNorm': grad_norm.item()}, self.total_steps)
            self.total_steps += 1
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        all_labels, all_probs = [], []
        total_val_loss = 0

        with torch.no_grad():
            for embeds, labels, mask in self.val_loader:
                embeds, labels = embeds.to(self.device), labels.to(self.device)
                logits = self.model(embeds)
                loss = self.compute_loss(logits, labels, mask)
                total_val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_probs.extend(torch.masked_select(probs, mask).cpu().numpy())
                all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        metrics = self.compute_ep_metric(all_probs, all_labels)
        metrics['Loss/Validation'] = total_val_loss / len(self.val_loader)

        self.log_metrics(metrics, epoch)
        return metrics

    def log_metrics(self, metrics, step):
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def train(self):
        print(f"\n=== Training (Logs: {self.save_dir}) ===")
        metrics={}
        try:
            for epoch in range(self.epochs):
                epoch+=1
                self.train_epoch(epoch)
                metrics = self.validate(epoch)
                print(f"Epoch {epoch}: ", " | ".join([f'{k}: {v:.4f}' for k,v in metrics.items()]))
                self.log_metrics({'LR': self.scheduler.get_last_lr()[0]}, epoch)
                if metrics['AUPRC'] > self.best_metric:
                    self.best_metric = metrics['AUPRC']
                    self.save_checkpoint("best_model.pth")
                self.scheduler.step()

            if self.save_dir:
                self.load_checkpoint(os.path.join(self.save_dir, "best_model.pth"))
            return metrics
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

    @staticmethod
    def compute_ep_metric(probs, labels, thresh=0.5):
        preds = (probs > thresh).astype(int)
        auprc = average_precision_score(labels, probs)
        mcc = matthews_corrcoef(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        return {"AUPRC": auprc, "MCC": mcc, "F1": f1}


def run_cross_validation(model, dataset, n_splits=5, device='cpu'):

    """Separate function for full CV – this is the clean place for it."""
    skf = GroupKFold(n_splits=n_splits)
    fold_results = []
    data = dataset.data.dropna(subset=['group_id'])
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, groups=data['group_id'])):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        loss_weight = data.iloc[train_idx].apply(lambda row: (len(row['X']) - len(row['Y']))/len(row['Y']), axis=1).mean().item()

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, collate_fn=pad_collate_fn)

        trainer = Trainer(model, train_loader, val_loader, device=device, loss_weight=loss_weight, save_dir=f'./results/fold{fold+1}')
        final_metrics = trainer.train()

        fold_results.append(final_metrics)



