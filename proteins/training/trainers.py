from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score
from torch.utils.data import DataLoader
from .losses import BinaryFocalLoss


class Trainer:
    """Handles training, validation, and logging for one run (single split or fold)."""
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device | str='cpu',
                 lr=1e-4, epochs=20, max_norm=None, weight_decay=0.01,
                 ckpt_dir=None, log_dir=None, run_name=None):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.criterion = BinaryFocalLoss()

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

        self.max_norm = max_norm if max_norm else float('inf')
        self.best_metric = -float('inf')
        self.total_steps = 0
        self.ckpt_dir, self.run_name, self.epochs = ckpt_dir, run_name, epochs
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for embeds, labels, mask in loop:
            embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
            logits = self.model(embeds)
            loss = self.criterion(logits, labels, mask=mask)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
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
                embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
                logits = self.model(embeds)
                loss = self.criterion(logits, labels, mask)
                total_val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_probs.extend(torch.masked_select(probs, mask).cpu().numpy())
                all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        main_score, metrics = self.compute_ep_metric(all_probs, all_labels)
        metrics['Loss/Validation'] = total_val_loss / len(self.val_loader)

        self.log_metrics(metrics, epoch)
        return main_score

    def log_metrics(self, metrics, step):
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def train(self):
        try:
            for epoch in range(self.epochs):
                epoch+=1
                self.train_epoch(epoch)
                score = self.validate(epoch)
                print(f"Epoch {epoch} score: {score}")
                self.log_metrics({'LR': self.scheduler.get_last_lr()[0]}, epoch)
                if score > self.best_metric:
                    self.best_metric = score
                    name = f'{self.run_name}_best_model.pth' if self.run_name else 'best_model.pth'
                    self.save_checkpoint(name)
                self.scheduler.step()
            return self.best_metric
        finally:
            if self.writer:
                self.writer.close()

    def save_checkpoint(self, filename="checkpoint.pth"):
        if not self.ckpt_dir:
            return
        path = self.ckpt_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': float(self.best_metric)
        }, path)

    def from_checkpoint(self, path):
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
        return auprc, {"AUPRC": auprc, "MCC": mcc, "F1": f1}



