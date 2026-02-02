from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score
from torch.utils.data import DataLoader
from .losses import BinaryFocalLoss
from proteins.plotting import save_scatter_logits_loss
import optuna
from collections import defaultdict


class EPTrainer:
    """Handles training, validation, and logging for one run (single split or fold)."""
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device | str='cpu',
                 lr=1e-4, epochs=20, max_norm=None, weight_decay=0.01, loss_reduction='mean',
                 ckpt_dir=None, log_dir=None, run_name=None, png_dir=None):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.criterion = BinaryFocalLoss(reduction=loss_reduction)
        self._criterion = BinaryFocalLoss(reduction='none')

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

        self.max_norm = max_norm if max_norm else float('inf')
        self.best_metric = -float('inf')
        self.total_steps = 0
        self.ckpt_dir, self.run_name, self.epochs = ckpt_dir, run_name, epochs
        self.png_dir = png_dir
        self.png_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.val_metrics = defaultdict(list)


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", position=1, leave=False)
        for embeds, labels, mask in loop:
            embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
            logits = self.model(embeds)
            loss = self.criterion(logits, labels, mask=mask)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            total_loss += loss.item()
            self.log_metrics({'Loss/Training': loss.item(),'Misc/GradNorm': grad_norm.item()}, self.total_steps)
            self.total_steps += 1
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        all_labels, all_logits, all_losses = [], [], []
        total_val_loss = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validation", position=1, leave=False)
            for embeds, labels, mask in loop:
                embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
                logits = self.model(embeds)
                loss = self.criterion(logits, labels, mask)
                _loss = self._criterion(logits, labels, mask)
                total_val_loss += loss.item()
                all_logits.extend(torch.masked_select(logits, mask).cpu().numpy())
                all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())
                all_losses.extend(torch.masked_select(_loss, mask).cpu().numpy())
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        all_losses = np.array(all_losses)
        all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
        main_score, metrics = self.compute_val_metric(all_probs, all_labels)
        save_scatter_logits_loss(all_logits, all_losses, all_labels, self.png_dir/f'{self.run_name}_epoch{epoch}.png')
        self.val_metrics['labels'].append(all_labels)
        self.val_metrics['logits'].append(all_logits)
        self.val_metrics['loses'].append(all_losses)

        self.log_metrics(metrics, epoch, prefix='ValMetrics')
        self.log_metrics({'Loss/Validation': total_val_loss / len(self.val_loader)}, epoch)
        return main_score

    def log_metrics(self, metrics, step, prefix=None):
        if self.writer:
            for key, value in metrics.items():
                key = f'{prefix}/{key}' if prefix else key
                self.writer.add_scalar(key, value, step)

    def train(self, trial=None):
        try:
            epochs = tqdm(range(self.epochs), desc="Epochs")
            for epoch in epochs:
                epoch+=1
                self.train_epoch(epoch)
                score = self.validate(epoch)
                print(f"Epoch {epoch} score: {score}")
                self.log_metrics({'Misc/LR': self.scheduler.get_last_lr()[0]}, epoch)
                if score > self.best_metric:
                    self.best_metric = score
                    name = f'{self.run_name}_best_model.pth' if self.run_name else 'best_model.pth'
                    self.save_checkpoint(name)
                self.scheduler.step()
                if trial is not None:
                    trial.report(score, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            self.val_metrics = {k: np.stack(v) for k, v in self.val_metrics.items()}
            return self.best_metric
        except optuna.TrialPruned:
            raise optuna.TrialPruned()
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
    def compute_val_metric(probs, labels, thresh=0.5):
        preds = (probs > thresh).astype(int)
        auprc = average_precision_score(labels, probs)
        mcc = matthews_corrcoef(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        return auprc, {"AUPRC": auprc, "MCC": mcc, "F1": f1}

class Trainer:

    def __init__(self, model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device | str='cpu',
                 lr=1e-4, epochs=20, max_norm=None, weight_decay=0.01, loss_reduction='mean',
                 ckpt_dir=None, log_dir=None, run_name=None):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.criterion = None

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

        self.max_norm = max_norm if max_norm else float('inf')
        self.best_metric = -float('inf')
        self.total_steps = 0
        self.ckpt_dir, self.run_name, self.epochs = ckpt_dir, run_name, epochs
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.val_metrics = []

    def run_epoch(self, epoch):
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

    def train(self):
        try:
            for epoch in range(self.epochs):
                epoch+=1
                self.run_epoch(epoch)
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

    def validate(self, epoch):
        self.model.eval()
        all_labels, all_outputs = [], []
        total_val_loss = 0

        with torch.no_grad():
            for embeds, labels, mask in self.val_loader:
                embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
                output = self.model(embeds)
                loss = self.criterion(output, labels, mask)
                total_val_loss += loss.item()
                all_outputs.extend(torch.masked_select(output, mask).cpu().numpy())
                all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        main_score, metrics = self.compute_val_metric(all_outputs, all_labels)
        metrics['Loss/Validation'] = total_val_loss / len(self.val_loader)
        self.log_metrics(metrics, epoch, prefix='ValMetrics')
        return main_score

    def compute_val_metric(self, model_outputs, labels):
        """
        :param model_outputs: output from model (logits)
        :param labels: labels (Y)
        :return: main validation score (float); all metrics (dict)
        """
        raise NotImplementedError('Compute validation metric not implemented')

    def log_metrics(self, metrics, step, prefix=None):
        if self.writer:
            for key, value in metrics.items():
                key = f'{prefix}/{key}' if prefix else key
                self.writer.add_scalar(key, value, step)

    def save_checkpoint(self, filename):
        if not self.ckpt_dir:
            return
        path = self.ckpt_dir / filename
        torch.save(self._base_save(), path)

    def _base_save(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': float(self.best_metric)
        }

    def load_checkpoint(self, filename):
        pass

