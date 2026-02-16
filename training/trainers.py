from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score
from torch.utils.data import DataLoader
from .losses import BinaryFocalLoss
import optuna
from collections import defaultdict
from .schedulers import get_lr_scheduler


class EPTrainer:
    """Handles training, validation, and logging for one run (single split or fold)."""
    def __init__(
            self,
            model,
            train_loader: DataLoader,
            val_loader: DataLoader,
            train_eval_loader: DataLoader | None = None,
            device: torch.device | str = 'cpu',
            scheduler_type: str = 'cosine',
            lr=1e-4,
            epochs=20,
            max_norm=None,
            weight_decay=0.01,
            gamma=2,
            alpha=0.5,
            jitter = 0.0,
            ckpt_dir=None,
            log_dir=None,
            data_dir=None
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.train_eval_loader = train_eval_loader or train_loader

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = get_lr_scheduler(self.optimizer, scheduler_type, epochs, len(train_loader), lr)

        self.criterion = BinaryFocalLoss(reduction='none', gamma=gamma, alpha=alpha)

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

        self.max_norm = max_norm if max_norm else float('inf')
        self.best_metric = -float('inf')
        self.total_steps = 0
        self.jitter = jitter
        self.ckpt_dir, self.data_dir, self.epochs = ckpt_dir, data_dir, epochs
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def train(self, trial=None):
        try:
            epochs = tqdm(range(self.epochs), desc="Epochs")
            for epoch in epochs:
                epoch+=1
                self.train_epoch()
                self.evaluate_epoch(self.train_eval_loader, epoch, self.train_metrics, prefix='TrainMetrics')
                score = self.evaluate_epoch(self.val_loader, epoch, self.val_metrics, prefix='ValMetrics')
                if score > self.best_metric:
                    self.best_metric = score
                    self.save_checkpoint('best_model.pth')
                epochs.set_postfix(val_score=score)
                if trial is not None:
                    trial.report(score, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            return self.best_metric
        except optuna.TrialPruned:
            return self.best_metric
        except Exception as e:
            print(f"Training failed: {e}")
            return self.best_metric
        finally:
            if self.data_dir:
                train_metrics = self._stack_metrics(self.train_metrics)
                val_metrics = self._stack_metrics(self.val_metrics)
                np.savez(self.data_dir / 'train_metrics.npz', **train_metrics)
                np.savez(self.data_dir / 'val_metrics.npz', **val_metrics)
            if self.writer:
                self.writer.close()

    def train_epoch(self, accumulate=True):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", position=1, leave=False)
        for batch in loop:
            self.optimizer.zero_grad()
            loss = self._accumulate_grads(batch) if accumulate else self._get_grads(batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.log_metrics({'Loss/Training': loss,'Misc/GradNorm': grad_norm.item(),
                              'Misc/LR': self.scheduler.get_last_lr()[0]}, self.total_steps)
            self.total_steps += 1
            loop.set_postfix(loss=loss)
            total_loss += loss
        return total_loss / len(self.train_loader)

    def _get_grads(self, batch):
        embeds, labels, mask = (t.to(self.device) for t in batch)
        noise = torch.randn_like(embeds) * self.jitter
        embeds = embeds + noise
        logits_full = self.model(embeds, mask=mask)
        loss = self.criterion(logits_full, labels, mask=mask)
        loss = loss.sum(-1) / labels.sum(-1)
        loss = torch.mean(loss)
        loss.backward()
        return loss.item()

    def _accumulate_grads(self, batch):
        accumulated_loss = 0
        accumulation_steps = sum([len(s[0]) for s in batch])
        for embeds, labels, mask in batch:
            embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
            noise = torch.randn_like(embeds) * self.jitter
            embeds = embeds + noise
            logits = self.model(embeds, mask=mask)
            loss = self.criterion(logits, labels, mask=mask)
            loss = loss.sum(-1) / labels.sum(-1)
            loss = loss.sum() / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
        return accumulated_loss

    def evaluate_epoch(self, loader, epoch, metrics_store, prefix):
        labels, logits, losses, batch_losses = self.collect_outputs(
            self.model, loader, self.criterion, self.device
        )
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        main_score, metrics = self.compute_val_metric(probs, labels)

        avg_loss = float(np.mean(batch_losses)) if len(batch_losses) else float('nan')
        metrics['Loss'] = avg_loss
        metrics_store['labels'].append(labels)
        metrics_store['logits'].append(logits)
        metrics_store['losses'].append(losses)
        metrics_store['avg_loss'].append(avg_loss)
        metrics_store['score'].append(main_score)

        self.log_metrics(metrics, epoch, prefix=prefix)
        return main_score

    @staticmethod
    def collect_outputs(model, loader, criterion, device):
        model.eval()
        all_labels, all_logits, all_losses, batch_losses = [], [], [], []

        def iter_batch(batch):
            if isinstance(batch[0], torch.Tensor):
                embeds, labels, mask = (b.to(device) for b in batch)
                logits = model(embeds, mask=mask)
                loss = criterion(logits, labels, mask)
                normed_loss = loss.sum(-1) / labels.sum(-1).clamp(min=1)
                all_logits.extend(torch.masked_select(logits, mask).detach().cpu().numpy())
                all_labels.extend(torch.masked_select(labels, mask).detach().cpu().numpy())
                all_losses.extend(torch.masked_select(loss, mask).detach().cpu().numpy())
                batch_losses.extend(normed_loss.detach().cpu().numpy())
            else:
                for sub_batch in batch:
                    iter_batch(sub_batch)

        with torch.no_grad():
            for batch in loader:
                iter_batch(batch)

        return (
            np.array(all_labels),
            np.array(all_logits),
            np.array(all_losses),
            np.array(batch_losses),
        )

    @staticmethod
    def _stack_metrics(metrics):
        stacked = {}
        for key, values in metrics.items():
            if not values:
                stacked[key] = np.array([])
            else:
                stacked[key] = np.stack(values)
        return stacked

    def log_metrics(self, metrics, step, prefix=None):
        if self.writer:
            for key, value in metrics.items():
                key = f'{prefix}/{key}' if prefix else key
                self.writer.add_scalar(key, value, step)

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

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", position=1, leave=False)
        for embeds, labels, mask in loop:
            embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
            logits = self.model(embeds, mask=mask)
            loss = self.criterion(logits, labels, mask=mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            self.log_metrics({'Loss/Training': loss.item(),'Misc/GradNorm': grad_norm.item()}, self.total_steps)
            self.total_steps += 1
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def test_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", position=1, leave=False)

        for embeds, labels, mask in loop:
            embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
            logits = self.model(embeds, mask=mask)
            loss = self.criterion(logits, labels, mask=mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            self.log_metrics({'Loss/Training': loss.item(),'Misc/GradNorm': grad_norm.item()}, self.total_steps)
            self.total_steps += 1
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def train(self):
        try:
            for epoch in range(self.epochs):
                epoch+=1
                self.train_epoch()
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

    def train_epoch_(self):
        self.model.eval()
        total_loss = 0
        from torch.nn.utils.rnn import pad_sequence
        loop = tqdm(self.train_loader, desc="Training", position=1, leave=False)
        sub_batches = next(iter(loop))

        self.optimizer.zero_grad()

        batch = [pad_sequence([z for t in tt for z in t], batch_first=True) for tt in zip(*sub_batches)]
        embeds, labels, mask = batch
        embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
        logits_full = self.model(embeds, mask=mask)
        loss_full = self.criterion(logits_full, labels, mask=mask)
        loss_normed = loss_full.sum(-1) / labels.sum(-1)
        loss_full_mean = torch.mean(loss_normed)
        loss_full_mean.backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }


        self.optimizer.zero_grad()
        accumulated_loss = 0
        accumulated_steps = sum([len(s[0]) for s in sub_batches])
        sub_losses = []
        for embeds, labels, mask in sub_batches:
            embeds, labels, mask = embeds.to(self.device), labels.to(self.device), mask.to(self.device)
            logits = self.model(embeds, mask=mask)
            # test_logits = self.model(torch.concat([embeds, torch.zeros(1, 100, 960, device=self.device, dtype=embeds.dtype)], dim=1), mask=torch.concat([mask, torch.zeros(1, 100, device=self.device, dtype=mask.dtype)], dim=1))
            loss_ = self.criterion(logits, labels, mask=mask)
            loss_normed_ = loss_.sum(-1) / labels.sum(-1)
            sub_losses.append(loss_)
            loss = loss_normed_.sum() / accumulated_steps
            loss.backward()
            accumulated_loss += loss.item()
        bucket_grads = {
            name: p.grad.detach().clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        sub_losses = pad_sequence([s for ss in sub_losses for s in ss], batch_first=True)
        res_grads = {name: torch.abs(bucket_grads[name] - grads[name]).sum() for name in grads}

        d=1
