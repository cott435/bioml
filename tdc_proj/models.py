import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.optim import AdamW
from tqdm.auto import tqdm  # For progress bars

# Assuming your dataset class (from previous) yields (sequence, labels) where:
# - sequence: str or list of AAs, e.g., 'ACDEFG...'
# - labels: list or tensor of 0/1 per AA position

# Vocab for from-scratch model (standard 20 AAs + PAD)
AA_VOCAB = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'PAD': 20}
VOCAB_SIZE = len(AA_VOCAB)

# --------------------- From-Scratch Transformer Model ---------------------
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_layers=2, num_heads=4, hidden_dim=256, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=AA_VOCAB['PAD'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, 1)  # Binary output per token
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len) with AA indices
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        # Transpose for Transformer (seq_len, batch_size, embed_dim)
        embeds = embeds.transpose(0, 1)
        if attention_mask is None:
            attention_mask = (input_ids != AA_VOCAB['PAD']).float()
        # Mask: (batch_size, seq_len) -> expand for attention
        src_key_padding_mask = (1 - attention_mask).bool()  # True where PAD
        encoded = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        encoded = encoded.transpose(0, 1)  # Back to (batch_size, seq_len, embed_dim)
        logits = self.classifier(encoded).squeeze(-1)  # (batch_size, seq_len)
        return logits

# Collate function for from-scratch dataloader
def collate_from_scratch(batch):
    sequences, labels = zip(*batch)
    # Tokenize sequences to indices
    input_ids = [torch.tensor([AA_VOCAB.get(aa, AA_VOCAB['PAD']) for aa in seq]) for seq in sequences]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=AA_VOCAB['PAD'])
    labels = pad_sequence([torch.tensor(lbl) for lbl in labels], batch_first=True, padding_value=-100)  # -100 ignores in loss
    attention_mask = (input_ids != AA_VOCAB['PAD']).float()
    return input_ids, labels, attention_mask

# --------------------- Pretrained ProtBERT Model ---------------------
def get_pretrained_model_and_tokenizer():
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)  # AAs are uppercase
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=1)  # Binary classification
    return model, tokenizer

# Collate function for pretrained dataloader (handles tokenization)
def collate_pretrained(batch, tokenizer, max_length=512):
    sequences, labels = zip(*batch)
    # ProtBERT tokenizes AAs with spaces: 'A C D E' -> tokens
    seq_strs = [' '.join(list(seq)) for seq in sequences]  # e.g., 'A C D E F'
    tokenized = tokenizer(seq_strs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    # Pad labels to match tokenized length (ignoring [CLS]/[SEP])
    padded_labels = []
    for lbl in labels:
        # ProtBERT adds [CLS] and [SEP], so pad labels accordingly (ignore specials with -100)
        lbl_tensor = torch.tensor([-100] + lbl + [-100])  # Add for [CLS] and [SEP]
        padded_labels.append(lbl_tensor)
    labels = pad_sequence(padded_labels, batch_first=True, padding_value=-100)
    return input_ids, labels, attention_mask

# --------------------- Trainer Class (Works for Both) ---------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-4, epochs=5, use_pretrained=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=lr) if use_pretrained else Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss(ignore_index=-100)  # Ignores padded labels

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids, labels, attention_mask = [b.to(self.device) for b in batch]
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.loss_fn(logits, labels.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Collect for metrics (ignore -100)
            mask = labels != -100
            preds = (torch.sigmoid(logits[mask]) > 0.5).cpu().numpy()
            lbls = labels[mask].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(self.train_loader), acc, f1

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids, labels, attention_mask = [b.to(self.device) for b in batch]
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(logits, labels.float())
                total_loss += loss.item()

                mask = labels != -100
                preds = (torch.sigmoid(logits[mask]) > 0.5).cpu().numpy()
                lbls = labels[mask].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(self.val_loader), acc, f1

    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate_epoch()
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")






