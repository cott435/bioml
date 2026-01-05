from utils import DataProcessor
from models import Trainer, get_pretrained_model_and_tokenizer, SimpleTransformerClassifier
from data_parse import get_tdc_epitope

data = get_tdc_epitope('IEDB_Jespersen')
dp = DataProcessor(data, batch_size=32)
dataloaders = dp.get_dataloaders()
df = data['train']
df['aa_len'] = df['Antigen'].apply(lambda x: len(x))
df['y_len'] = df['Y'].apply(lambda x: len(x))
df['active_ratio'] = df['y_len'] / df['aa_len']
global_weight = (1 - df['active_ratio'].mean()) / df['active_ratio'].mean()

model_scratch = SimpleTransformerClassifier()
trainer_scratch = Trainer(model_scratch, dataloaders['train'], dataloaders['valid'], use_pretrained=False, loss_weight=global_weight)
trainer_scratch.train()

train_loader_scratch = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_from_scratch)
val_loader_scratch = DataLoader(val_dataset, batch_size=16, collate_fn=collate_from_scratch)


# For Pretrained:
model_pretrained, tokenizer = get_pretrained_model_and_tokenizer()
collate_fn_pretrained = lambda batch: collate_pretrained(batch, tokenizer)
train_loader_pretrained = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_pretrained)  # Smaller batch for memory
val_loader_pretrained = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn_pretrained)

trainer_pretrained = Trainer(model_pretrained, train_loader_pretrained, val_loader_pretrained, use_pretrained=True, lr=5e-5)  # Lower LR for fine-tuning
trainer_pretrained.train()
