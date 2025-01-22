from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import torch
import os
from transformers import EarlyStoppingCallback


def train_and_save( X_train, y_train, X_val, y_val, model_name, num_labels, max_length, num_epochs):

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        fp16_flag = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        fp16_flag = False 
    else:
        device = torch.device("cpu")
        fp16_flag = False
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir='./model_cache',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir='./logs',
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=fp16_flag,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print("Validation Results:", results)

    save_dir = f"../saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved at {save_dir}")