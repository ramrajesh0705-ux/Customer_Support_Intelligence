from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder


def build_texts(df):
    return (df['ticket_subject'].fillna('') + ' ' + df['ticket_description'].fillna('')).tolist()


class TicketTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def train_transformer_classifier(train_df, test_df, model_dir: Path, epochs: int = 1):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    labels = sorted(train_df['ticket_type'].unique())
    encoder = LabelEncoder().fit(labels)

    train_texts = build_texts(train_df)
    train_labels = encoder.transform(train_df['ticket_type'])
    test_texts = build_texts(test_df)
    test_labels = encoder.transform(test_df['ticket_type'])

    train_dataset = TicketTextDataset(train_texts, train_labels, tokenizer)
    eval_dataset = TicketTextDataset(test_texts, test_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(encoder.classes_)
    )

    args = TrainingArguments(
        output_dir=str(model_dir / 'transformer_ticket_type'),
        evaluation_strategy='epoch',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        disable_tqdm=True,
    )

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro'),
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    transformer_dir = model_dir / 'transformer_ticket_type'
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)
    joblib_path = transformer_dir / 'ticket_type_labels.joblib'
    import joblib
    joblib.dump(encoder, joblib_path)
