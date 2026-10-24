"""
High-Accuracy Multi-Domain Sentiment Fine-Tuner.
Trained on large-scale datasets (Amazon Products + Yelp Services + IMDB Movies).
Recommended model: 'microsoft/deberta-v3-base' or 'roberta-base'.

Usage:
  # Fast test run (10k reviews, 1 epoch):
  python src/train_transformer.py --model roberta-base --samples_per_domain 3500 --epochs 1

  # High accuracy production training (75k+ reviews, 2 epochs with GPU):
  python src/train_transformer.py --model microsoft/deberta-v3-base --samples_per_domain 25000 --epochs 2 --fp16
"""
import argparse
import os
import sys
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from data import load_multi_domain, load_imdb

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Transformer on sentiment reviews")
    p.add_argument("--model", default="microsoft/deberta-v3-base",
                   help="Hugging Face model checkpoint (e.g. microsoft/deberta-v3-base, roberta-base, distilbert-base-uncased)")
    p.add_argument("--dataset", choices=["multi", "imdb"], default="multi",
                   help="'multi' mixes Amazon, Yelp, and IMDB; 'imdb' uses movies only")
    p.add_argument("--samples_per_domain", type=int, default=25000,
                   help="Number of samples per domain (3 domains * 25k = 75k train samples)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--fp16", action="store_true", help="Use 16-bit mixed precision (GPU only)")
    p.add_argument("--out", default="models/transformer", help="Output directory to save model")
    return p.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "true_negative": int(cm[0][0]),
        "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]),
        "true_positive": int(cm[1][1])
    }

def main():
    args = parse_args()
    print("=" * 60)
    print(f"Fine-Tuning: {args.model}")
    print(f"Dataset:     {args.dataset} (samples per domain: {args.samples_per_domain})")
    print(f"CUDA:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    # 1. Load Data
    if args.dataset == "multi":
        train_ds, test_ds = load_multi_domain(samples_per_domain=args.samples_per_domain)
    else:
        train_ds, test_ds = load_imdb(train_size=args.samples_per_domain)

    # 2. Tokenize with Fast Tokenizer
    print("Loading tokenizer and tokenizing text...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_encoded = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_encoded = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3. Model Architecture
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # 4. Training Arguments
    use_fp16 = args.fp16 and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=test_encoded,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 5. Train & Evaluate
    print("Starting training...")
    trainer.train()

    print("\nEvaluating on unseen test set...")
    results = trainer.evaluate()
    print("=" * 60)
    print(f"Final Test Accuracy:  {results.get('eval_accuracy', 0):.2%}")
    print(f"Final Test F1-Score:  {results.get('eval_f1', 0):.4f}")
    print(f"Final Test Precision: {results.get('eval_precision', 0):.4f}")
    print(f"Final Test Recall:    {results.get('eval_recall', 0):.4f}")
    print("=" * 60)

    # 6. Save model and tokenizer
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Model saved successfully to '{args.out}'.")

if __name__ == "__main__":
    main()
