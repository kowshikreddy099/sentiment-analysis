"""Predict sentiment from Python or Command Line.
Usage:
  python src/predict.py "The build quality is stellar, completely blew me away!"
  python src/predict.py --model lstm "Defective upon arrival, will return."
"""
import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import clean_real_world_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class LSTMPredictor:
    def __init__(self, path=os.path.join(ROOT, "models/lstm")):
        from data import Vocab
        from lstm_model import LSTMClassifier
        vocab_path = os.path.join(path, "vocab.json")
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(vocab_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model files not found in {path}. Run 'python src/train_lstm.py' first.")
        cfg = json.load(open(vocab_path))
        self.vocab = Vocab()
        self.vocab.itos = cfg["itos"]
        self.vocab.stoi = {w: i for i, w in enumerate(cfg["itos"])}
        self.max_len = cfg["max_len"]
        self.model = LSTMClassifier(len(self.vocab))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def __call__(self, text):
        clean_txt = clean_real_world_text(text)
        x = torch.tensor([self.vocab.encode(clean_txt, self.max_len)])
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        return {
            "positive": round(prob, 4),
            "negative": round(1.0 - prob, 4)
        }

class TransformerPredictor:
    def __init__(self, path=None):
        from transformers import pipeline
        # Check local paths first
        local_transformer = os.path.join(ROOT, "models/transformer")
        local_bert = os.path.join(ROOT, "models/bert")

        if path:
            target = path
        elif os.path.exists(os.path.join(local_transformer, "config.json")):
            target = local_transformer
        elif os.path.exists(os.path.join(local_bert, "config.json")):
            target = local_bert
        else:
            # High-accuracy public model tuned on real-world reviews
            target = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

        print(f"Loaded Transformer pipeline from: {target}")
        self.pipe = pipeline("text-classification", model=target, top_k=None, truncation=True)

    def __call__(self, text):
        clean_txt = clean_real_world_text(text)
        if not clean_txt:
            return {"positive": 0.5, "negative": 0.5}
        raw_outputs = self.pipe(clean_txt)
        entries = raw_outputs[0] if isinstance(raw_outputs[0], list) else raw_outputs

        results = {}
        for item in entries:
            label = item["label"].lower()
            score = round(float(item["score"]), 4)
            # Normalize common label formats
            if label in ["positive", "label_1", "pos", "4 stars", "5 stars"]:
                results["positive"] = score
            elif label in ["negative", "label_0", "neg", "1 star", "2 stars"]:
                results["negative"] = score
            else:
                results[label] = score

        # Ensure both positive and negative keys exist
        if "positive" in results and "negative" not in results:
            results["negative"] = round(1.0 - results["positive"], 4)
        elif "negative" in results and "positive" not in results:
            results["positive"] = round(1.0 - results["negative"], 4)

        return results

# Alias for backward compatibility
BertPredictor = TransformerPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "bert", "lstm"], default="transformer")
    parser.add_argument("text", help="Review text to classify")
    args = parser.parse_args()

    pred = LSTMPredictor() if args.model == "lstm" else TransformerPredictor()
    scores = pred(args.text)
    winner = max(scores, key=scores.get)
    print("\nResult:")
    print(f"  Sentiment:  {winner.upper()} ({scores[winner]:.1%})")
    print(f"  Scores:     {scores}\n")
