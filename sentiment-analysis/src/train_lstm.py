"""Train a BiLSTM sentiment classifier on multi-domain reviews (Amazon + Yelp + IMDB).
Usage: python src/train_lstm.py --epochs 5 --samples_per_domain 25000
"""
import argparse, json, os, sys
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
sys.path.insert(0, os.path.dirname(__file__))
from data import load_imdb, load_multi_domain, Vocab
from lstm_model import LSTMClassifier

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dataset", choices=["multi", "imdb"], default="multi")
p.add_argument("--samples_per_domain", type=int, default=25000)
p.add_argument("--train_size", type=int, default=None)
p.add_argument("--out", default="models/lstm")
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if a.dataset == "multi":
    train, test = load_multi_domain(a.samples_per_domain)
else:
    train, test = load_imdb(a.train_size, a.train_size and a.train_size // 2)
vocab = Vocab(train["text"])
print("Vocab size:", len(vocab))

def to_tensor(ds):
    X = torch.tensor([vocab.encode(t, a.max_len) for t in ds["text"]])
    y = torch.tensor(ds["label"], dtype=torch.float)
    return TensorDataset(X, y)

tr = DataLoader(to_tensor(train), batch_size=a.batch, shuffle=True)
te = DataLoader(to_tensor(test), batch_size=a.batch)
model = LSTMClassifier(len(vocab)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
loss_fn = nn.BCEWithLogitsLoss()

def evaluate():
    model.eval(); preds, golds = [], []
    with torch.no_grad():
        for X, y in te:
            preds += (torch.sigmoid(model(X.to(device))) > 0.5).long().cpu().tolist()
            golds += y.long().tolist()
    return preds, golds

best = 0
os.makedirs(a.out, exist_ok=True)
for ep in range(1, a.epochs + 1):
    model.train(); total = 0
    for X, y in tr:
        X, y = X.to(device), y.to(device)
        opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item() * len(y)
    preds, golds = evaluate(); acc = accuracy_score(golds, preds)
    print(f"Epoch {ep}: loss={total/len(tr.dataset):.4f} test_acc={acc:.4f}")
    if acc > best:
        best = acc
        torch.save(model.state_dict(), f"{a.out}/model.pt")
        json.dump({"itos": vocab.itos, "max_len": a.max_len}, open(f"{a.out}/vocab.json", "w"))
preds, golds = evaluate()
print(classification_report(golds, preds, target_names=["negative", "positive"]))
print(f"Best accuracy {best:.4f}. Saved to {a.out}/")
