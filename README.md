# Review Sentiment Analysis

**Live demo:** https://kowshikreddy099.github.io/sentiment-analysis/
**Source code:** https://github.com/kowshikreddy099/sentiment-analysis

A deep-learning project that classifies English movie, product and service reviews as **Positive** or **Negative**, built with Transformers (DistilBERT / DeBERTa-v3) and an optional BiLSTM model.

---

## Try it now — no install needed

Open the **[Live Demo](https://kowshikreddy099.github.io/sentiment-analysis/)**:

- Type or paste any review and click **Analyze sentiment** (or press **Ctrl + Enter**)
- Click one of the built-in example reviews to test instantly
- You get **Positive / Negative** with confidence bars for both classes
- The page keeps your last 5 predictions

> The demo runs the DistilBERT model entirely inside your browser, so the first visit takes about 10–30 seconds while the model (~65 MB) downloads. After that it is fast.

---

## What this project does

Classifies real-world reviews — not just movie reviews — as positive or negative, trained on a large mixed dataset:

| Source | Domain | Used (default) |
|---|---|---|
| Amazon Polarity | Products | 25,000 |
| Yelp Polarity | Restaurants / services | 25,000 |
| IMDb | Movies | 25,000 |
| **Total** | | **75,000 train + ~11,000 test** |

Raise `--samples_per_domain` (up to 560k Yelp / 3.6M Amazon) for even more data.

### Accuracy you can expect (on the mixed test set)

| Model | Accuracy |
|---|---|
| BiLSTM | ~88–90% |
| DistilBERT | ~92–93% |
| RoBERTa-base | ~94–95% |
| **DeBERTa-v3-base (default)** | **~95–96%** |

(Typical published ranges; your exact result is printed at the end of training.)

### Real-world robustness

- Text cleanup: HTML entities, `&amp;`, repeated letters ("soooo"), repeated "!!!!"
- Contractions expanded so negations survive ("didn't" → "did not")
- Amazon review titles merged into the review text
- Best model picked by F1 score, with learning-rate warmup + weight decay

---

## Project structure

```
sentiment-analysis/
├── app.py                     # Gradio web app (run this locally)
├── requirements.txt           # Python dependencies
├── src/
│   ├── data.py                # Multi-domain data loading + text cleaning
│   ├── lstm_model.py          # BiLSTM architecture
│   ├── train_lstm.py          # Train the BiLSTM
│   ├── train_transformer.py   # Train DistilBERT / RoBERTa / DeBERTa-v3
│   ├── train_bert.py          # Alias for train_transformer.py
│   └── predict.py             # CLI prediction + predictor classes
├── notebooks/
│   └── train_on_colab.ipynb   # One-click Google Colab training notebook
├── models/                    # Trained models go here (not committed)
├── data/                      # Downloaded datasets (not committed)
├── PUBLISHING.md              # GitHub + Hugging Face publishing guide
└── README.md
```

---

## Run locally (VS Code, Windows PowerShell)

1. Open the `sentiment-analysis` folder in VS Code (`File → Open Folder`)
2. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Try predictions from the terminal:

```powershell
python src/predict.py "Not bad at all, works great for the price!"
```

4. Launch the desktop-style web app:

```powershell
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `pip` can't find requirements.txt | You are in the wrong folder — run `cd sentiment-analysis` first |
| `venv\Scripts\activate` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Wrong Python interpreter in VS Code | `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `venv` |
| Out of GPU memory (CUDA OOM) | Lower `--samples_per_domain` or use `--model distilbert-base-uncased` |
| Port 7860 busy | Close the old app window, or run `python app.py` after stopping it with Ctrl+C |

---

## Training

**Recommended: free Google Colab GPU** (a laptop CPU would take a day+)

1. Open https://colab.research.google.com → upload `notebooks/train_on_colab.ipynb`
2. *Runtime → Change runtime type → T4 GPU*
3. Run all cells (75k reviews, ~45–70 minutes) and download `trained_models.zip`
4. Unzip it into this project's `models/` folder

**Or train locally with an NVIDIA GPU:**

```powershell
python src/train_transformer.py --model microsoft/deberta-v3-base --samples_per_domain 25000 --epochs 2 --fp16
python src/train_lstm.py --samples_per_domain 25000 --epochs 5
```

Quick CPU smoke test:

```powershell
python src/train_transformer.py --model distilbert-base-uncased --samples_per_domain 2000 --epochs 1
```

Without a trained model, `app.py` and `predict.py` automatically fall back to a public pre-trained DistilBERT checkpoint — so everything works out of the box.

---

## Publishing

See **[PUBLISHING.md](PUBLISHING.md)** for the full walkthrough:

- **GitHub** — push the code (portfolio + backup; it does not host the Python app)
- **GitHub Pages** — hosts the browser-based live demo used above
- **Hugging Face Spaces** — free hosting for the full Gradio app with your trained model

---

## Tech stack

- **Transformers:** Hugging Face `transformers` + `torch` (DistilBERT / RoBERTa / DeBERTa-v3)
- **BiLSTM:** PyTorch with GloVe-style embeddings, bidirectional LSTM + attention pooling
- **Data:** Hugging Face `datasets` (Amazon Polarity, Yelp Polarity, IMDb)
- **Interface:** Gradio (local) + browser-side Transformers.js (live demo)
