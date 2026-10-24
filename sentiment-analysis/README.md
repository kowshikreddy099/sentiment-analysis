# Real-World Review Sentiment Analysis (DeBERTa / BERT + BiLSTM)

Classifies product, service and movie reviews as **positive** or **negative**.
Trained on a large mixed dataset so it works on real reviews, not just movies:

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
(Typical published ranges; your result is printed at the end of training.)

### Real-world robustness
- Text cleanup: HTML, `&amp;`, repeated letters ("soooo"), repeated "!!!!"
- Contractions expanded so negations survive ("didn't" → "did not")
- Amazon review titles merged into the text
- Model picked by F1 score, with warmup + weight decay to avoid overfitting

## Step 1 — Train on a free GPU (recommended)
Training 75k reviews on a laptop CPU would take a day+. Use Google Colab:
1. Open https://colab.research.google.com → Upload `notebooks/train_on_colab.ipynb`
2. Runtime → Change runtime type → **T4 GPU**
3. Upload this project zip in the notebook (cell 2) and run all cells
4. Download `trained_models.zip` and unzip into this project's `models/` folder

## Step 2 — Run locally
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/predict.py "Not bad at all, works great for the price!"
python app.py                     # open http://127.0.0.1:7860
```
Local GPU training instead of Colab:
```bash
python src/train_transformer.py --model microsoft/deberta-v3-base --samples_per_domain 25000 --epochs 2 --fp16
python src/train_lstm.py --samples_per_domain 25000 --epochs 5
```
Quick CPU test run: `--model distilbert-base-uncased --samples_per_domain 2000 --epochs 1`

## Step 3 — Publish & get a live link (Hugging Face Spaces, free)
1. Sign up at https://huggingface.co → New → **Space** → SDK **Gradio** → name `review-sentiment`
2. Files → Add file → Upload: `app.py`, `requirements.txt`, `src/`, `models/`
   (model >10MB? use Git LFS: `git lfs install && git lfs track "*.safetensors" "*.pt"`)
3. Wait for build → your live link: `https://huggingface.co/spaces/<username>/review-sentiment`

Tip: model too big to upload? Push it to the Hub from Colab
(`trainer.push_to_hub()` / `huggingface-cli upload`) and pass its name to `TransformerPredictor("<username>/<model>")` in `app.py`.

The app works before training too, using a public pre-trained checkpoint.

## Project structure
```
app.py                       Gradio web app
src/data.py                  dataset loading + real-world text cleaning
src/train_transformer.py     DeBERTa/RoBERTa/BERT fine-tuning
src/train_bert.py            alias for train_transformer.py
src/train_lstm.py            BiLSTM training
src/lstm_model.py            BiLSTM architecture
src/predict.py               CLI + predictor classes
notebooks/train_on_colab.ipynb  one-click GPU training
models/                      trained models go here
```
