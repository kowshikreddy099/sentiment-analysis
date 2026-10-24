"""Data loading and preprocessing utilities for Sentiment Analysis.
Supports IMDB, Amazon Polarity, and Yelp Polarity for multi-domain real-world robustness.
"""
import re
import html
from collections import Counter
from datasets import load_dataset, concatenate_datasets, DatasetDict

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am"
}

def clean_real_world_text(text):
    """Normalize noisy real-world review text while preserving sentiment cues."""
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (&amp;, &lt;, etc.)
    text = html.unescape(text)
    # Remove HTML break tags
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Expand common contractions so negation words like 'not' stay intact
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    # Compress repeated punctuation (e.g. '!!!!' -> '!!', '????' -> '??')
    text = re.sub(r"([!?.]){2,}", r"\1\1", text)
    # Compress repeated letters (e.g. 'sooooo' -> 'soo', 'loooove' -> 'loove')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_imdb(train_size=None, test_size=None, seed=42):
    """Load binary IMDB movie reviews dataset."""
    ds = load_dataset("imdb")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)
    if train_size: train = train.select(range(min(train_size, len(train))))
    if test_size: test = test.select(range(min(test_size, len(test))))
    return train, test

def load_multi_domain(samples_per_domain=25000, test_ratio=0.15, seed=42):
    """
    Load and mix reviews across Amazon (Products), Yelp (Services), and IMDB (Movies).
    This prevents domain-overfitting and makes the model highly effective on real-world reviews.
    Total training pool: up to 100,000+ reviews.
    """
    datasets_to_load = [
        ("imdb", "train", "test", "text", "label"),
        ("amazon_polarity", "train", "test", "content", "label"),
        ("yelp_polarity", "train", "test", "text", "label")
    ]
    
    train_splits = []
    test_splits = []
    
    for name, tr_key, te_key, text_col, label_col in datasets_to_load:
        try:
            print(f"Loading {name}...")
            raw = load_dataset(name)
            tr = raw[tr_key].shuffle(seed=seed)
            te = raw[te_key].shuffle(seed=seed)
            
            # Subsample if requested
            if samples_per_domain:
                tr = tr.select(range(min(samples_per_domain, len(tr))))
                te_count = max(500, int(len(tr) * test_ratio))
                te = te.select(range(min(te_count, len(te))))
            
            # Standardize column names to 'text' and 'label'
            def standardize(example):
                txt = example[text_col]
                # If dataset also has a 'title' column (like Amazon), prepend it
                if "title" in example and example["title"]:
                    txt = f"{example['title']}. {txt}"
                return {"text": clean_real_world_text(txt), "label": int(example[label_col])}
            
            remove_cols = tr.column_names
            tr = tr.map(standardize, remove_columns=remove_cols)
            te = te.map(standardize, remove_columns=remove_cols)
            
            train_splits.append(tr)
            test_splits.append(te)
        except Exception as e:
            print(f"Warning: Failed to load {name} ({e}). Continuing with others.")
            
    if not train_splits:
        raise RuntimeError("No datasets could be loaded.")
        
    combined_train = concatenate_datasets(train_splits).shuffle(seed=seed)
    combined_test = concatenate_datasets(test_splits).shuffle(seed=seed)
    print(f"Loaded multi-domain dataset: {len(combined_train)} train, {len(combined_test)} test samples.")
    return combined_train, combined_test

def tokenize_words(text):
    return clean_real_world_text(text).lower().split()

class Vocab:
    PAD, UNK = 0, 1
    def __init__(self, texts=None, max_size=30000, min_freq=2):
        self.itos = ["<pad>", "<unk>"]
        if texts is not None:
            c = Counter(t for x in texts for t in tokenize_words(x))
            self.itos += [w for w, f in c.most_common(max_size) if f >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        
    def encode(self, text, max_len=256):
        words = tokenize_words(text)
        ids = [self.stoi.get(t, self.UNK) for t in words][:max_len]
        return ids + [self.PAD] * (max_len - len(ids))
        
    def __len__(self):
        return len(self.itos)
