import os
import random
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.datasets.amazon_dataset import load_all_datasets_to_df
from src.datasets.splits import make_split, save_split, get_or_make_split

TFIDF_DIR = "data/tfidf"
X_PATH = os.path.join(TFIDF_DIR, "X.npz")
Y_PATH = os.path.join(TFIDF_DIR, "y.npy")
ENCODER_PATH = os.path.join(TFIDF_DIR, "encoder.joblib")


class TFIDFEncoder:
    def __init__(self, stop_words="english", max_features=5000):
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words, max_features=max_features
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    @property
    def vocabulary(self):
        return self.vectorizer.get_feature_names_out()


def encode_dataframe(df):
    encoder = TFIDFEncoder()
    print(f"\n[ENCODE] TF-IDF encoding {len(df)} documents...")
    start = time.perf_counter()
    X = encoder.fit_transform(df["text"])
    elapsed = time.perf_counter() - start
    print(f"[ENCODE] TF-IDF done in {elapsed:.2f}s -> matrix {X.shape}")
    y = df["category"].to_numpy()
    return X, y, encoder


def get_tfidf_features(depth):
    df = load_all_datasets_to_df(depth)
    X, y, encoder = encode_dataframe(df)
    return X, y, encoder


def save_tfidf(X, y, encoder, out_dir=TFIDF_DIR):
    os.makedirs(out_dir, exist_ok=True)
    sparse.save_npz(os.path.join(out_dir, "X.npz"), X)
    np.save(os.path.join(out_dir, "y.npy"), np.asarray(y))
    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    print(f"\n[SAVE] TF-IDF matrix: {X.shape} -> {out_dir}")
    train_idx, test_idx = make_split(y)
    save_split(out_dir, train_idx, test_idx)


def load_tfidf(out_dir=TFIDF_DIR):
    if os.path.exists(os.path.join(out_dir, "X.npz")) and \
       os.path.exists(os.path.join(out_dir, "y.npy")) and \
       os.path.exists(os.path.join(out_dir, "encoder.joblib")):
           
        print(f"\n[LOAD] Found existing TF-IDF features in {out_dir}. Loading...")
        X = sparse.load_npz(os.path.join(out_dir, "X.npz"))
        y = np.load(os.path.join(out_dir, "y.npy"), allow_pickle=True)
        encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
        get_or_make_split(y, out_dir)
        return X, y, encoder
    
    print(f"\n[LOAD] No existing TF-IDF features found in {out_dir}. Computing from scratch...")
    depth = int(out_dir.split("_depth")[-1]) if "_depth" in out_dir else None
    X, y, encoder = get_tfidf_features(depth)
    save_tfidf(X, y, encoder, out_dir)
    return X, y, encoder