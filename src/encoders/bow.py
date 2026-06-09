import os
import joblib
import numpy as np
from scipy import sparse
from src.datasets.amazon_dataset import load_all_datasets_to_df
from src.datasets.splits import make_split, save_split, get_or_make_split

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import random

BOW_DIR = "data/bow"
X_PATH = os.path.join(BOW_DIR, "X.npz")
Y_PATH = os.path.join(BOW_DIR, "y.npy")
ENCODER_PATH = os.path.join(BOW_DIR, "encoder.joblib")

class BOWEncoder:
    def __init__(self, stop_words="english", max_features=5000):
        self.vectorizer = CountVectorizer(
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
    encoder = BOWEncoder()
    X = encoder.fit_transform(df["text"])
    y = df["category"].to_numpy()
    return X, y, encoder


def get_bow_features():
    df = load_all_datasets_to_df()
    X, y, encoder = encode_dataframe(df)
    save_bow(X, y, encoder)
    return X, y, encoder


def save_bow(X, y, encoder, out_dir=BOW_DIR):
    os.makedirs(out_dir, exist_ok=True)
    sparse.save_npz(os.path.join(out_dir, "X.npz"), X)
    np.save(os.path.join(out_dir, "y.npy"), np.asarray(y))
    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    print(f"\n[SAVE] BOW matrix: {X.shape} -> {out_dir}")
    train_idx, test_idx = make_split(y)
    save_split(out_dir, train_idx, test_idx)


def load_bow(out_dir=BOW_DIR):
    if os.path.exists(os.path.join(out_dir, "X.npz")) and \
       os.path.exists(os.path.join(out_dir, "y.npy")) and \
       os.path.exists(os.path.join(out_dir, "encoder.joblib")):
           
        print(f"\n[LOAD] Found existing BOW features in {out_dir}. Loading...")
        X = sparse.load_npz(os.path.join(out_dir, "X.npz"))
        y = np.load(os.path.join(out_dir, "y.npy"), allow_pickle=True)
        encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
        get_or_make_split(y, out_dir)
        return X, y, encoder
    
    print(f"\n[LOAD] No existing BOW features found in {out_dir}. Computing from scratch...")
    X, y, encoder = get_bow_features()
    save_bow(X, y, encoder, out_dir)
    return X, y, encoder
