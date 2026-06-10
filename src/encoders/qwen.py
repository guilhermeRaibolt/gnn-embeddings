import os

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.datasets.amazon_dataset import load_all_datasets_to_df
from src.datasets.splits import make_split, save_split, get_or_make_split

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DIR = "data/qwen"
X_PATH = os.path.join(QWEN_DIR, "X.npy")
Y_PATH = os.path.join(QWEN_DIR, "y.npy")
ENCODER_PATH = os.path.join(QWEN_DIR, "encoder.joblib")


class QwenEncoder:
    def __init__(
        self,
        model_name=DEFAULT_QWEN_MODEL,
        batch_size=32,
        normalize=True,
        device=None,
        max_seq_length=512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16} if self.device == "cuda" else None,
        )
        self.model.max_seq_length = max_seq_length

    def fit(self, texts):
        # Frozen pretrained encoder — nothing to learn.
        return self

    def transform(self, texts):
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

    def fit_transform(self, texts):
        return self.transform(texts)

    @property
    def embedding_dim(self):
        return self.model.get_sentence_embedding_dimension()


def encode_dataframe(df):
    encoder = QwenEncoder()
    X = encoder.fit_transform(df["text"])
    y = df["category"].to_numpy()
    return X, y, encoder


def get_qwen_features(depth):
    df = load_all_datasets_to_df(depth)
    X, y, encoder = encode_dataframe(df)
    save_qwen(X, y, encoder)
    return X, y, encoder


def save_qwen(X, y, encoder, out_dir=QWEN_DIR):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), np.asarray(X))
    np.save(os.path.join(out_dir, "y.npy"), np.asarray(y))
    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    print(f"\n[SAVE] Qwen matrix: {X.shape} -> {out_dir}")
    train_idx, test_idx = make_split(y)
    save_split(out_dir, train_idx, test_idx)


def load_qwen(out_dir=QWEN_DIR):
    if os.path.exists(os.path.join(out_dir, "X.npy")) and \
        os.path.exists(os.path.join(out_dir, "y.npy")) and \
        os.path.exists(os.path.join(out_dir, "encoder.joblib")):

        print(f"\n[LOAD] Found existing Qwen features in {out_dir}. Loading...")
        X = np.load(os.path.join(out_dir, "X.npy"))
        y = np.load(os.path.join(out_dir, "y.npy"), allow_pickle=True)
        encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
        get_or_make_split(y, out_dir)
        return X, y, encoder

    print(f"\n[LOAD] No existing Qwen features found in {out_dir}. Computing from scratch...")
    depth = int(out_dir.split("_depth")[-1]) if "_depth" in out_dir else None
    X, y, encoder = get_qwen_features(depth)
    save_qwen(X, y, encoder, out_dir)
    return X, y, encoder