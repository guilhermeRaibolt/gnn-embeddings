import os

from src.datasets.amazon_dataset import load_all_datasets_to_df

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import torch

DEFAULT_BERT_MODEL = "all-MiniLM-L6-v2"
SBERT_DIR = "data/sbert"
X_PATH = os.path.join(SBERT_DIR, "X.npz")
Y_PATH = os.path.join(SBERT_DIR, "y.npy")
ENCODER_PATH = os.path.join(SBERT_DIR, "encoder.joblib")

class SBERTEncoder:
    def __init__(
        self,
        model_name=DEFAULT_BERT_MODEL,
        batch_size=64,
        normalize=True,
        device=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

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
    encoder = SBERTEncoder()
    X = encoder.fit_transform(df["text"])
    y = df["category"]
    return X, y, encoder


def get_sbert_features():
    df = load_all_datasets_to_df()
    X, y, encoder = encode_dataframe(df)
    return X, y, encoder


def save_sbert(X, y, encoder, out_dir=SBERT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    sparse.save_npz(os.path.join(out_dir, "X.npz"), X)
    np.save(os.path.join(out_dir, "y.npy"), np.asarray(y))
    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    print(f"\n[SAVE] SBERT matrix: {X.shape} -> {out_dir}")


def load_sbert(out_dir=SBERT_DIR):
    if os.path.exists(os.path.join(out_dir, "X.npz")) and \
        os.path.exists(os.path.join(out_dir, "y.npy")) and \
        os.path.exists(os.path.join(out_dir, "encoder.joblib")):
        X = sparse.load_npz(os.path.join(out_dir, "X.npz"))
        y = np.load(os.path.join(out_dir, "y.npy"), allow_pickle=True)
        encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
        return X, y, encoder
    
    X, y, encoder = get_sbert_features()
    save_sbert(X, y, encoder, out_dir)
    return X, y, encoder