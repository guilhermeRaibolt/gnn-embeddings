import os
import time

import joblib
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

from src.datasets.amazon_dataset import load_all_datasets_to_df
from src.datasets.splits import make_split, save_split, get_or_make_split

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_TEXT_DIR = "data/clip_text"
X_PATH = os.path.join(CLIP_TEXT_DIR, "X.npy")
Y_PATH = os.path.join(CLIP_TEXT_DIR, "y.npy")
ENCODER_PATH = os.path.join(CLIP_TEXT_DIR, "encoder.joblib")


class CLIPTextEncoder:
    def __init__(
        self,
        model_name=DEFAULT_CLIP_MODEL,
        batch_size=256,
        normalize=True,
        device=None,
        max_seq_length=77,  # CLIP's native hard context limit
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length

        # Initialize Hugging Face tokenizer and model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def fit(self, texts):
        # Frozen pretrained encoder — nothing to learn.
        return self

    def transform(self, texts):
        texts = list(texts)
        embeddings = []

        # Process text collections in mini-batches matching the existing pipelines
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            
            # Tokenize batch inputs safely with truncation and padding
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # pooler_output corresponds to the embedded representations of the [EOS] token
                batch_emb = outputs.pooler_output

                if self.normalize:
                    batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=-1)
                
                embeddings.append(batch_emb.cpu().numpy())

        return np.vstack(embeddings)

    def fit_transform(self, texts):
        return self.transform(texts)

    @property
    def embedding_dim(self):
        return self.model.config.hidden_size


def encode_dataframe(df):
    encoder = CLIPTextEncoder()
    print(f"\n[ENCODE] CLIP-text encoding {len(df)} documents...")
    start = time.perf_counter()
    X = encoder.fit_transform(df["text"])
    elapsed = time.perf_counter() - start
    print(f"[ENCODE] CLIP-text done in {elapsed:.2f}s -> matrix {X.shape}")
    y = df["category"].to_numpy()
    return X, y, encoder


def get_clip_text_features(depth):
    df = load_all_datasets_to_df(depth)
    X, y, encoder = encode_dataframe(df)
    return X, y, encoder


def save_clip_text(X, y, encoder, out_dir=CLIP_TEXT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), np.asarray(X))
    np.save(os.path.join(out_dir, "y.npy"), np.asarray(y))
    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    print(f"\n[SAVE] CLIP-text matrix: {X.shape} -> {out_dir}")
    train_idx, test_idx = make_split(y)
    save_split(out_dir, train_idx, test_idx)


def load_clip_text(out_dir=CLIP_TEXT_DIR):
    if os.path.exists(os.path.join(out_dir, "X.npy")) and \
        os.path.exists(os.path.join(out_dir, "y.npy")) and \
        os.path.exists(os.path.join(out_dir, "encoder.joblib")):

        print(f"\n[LOAD] Found existing CLIP-text features in {out_dir}. Loading...")
        X = np.load(os.path.join(out_dir, "X.npy"))
        y = np.load(os.path.join(out_dir, "y.npy"), allow_pickle=True)
        encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
        get_or_make_split(y, out_dir)
        return X, y, encoder

    print(f"\n[LOAD] No existing CLIP-text features found in {out_dir}. Computing from scratch...")
    depth = int(out_dir.split("_depth")[-1]) if "_depth" in out_dir else None
    X, y, encoder = get_clip_text_features(depth)
    save_clip_text(X, y, encoder, out_dir)
    return X, y, encoder