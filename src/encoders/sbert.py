from src.datasets.amazon_dataset import load_all_datasets_to_df

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import torch

DEFAULT_BERT_MODEL = "all-MiniLM-L6-v2"

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

    print(f"\n[SANITY CHECK] Embedding shape: {X.shape} (dim={encoder.embedding_dim})")
    return X, y, encoder


def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n[SCORES] SBERT Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    return clf

if __name__ == "__main__":

    df = load_all_datasets_to_df()
    X, y, encoder = encode_dataframe(df)
    lr_model = train_logistic_regression(X, y)

    print("\n[TEST] Predicting categories for new inputs:")

    mock_data = [
        "Acoustic Guitar",
        "Truck 4x4",
        "Car Radio Sound System",
    ]

    mock_X = encoder.transform(mock_data)
    mock_pred = lr_model.predict(mock_X)
    for input, pred in zip(mock_data, mock_pred):
        print(f"Input: {input}\nPredicted Category: {pred}\n")
