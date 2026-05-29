from src.datasets.amazon_dataset import load_all_datasets_to_df

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import random

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
    X = encoder.fit_transform(df["text"])
    y = df["category"]

    print("\n[SANITY CHECK] Sample vocabulary:", random.choices(list(encoder.vocabulary), k=10))
    return X, y, encoder

def train_logistic_regression(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\n[SCORES] TF-IDF Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}")
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