from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.encoders.tf_idf import load_tfidf


def train_logistic_regression(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n[SCORES] TF-IDF Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    return clf


if __name__ == "__main__":
    X, y, encoder = load_tfidf()
    clf = train_logistic_regression(X, y)

    # print("\n[TEST] Predicting categories for new inputs:")
    # mock_data = [
    #     "Acoustic Guitar",
    #     "Truck 4x4",
    #     "Car Radio Sound System",
    # ]
    # mock_X = encoder.transform(mock_data)
    # mock_pred = clf.predict(mock_X)
    # for text, pred in zip(mock_data, mock_pred):
    #     print(f"Input: {text}\nPredicted Category: {pred}\n")
