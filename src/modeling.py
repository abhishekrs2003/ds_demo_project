from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


def train_model(df, test_size=0.2, random_state=42):
    y = df["target"]
    X = df.drop("target", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")


def save_model(model, model_path):
    joblib.dump(model, model_path)