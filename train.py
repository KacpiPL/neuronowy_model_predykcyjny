import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

def main():
    X, y = load_iris(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    model = MLPClassifier(
        hidden_layer_sizes=(16,), max_iter=1000, random_state=42
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_val_sc)
    print(classification_report(y_val, y_pred, digits=4))

    joblib.dump(model, OUT_DIR / "model.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    print(f"Zapisano model i scaler w {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
