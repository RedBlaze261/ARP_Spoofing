import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data():
    normal = pd.read_csv("data/features/normal.csv")
    attack = pd.read_csv("data/features/attack.csv")

    normal["label"] = 0
    attack["label"] = 1

    df = pd.concat([normal, attack], ignore_index=True)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    scaler = joblib.load("models/scaler.joblib")
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_scaled, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-cm", action="store_true")
    args = parser.parse_args()

    X, y = load_data()
    model = tf.keras.models.load_model("models/lstm_model.h5")

    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs >= args.threshold).astype(int)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))

    if args.show_cm:
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
