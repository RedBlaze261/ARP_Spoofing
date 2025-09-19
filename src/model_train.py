import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

def load_data():
    # Load feature CSVs
    normal = pd.read_csv("data/features/normal.csv")
    attack = pd.read_csv("data/features/attack.csv")
    pure_attack = pd.read_csv("data/features/pure_attack.csv")  # include pure attack

    # Assign labels
    normal["label"] = 0
    attack["label"] = 1
    pure_attack["label"] = 1  # attacks

    # Combine all data
    df = pd.concat([normal, attack, pure_attack], ignore_index=True)

    # Separate features and labels
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.joblib")

    # reshape for LSTM: (samples, timesteps=1, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_scaled, y, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", default="64,32", help="Hidden layer sizes, e.g. 64,32")
    parser.add_argument("--max-iter", type=int, default=50)
    args = parser.parse_args()

    hidden = [int(h) for h in args.hidden.split(",")]

    X, y, scaler = load_data()

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Compute class weights for imbalance
    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(hidden[0], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden[1]))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.max_iter,
        class_weight=class_weights,
        batch_size=32
    )

    # Save model
    model.save("models/lstm_model.h5")
    print("[+] Model saved to models/lstm_model.h5")

