import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV file with extracted features")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    X = df.drop(columns=["label", "ts"])

    scaler = joblib.load("models/scaler.joblib")
    X_scaled = scaler.transform(X)

    model = load_model("models/lstm_model.h5")

    preds = model.predict(X_scaled, verbose=1)
    preds_class = (preds > 0.5).astype(int)

    df["pred"] = preds_class

    print(df[["ts", "label", "pred"]].head(20))
    print("\nTotal attack packets predicted:", df["pred"].sum())
    print("Total normal packets predicted:", len(df) - df["pred"].sum())
