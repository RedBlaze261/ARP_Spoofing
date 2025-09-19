import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess(csv_glob, output_dir="data/processed", scaler_path="models/scaler.joblib"):
    all_files = glob.glob(csv_glob)
    dfs = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dfs, ignore_index=True)

    X = data.drop(columns=["label"])
    y = data["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    print(f"[+] Scaler saved to {scaler_path}")

    # Split train/test 80/20
    split = int(0.8 * len(X_scaled))
    np.savez(f"{output_dir}/train.npz", X=X_scaled[:split], y=y[:split])
    np.savez(f"{output_dir}/test.npz", X=X_scaled[split:], y=y[split:])
    print(f"[+] Preprocessed datasets saved in {output_dir}")

if __name__ == "__main__":
    preprocess("data/features/*.csv")
