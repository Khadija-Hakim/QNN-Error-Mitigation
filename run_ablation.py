import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from src.qnn import train_qnn
from src.noisy_simulation import (
    make_simulator, predict_shot_probabilities
)

RANDOM_STATE = 42
N_SPLITS = 5
DEPTHS = [1, 2, 3]
SHOTS = 1000

# A single, explicitly specified medium-noise condition for the depth study.
MEDIUM_NOISE = {"p1": 0.01, "p2": 0.02, "p_readout": 0.03}

iris = load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]

cv = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

def metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
    }

rows = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    for depth in DEPTHS:
        print(f"Training fold {fold}, circuit depth {depth}...")
        params = train_qnn(
            X_train, y_train, depth=depth,
            seed=RANDOM_STATE + fold + depth
        )

        ideal_simulator = make_simulator(
            noise_settings=None, seed=3000 + fold + depth
        )
        ideal_probs = predict_shot_probabilities(
            X_test, params, depth, ideal_simulator,
            shots=SHOTS, seed=3000 + fold + depth
        )
        rows.append({
            "fold": fold, "depth": depth,
            "condition": "Ideal finite-shot",
            **metrics(y_test, ideal_probs),
        })

        noisy_simulator = make_simulator(
            noise_settings=MEDIUM_NOISE,
            seed=4000 + fold + depth,
        )
        noisy_probs = predict_shot_probabilities(
            X_test, params, depth, noisy_simulator,
            shots=SHOTS, seed=4000 + fold + depth
        )
        rows.append({
            "fold": fold, "depth": depth,
            "condition": "Medium noise",
            **metrics(y_test, noisy_probs),
        })

results = pd.DataFrame(rows)
results.to_csv("results/depth_ablation.csv", index=False)

summary = results.groupby(["depth", "condition"])[
    ["accuracy", "precision", "recall", "f1_score"]
].mean()
summary.to_csv("results/depth_ablation_summary.csv")

print("\n========== Depth ablation summary ==========")
print(summary)
print("\nSaved depth ablation CSV files.")
