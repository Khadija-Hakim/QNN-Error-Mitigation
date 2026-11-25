import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from src.qnn import qnn_predict_proba
from src.noisy_simulation import (
    make_simulator, predict_shot_probabilities
)

SHOTS = 1000
RANDOM_SEED = 2026

NOISE_LEVELS = {
    "Low noise": {
        "p1": 0.001,
        "p2": 0.005,
        "p_readout": 0.01,
    },
    "Medium noise": {
        "p1": 0.01,
        "p2": 0.02,
        "p_readout": 0.03,
    },
}

os.makedirs("results", exist_ok=True)

def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
    }


fold_files = sorted(
    glob.glob("results/fold_data/fold_*.npz"),
    key=lambda path: int(os.path.basename(path).split("_")[1].split(".")[0])
)

if not fold_files:
    raise FileNotFoundError(
        "No saved fold data found. Run `python main.py` first."
    )

all_rows = []

for fold_number, fold_file in enumerate(fold_files, start=1):
    data = np.load(fold_file)
    X_test = data["X_test"]
    y_test = data["y_test"]
    params = data["params"]
    depth = int(data["depth"])

    print(f"\n========== Noisy experiments: Fold {fold_number} ==========")

    # Exact statevector reference from the same trained parameters.
    exact_probs = qnn_predict_proba(X_test, params, depth)
    exact_metrics = calculate_metrics(y_test, exact_probs)
    all_rows.append({
        "fold": fold_number,
        "condition": "Ideal exact statevector",
        "shots": np.nan,
        "p1": 0.0, "p2": 0.0, "p_readout": 0.0,
        **exact_metrics,
    })

    # Ideal finite-shot simulation: same shot count as noisy experiments.
    ideal_simulator = make_simulator(
        noise_settings=None, seed=RANDOM_SEED + fold_number
    )
    ideal_probs = predict_shot_probabilities(
        X_test, params, depth, ideal_simulator,
        shots=SHOTS, seed=RANDOM_SEED + fold_number
    )
    ideal_metrics = calculate_metrics(y_test, ideal_probs)
    all_rows.append({
        "fold": fold_number,
        "condition": "Ideal finite-shot",
        "shots": SHOTS,
        "p1": 0.0, "p2": 0.0, "p_readout": 0.0,
        **ideal_metrics,
    })

    # Noisy finite-shot simulations using the SAME trained parameters.
    for level_index, (level_name, settings) in enumerate(NOISE_LEVELS.items()):
        noisy_simulator = make_simulator(
            noise_settings=settings,
            seed=RANDOM_SEED + fold_number + level_index + 10,
        )
        noisy_probs = predict_shot_probabilities(
            X_test, params, depth, noisy_simulator,
            shots=SHOTS,
            seed=RANDOM_SEED + fold_number + level_index + 10,
        )
        noisy_metrics = calculate_metrics(y_test, noisy_probs)

        all_rows.append({
            "fold": fold_number,
            "condition": level_name,
            "shots": SHOTS,
            **settings,
            **noisy_metrics,
        })

        print(
            f"{level_name}: accuracy={noisy_metrics['accuracy']:.4f}, "
            f"F1={noisy_metrics['f1_score']:.4f}"
        )

results_df = pd.DataFrame(all_rows)
results_df.to_csv("results/noisy_results.csv", index=False)

comparison = results_df.groupby("condition")[
    ["accuracy", "precision", "recall", "f1_score"]
].mean()
comparison.to_csv("results/noise_comparison.csv")

print("\n========== Mean results across folds ==========")
print(comparison)
print("\nSaved results/noisy_results.csv and results/noise_comparison.csv.")
