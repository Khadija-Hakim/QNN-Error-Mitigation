import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from src.qnn import train_qnn, qnn_predict_proba, QNN_DEPTH

RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs("results", exist_ok=True)
os.makedirs("results/fold_data", exist_ok=True)

# Binary Iris: class 0 (Setosa) and class 1 (Versicolor).
iris = load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]

print("Dataset shape:", X.shape)
print("Class distribution:", np.bincount(y))

cv = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

results = []
fold_assignments = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    print(f"\n========== Fold {fold} ==========")

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for index in train_idx:
        fold_assignments.append({
            "sample_index": int(index), "fold": fold, "set": "train"
        })
    for index in test_idx:
        fold_assignments.append({
            "sample_index": int(index), "fold": fold, "set": "test"
        })

    # Fit all preprocessing on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    # Classical baseline 1: Logistic Regression.
    logistic_model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    )
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)

    # Classical baseline 2: RBF-SVM.
    svm_model = SVC(kernel="rbf")
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    # Train one QNN per fold. These exact parameters are reused by run_noisy.py.
    trained_params = train_qnn(
        X_train, y_train, depth=QNN_DEPTH, seed=RANDOM_STATE + fold
    )

    qnn_probabilities = qnn_predict_proba(
        X_test, trained_params, depth=QNN_DEPTH
    )
    qnn_predictions = (qnn_probabilities >= 0.5).astype(int)

    # Save everything needed to reproduce the ideal/noisy comparison.
    np.savez_compressed(
        f"results/fold_data/fold_{fold}.npz",
        X_test=X_test,
        y_test=y_test,
        params=trained_params,
        depth=np.asarray(QNN_DEPTH),
    )

    model_predictions = {
        "Logistic Regression": logistic_predictions,
        "RBF-SVM": svm_predictions,
        "QNN Ideal Simulator (exact)": qnn_predictions,
    }

    for model_name, predictions in model_predictions.items():
        row = {
            "fold": fold,
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        results.append(row)
        print(
            f"{model_name}: Accuracy = {row['accuracy']:.4f}, "
            f"F1 = {row['f1_score']:.4f}"
        )

results_df = pd.DataFrame(results)
results_df.to_csv("results/week1_results.csv", index=False)

fold_df = pd.DataFrame(fold_assignments)
fold_df.to_csv("results/fold_assignments.csv", index=False)

summary = results_df.groupby("model")[
    ["accuracy", "precision", "recall", "f1_score"]
].mean()
summary.to_csv("results/model_summary.csv")

print("\n========== Average Results ==========")
print(summary)
print("\nSaved baseline results and reusable fold data.")
