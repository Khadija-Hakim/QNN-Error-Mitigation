
import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
QNN_DEPTH = 2
MAX_ITERATIONS = 100

os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 2. Load binary Iris dataset
# --------------------------------------------------

iris = load_iris()

# Select only the first two classes
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]

print("Dataset shape:", X.shape)
print("Class distribution:", np.bincount(y))


# --------------------------------------------------
# 3. QNN circuit and prediction
# --------------------------------------------------

def qnn_probability(features, params, depth=QNN_DEPTH):

    qc = QuantumCircuit(2)

    # Encode two input features into two qubits
    # tanh keeps values between -1 and 1
    angles = np.pi * (np.tanh(features) + 1) / 2

    qc.ry(float(angles[0]), 0)
    qc.ry(float(angles[1]), 1)

    # Trainable circuit layers
    for layer in range(depth):

        qc.ry(float(params[layer, 0]), 0)
        qc.rz(float(params[layer, 1]), 0)

        qc.ry(float(params[layer, 2]), 1)
        qc.rz(float(params[layer, 3]), 1)

        # Entangling operation
        qc.cx(0, 1)

    # Exact statevector simulation
    state = Statevector.from_instruction(qc)
    probabilities = state.probabilities()

    # Qiskit uses little-endian qubit ordering.
    # Indices 1 and 3 correspond to qubit 0 = 1.
    probability_class_1 = probabilities[1] + probabilities[3]

    return float(probability_class_1)


def qnn_predict_proba(X, params, depth=QNN_DEPTH):

    return np.array([
        qnn_probability(x, params, depth)
        for x in X
    ])


def qnn_loss(flat_params, X_train, y_train, depth):

    params = flat_params.reshape(depth, 4)

    probabilities = qnn_predict_proba(
        X_train, params, depth
    )

    probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )

    # Binary cross-entropy loss
    loss = -np.mean(
        y_train * np.log(probabilities)
        + (1 - y_train) * np.log(1 - probabilities)
    )

    return loss


def train_qnn(X_train, y_train, depth=QNN_DEPTH, seed=42):

    rng = np.random.default_rng(seed)

    initial_params = rng.normal(
        loc=0,
        scale=0.1,
        size=(depth, 4)
    )

    result = minimize(
        qnn_loss,
        initial_params.flatten(),
        args=(X_train, y_train, depth),
        method="COBYLA",
        options={
            "maxiter": MAX_ITERATIONS,
            "rhobeg": 0.2
        }
    )

    trained_params = result.x.reshape(depth, 4)

    return trained_params


# --------------------------------------------------
# 4. Five-fold cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
fold_assignments = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):

    print(f"\n========== Fold {fold} ==========")

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Save deterministic fold assignments
    for index in train_idx:
        fold_assignments.append({
            "sample_index": int(index),
            "fold": fold,
            "set": "train"
        })

    for index in test_idx:
        fold_assignments.append({
            "sample_index": int(index),
            "fold": fold,
            "set": "test"
        })

    # Fit preprocessing only on training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=2)

    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    # --------------------------------------------------
    # 5. Classical baseline: Logistic Regression
    # --------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)

    # --------------------------------------------------
    # 6. Classical baseline: RBF-SVM
    # --------------------------------------------------

    svm_model = SVC(kernel="rbf")

    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)

    # --------------------------------------------------
    # 7. Train QNN on ideal simulator
    # --------------------------------------------------

    trained_params = train_qnn(
        X_train,
        y_train,
        depth=QNN_DEPTH,
        seed=RANDOM_STATE + fold
    )

    qnn_probabilities = qnn_predict_proba(
        X_test,
        trained_params,
        QNN_DEPTH
    )

    qnn_predictions = (
        qnn_probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------
    # 8. Evaluate all models
    # --------------------------------------------------

    model_predictions = {
        "Logistic Regression": logistic_predictions,
        "RBF-SVM": svm_predictions,
        "QNN Ideal Simulator": qnn_predictions
    }

    for model_name, predictions in model_predictions.items():

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "fold": fold,
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        print(
            f"{model_name}: "
            f"Accuracy = {accuracy:.4f}, "
            f"F1 = {f1:.4f}"
        )


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/week1_results.csv",
    index=False
)

fold_df = pd.DataFrame(fold_assignments)

fold_df.to_csv(
    "results/fold_assignments.csv",
    index=False
)

print("\n========== Average Results ==========")

summary = results_df.groupby("model")[
    ["accuracy", "precision", "recall", "f1_score"]
].mean()

print(summary)

summary.to_csv("results/model_summary.csv")

print("\nResults saved in the results/ folder.")
