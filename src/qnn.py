import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

QNN_DEPTH = 2
MAX_ITERATIONS = 100


def build_qnn_circuit(features, params, depth=QNN_DEPTH, measure=False):
    """Build the same two-qubit QNN architecture used in the original main.py."""
    if measure:
        qc = QuantumCircuit(2, 1)
    else:
        qc = QuantumCircuit(2)

    angles = np.pi * (np.tanh(features) + 1) / 2
    qc.ry(float(angles[0]), 0)
    qc.ry(float(angles[1]), 1)

    for layer in range(depth):
        qc.ry(float(params[layer, 0]), 0)
        qc.rz(float(params[layer, 1]), 0)
        qc.ry(float(params[layer, 2]), 1)
        qc.rz(float(params[layer, 3]), 1)
        qc.cx(0, 1)

    if measure:
        # Class label 1 corresponds to qubit 0 being measured as 1.
        qc.measure(0, 0)

    return qc


def qnn_probability(features, params, depth=QNN_DEPTH):
    qc = build_qnn_circuit(features, params, depth=depth, measure=False)
    state = Statevector.from_instruction(qc)
    probabilities = state.probabilities()
    # Qiskit's little-endian ordering: indices 1 and 3 have qubit 0 = 1.
    return float(probabilities[1] + probabilities[3])


def qnn_predict_proba(X, params, depth=QNN_DEPTH):
    return np.asarray([qnn_probability(x, params, depth) for x in X])


def qnn_loss(flat_params, X_train, y_train, depth):
    params = flat_params.reshape(depth, 4)
    probabilities = qnn_predict_proba(X_train, params, depth)
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(
        y_train * np.log(probabilities)
        + (1 - y_train) * np.log(1 - probabilities)
    ))


def train_qnn(X_train, y_train, depth=QNN_DEPTH, seed=42,
              max_iterations=MAX_ITERATIONS):
    rng = np.random.default_rng(seed)
    initial_params = rng.normal(loc=0, scale=0.1, size=(depth, 4))

    result = minimize(
        qnn_loss,
        initial_params.flatten(),
        args=(X_train, y_train, depth),
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.2},
    )
    return result.x.reshape(depth, 4)
