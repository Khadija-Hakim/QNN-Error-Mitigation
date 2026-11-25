import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from src.qnn import build_qnn_circuit


def make_noise_model(p1=0.001, p2=0.005, p_readout=0.01):
    """
    Construct a simple illustrative noise model.

    p1: depolarizing probability applied after SX/X one-qubit gates.
    p2: depolarizing probability applied after CX two-qubit gates.
    p_readout: symmetric probability of reporting the wrong measurement bit.

    These are experimental settings, NOT measured IBM hardware calibration data.
    """
    for name, value in {
        "p1": p1, "p2": p2, "p_readout": p_readout
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    noise_model = NoiseModel()

    if p1 > 0:
        one_qubit_error = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(one_qubit_error, ["sx", "x"])

    if p2 > 0:
        two_qubit_error = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])

    if p_readout > 0:
        # Rows represent the true bit; columns represent the reported bit.
        readout_error = ReadoutError([
            [1 - p_readout, p_readout],
            [p_readout, 1 - p_readout],
        ])
        noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def make_simulator(noise_settings=None, seed=1234):
    if noise_settings is None:
        return AerSimulator(seed_simulator=seed)

    model = make_noise_model(**noise_settings)
    return AerSimulator(noise_model=model, seed_simulator=seed)


def predict_shot_probabilities(
    X, params, depth, simulator, shots=1000, seed=1234
):
    """
    Return estimated P(class 1) for each input using finite shots.
    The circuit measures qubit 0 into classical bit 0.
    """
    circuits = [
        build_qnn_circuit(x, params, depth=depth, measure=True)
        for x in X
    ]

    compiled_circuits = transpile(
        circuits, simulator, optimization_level=0
    )

    job = simulator.run(
        compiled_circuits,
        shots=shots,
        seed_simulator=seed,
    )
    result = job.result()

    probabilities = []
    for circuit_index in range(len(circuits)):
        counts = result.get_counts(circuit_index)
        ones = counts.get("1", 0)
        probabilities.append(ones / shots)

    return np.asarray(probabilities, dtype=float)
