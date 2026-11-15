# From Simulator to Hardware: Error Mitigation for QNN Inference

## Project Overview

This project investigates the performance difference between a Quantum Neural Network (QNN) running on a quantum simulator and a real quantum computer.

Quantum simulators provide a controlled environment for developing and testing quantum models. However, real quantum hardware is affected by noise, gate errors, and measurement errors, which can affect model predictions.

The main objective of this project is to evaluate how much performance a QNN loses when transferred from an ideal simulator to real quantum hardware and whether error mitigation techniques can reduce this performance gap.

## Research Question

How much classification performance does a simulator-trained QNN lose when executed on real quantum hardware, and to what extent can TREX and Zero-Noise Extrapolation recover this performance?

## Project Objectives

- Implement and train a small Quantum Neural Network using Qiskit.
- Compare QNN performance on ideal and noisy quantum simulators.
- Evaluate the trained model on real IBM quantum hardware.
- Investigate the effectiveness of TREX and Zero-Noise Extrapolation (ZNE).
- Analyze the effect of circuit depth on model performance.
- Compare accuracy, measurement shot cost, and execution time.

## Dataset

**Iris Dataset**

We use the Iris dataset from Scikit-learn for binary classification.

- Total samples: 100 (two selected classes)
- Original features: 4
- Selected classes: Setosa and Versicolor
- Reduced features: 2 using Principal Component Analysis (PCA)

The features are standardized before PCA. A deterministic stratified 5-fold cross-validation strategy is used for evaluation. Preprocessing is fitted on each training fold separately to prevent data leakage.

## Models

### Classical Baselines

We use two classical machine learning models for comparison:

1. Logistic Regression
2. Support Vector Machine (RBF Kernel)

### Quantum Neural Network

The initial QNN consists of:

- Two qubits.
- Angle encoding for the two input features.
- Trainable RY and RZ rotation gates.
- CNOT gates for entanglement.
- Trainable circuit depth represented by L.

The QNN is initially trained using an ideal statevector simulator.

## Experimental Scenarios

We plan to evaluate the trained QNN under four conditions:

| Scenario | Description |
|---|---|
| Ideal Simulator | Execution without intentionally introduced noise |
| Noisy Simulator | Execution with an artificial noise model |
| Real Quantum Hardware | Execution on IBM Quantum without mitigation |
| Hardware + TREX + ZNE | Execution using error mitigation techniques |

### Error Mitigation Techniques

**1. TREX (Twirled Readout Error eXtinction)**

A readout error mitigation technique that addresses errors introduced during quantum measurements.

**2. Zero-Noise Extrapolation (ZNE)**

A technique that evaluates circuits at different effective noise levels and extrapolates the results to estimate the zero-noise expectation value.

## Evaluation Metrics

We evaluate the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Measurement shot cost
- Execution time

## Project Structure

```text
QNN-Error-Mitigation/
│
├── README.md
├── requirements.txt
├── config.py
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── classical.py
│   ├── qnn.py
│   └── metrics.py
│
├── results/
│
└── docs/
    └── implementation_plan.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Khadija-Hakim/QNN-Error-Mitigation.git
cd QNN-Error-Mitigation
```

### 2. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it:

**Windows (Git Bash):**
```bash
source .venv/Scripts/activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

## Results

The initial implementation generates the following files in the `results/` directory:

| File | Description |
|---|---|
| `fold_metrics.csv` | Evaluation metrics for each model across five folds |
| `fold_assignments.csv` | Training and testing sample assignments |
| `summary_metrics.csv` | Mean and standard deviation of model performance |
| `run_config.json` | Experiment configuration and random seed |

Actual experimental results will be added after running the implementation.

## Current Progress

- [x] Project research question and experimental approach finalized.
- [x] Binary Iris dataset selected.
- [x] Initial two-qubit QNN architecture defined.
- [ ] Data pipeline and five-fold evaluation executed.
- [ ] Classical baselines evaluated.
- [ ] QNN trained and evaluated on ideal simulator.
- [ ] Noisy simulator experiments.
- [ ] IBM Quantum hardware experiments.
- [ ] TREX and ZNE implementation.
- [ ] Final comparison and analysis.

*Note: Checklist items should be updated as implementation and experiments are completed.*

## Future Work

The next stages will focus on noisy simulator evaluation, real IBM quantum hardware execution, and implementing TREX and ZNE. The final experiments will compare the effectiveness of error mitigation techniques across different circuit depths.

## References

1. Temme, K., Bravyi, S., and Gambetta, J. M. (2017). Error mitigation for short-depth quantum circuits. Physical Review Letters, 119, 180509.
2. Kandala, A., et al. (2019). Error mitigation extends the computational reach of a noisy quantum processor. Nature, 567, 491–495.
3. van den Berg, E., et al. (2022). Reference details to be finalized after bibliographic verification.

---

**Project:** From Simulator to Hardware: Error Mitigation for QNN Inference  
**Tools:** Python, Qiskit, Scikit-learn, IBM Quantum  
**Dataset:** Iris (Binary Classification)
