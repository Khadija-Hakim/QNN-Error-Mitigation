# QNN Error Mitigation: noisy simulator experiments

## Files

- `main.py`: trains the QNN and classical baselines using stratified 5-fold CV.
  It saves each fold's test features, labels, and trained QNN parameters in
  `results/fold_data/`.
- `src/qnn.py`: shared QNN architecture, exact statevector prediction, loss,
  and training functions.
- `src/noisy_simulation.py`: Aer simulator helpers, depolarizing noise,
  readout noise, and finite-shot prediction.
- `run_noisy.py`: compares exact ideal, ideal finite-shot, low-noise, and
  medium-noise conditions using the same saved parameters and test samples.
- `run_ablation.py`: retrains at circuit depths 1, 2, and 3 and compares
  ideal finite-shot against one medium-noise setting.

## Setup

Run from the project root:

```bash
python -m pip install -r requirements.txt
```

## Run order

```bash
python main.py
python run_noisy.py
python run_ablation.py
```

Run `main.py` first. The other scripts use the fold data it saves.

## Outputs

- `results/week1_results.csv`
- `results/model_summary.csv`
- `results/fold_assignments.csv`
- `results/fold_data/fold_1.npz` ... `fold_5.npz`
- `results/noisy_results.csv`
- `results/noise_comparison.csv`
- `results/depth_ablation.csv`
- `results/depth_ablation_summary.csv`

## Experimental notes

The low and medium noise probabilities are illustrative simulation settings,
not measured IBM Quantum device calibration data. The noisy experiment uses
finite-shot measurement. The ideal finite-shot condition uses the same shot
count, so sampling noise is present in both conditions. The exact statevector
condition is included as a separate reference.

This project version does not implement TREX, ZNE, or execution on real IBM
Quantum hardware. Those should only be reported as completed after they are
actually implemented and run.
