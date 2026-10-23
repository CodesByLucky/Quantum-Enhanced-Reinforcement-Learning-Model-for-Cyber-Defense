# Results: Classical DQN vs Quantum DQN

**Run date:** 2 February 2026 · **Backend:** IBM `ibm_fez` (156 qubits) · **Seed:** 42

## Configuration

Both models were trained and tested under identical conditions.

| Parameter | DQN (classical) | QDQN (hybrid quantum) |
|---|---|---|
| Training samples | 60 | 60 |
| Test samples | 20 | 20 |
| Epochs | 2 | 2 |
| Gradient updates | 6 (3 per epoch) | 6 (3 per epoch) |
| Batch size | 20 | 20 |
| Model | 13 → 128 → 128 → 8 | 13 → 32 → 4 → 4-qubit circuit → 16 → 8 |
| Shots | – | 256 |

## Results

| Metric | DQN | QDQN |
|---|---|---|
| Accuracy | 10.0% (2 of 20) | 15.0% (3 of 20) |
| 95% confidence interval (Clopper-Pearson) | 1.2% – 31.7% | 3.2% – 37.9% |
| Weighted F1 | 0.033 | 0.039 |
| Training time (wall clock) | 0.12 s | 380.4 s (~3,100× slower) |
| Quantum jobs | – | 7 (within the free-tier limit of 8) |
| Final training loss | 0.018 | 0.105 |

Raw numbers: [`checkpoints/results.json`](checkpoints/results.json).

## Interpretation

- **No accuracy difference can be claimed.** Random guessing among 8 classes scores 12.5%. Both models are at chance level, the gap is a single test sample, and the confidence intervals overlap almost entirely.
- **Neither model had a real chance to learn.** Six gradient updates on 60 samples is far too little training; the tiny budget was set by the IBM Quantum free tier.
- **Only the classical head of the QDQN was trained.** The circuit runs outside PyTorch's autograd, so the encoder and the 8 quantum rotation weights received no gradients.
- **In this run the QDQN head received 156 expectation values**, one per physical qubit of `ibm_fez`, because the observables were not mapped through the transpiler layout. This has since been fixed in `quantum_network.py` (the head now receives the 4 values from the circuit's qubits), so `checkpoints/qdqn.pt` reflects the earlier setup.

## What the run does show

- A hybrid quantum-classical Q-network running end to end on real quantum hardware, with transpilation done once and circuits batched into 20-circuit jobs to stay inside free-plan limits.
- The cost of quantum execution today: about 3,100× the wall-clock time of the classical model for the same update budget. The [project report](docs/Quantum_Cyber_Defence_Report.pdf) projects this to roughly $126,920 per training run for a 10,000-sample dataset at commercial QPU rates.

## Next steps

1. Train the circuit: parameter-shift gradients (for example `EstimatorQNN` + `TorchConnector` from qiskit-machine-learning).
2. Use a contextual-bandit target (reward only, γ = 0), since each sample is independent.
3. Compare on the full 320/80 split with the local simulator over multiple seeds, and report confidence intervals.
4. Confirm the best configuration with a small hardware run.
