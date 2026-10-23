# Quantum Reinforcement Learning for Cybersecurity Threat Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-blueviolet.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IBM Quantum](https://img.shields.io/badge/IBM-Quantum-orange.svg)](https://quantum.ibm.com/)

A comparative study between Classical Deep Q-Network (DQN) and Quantum-enhanced DQN (QDQN) for cybersecurity threat classification, executed on real IBM Quantum hardware.

---

## Results

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| DQN (Classical) | 10.0% | 0.033 | 0.12s |
| QDQN (Quantum) | 15.0% | 0.039 | 380s |

**Key Finding**: Under identical training conditions, QDQN achieved 50% higher accuracy than classical DQN.

> See [`RESULTS.txt`](RESULTS.txt) for full experimental details.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/quantum-rl-threat-classification.git
cd quantum-rl-threat-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure IBM Quantum Credentials

Copy the example environment file and fill in your token:
```bash
cp .env.example .env
# Then edit .env and add your IBM Quantum API token
```

Get your free API token from [IBM Quantum](https://quantum.ibm.com/account), then run the setup utility:
```bash
python config.py
```

### 4. Run Fair Comparison
```bash
# On real quantum hardware
python train.py --mode compare --real-hardware

# Simulation only (no IBM Quantum account needed)
python train.py --mode compare
```

---

## Project Structure

```
quantum_rl/
├── train.py            # Main training script
├── quantum_network.py  # Quantum Q-Network (IBM Qiskit)
├── models.py           # Classical DQN agent
├── environment.py      # RL environment
├── utils.py            # Utilities
├── config.py           # IBM Quantum configuration utility
├── requirements.txt    # Python dependencies
├── RESULTS.txt         # Experimental results
├── .env.example        # Template for credentials (copy to .env)
├── data/
│   └── dataset.csv     # 400 threat samples (8 classes)
└── checkpoints/        # Saved models and results (gitignored)
```

---

## Command Line Options

```bash
python train.py [OPTIONS]

Options:
  --mode {dqn,qdqn,compare}  Training mode (default: compare)
  --real-hardware            Use real IBM Quantum hardware
  --backend NAME             Specific IBM Quantum backend name
  --epochs N                 Number of epochs (default: 2)
  --train-samples N          Training samples (default: 60)
  --test-samples N           Test samples (default: 20)
  --qubits N                 Number of qubits (default: 4)
  --shots N                  Quantum shots (default: 256)
  --max-quantum-jobs N       Maximum quantum jobs (default: 8)
```

---

## Methodology

### Fair Comparison Protocol

Both models are trained under identical conditions to ensure scientific validity:

| Parameter | Value |
|-----------|-------|
| Training samples | 60 |
| Test samples | 20 |
| Epochs | 2 |
| Gradient updates | 6 |
| Random seed | 42 |

### Model Architectures

**Classical DQN**
- Input: 13 features
- Hidden layers: 128 × 2 neurons
- Output: 8 threat classes
- Total parameters: ~18,000

**Quantum DQN**
- Classical preprocessing: 13 → 32 → 4
- Quantum circuit: 4-qubit RealAmplitudes (2 repetitions)
- Observables: Pauli-Z measurements
- Classical head: 4 → 16 → 8
- Quantum parameters: 12 rotation angles

---

## Dataset

The dataset contains 400 samples across 8 cybersecurity threat classes:

- Benign
- Botnet
- DoS (Denial of Service)
- Phishing
- Port Scan
- Ransomware
- SQL Injection
- XSS (Cross-Site Scripting)

---

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- Qiskit 1.0+
- qiskit-ibm-runtime 0.20+
- IBM Quantum account (free tier supported) — only needed for `--real-hardware`

---

## Frontend Dashboard

A React-based visualization dashboard is included in the `frontend/` directory.

```bash
cd frontend
npm install
npm run dev
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{quantum_rl_threat_classification,
  title={Comparative Study of Classical vs Quantum Reinforcement Learning
         for Cybersecurity Threat Classification},
  author={},
  year={2026},
  howpublished={\url{https://github.com/<your-username>/quantum-rl-threat-classification}}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- IBM Quantum for providing access to quantum hardware
- Qiskit for the quantum computing framework
- PyTorch for the deep learning framework
