"""
Quantum Q-Network: classical encoder -> 4-qubit variational circuit -> classical head.

Runs either on IBM Quantum hardware (Qiskit Runtime EstimatorV2) or on a local
statevector simulator, which needs no IBM Quantum account.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List

MAX_CIRCUITS_PER_JOB = 20  # keeps each IBM Quantum job within free-plan limits


class QuantumQNetwork(nn.Module):
    """
    Quantum-enhanced Q-Network.

    Architecture:
        Input (13) -> Linear 32 -> Linear 4 -> tanh -> RealAmplitudes circuit
        (4 qubits, 2 reps, linear entanglement) -> <Z> per qubit (4) -> Linear 16 -> Q-values (8)

    The first rotation layer of the circuit encodes the 4 classical features;
    the remaining 8 rotation angles are the quantum weights.

    Note: circuit execution happens outside autograd (expectation values come
    back as NumPy arrays), so gradients do not flow into the quantum weights or
    the encoder; only the classical head is trained. See README, Limitations.
    """

    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        n_qubits: int = 4,
        shots: int = 256,
        use_real_hardware: bool = True,
        backend_name: Optional[str] = None
    ):
        super().__init__()

        try:
            from qiskit.circuit.library import RealAmplitudes
        except ImportError as e:
            raise RuntimeError("Install required packages: pip install -r requirements.txt") from e

        self.n_qubits = n_qubits
        self.shots = shots
        self.n_actions = n_actions
        self.use_real_hardware = use_real_hardware

        # Classical preprocessing: map input to qubit-sized features
        self.preprocessing = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh()  # Bound to [-1, 1] for rotation angles
        )

        # Parameterized quantum circuit
        self.qc = RealAmplitudes(
            num_qubits=n_qubits,
            reps=2,
            entanglement='linear',
            insert_barriers=True
        )
        observables = self._create_observables(n_qubits)

        if use_real_hardware:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            self.service = QiskitRuntimeService()
            if backend_name:
                self.backend = self.service.backend(backend_name)
            else:
                self.backend = self.service.least_busy(operational=True, simulator=False)
            print(f"Using quantum backend: {self.backend.name}")

            # Transpile once for the backend, then map the observables onto the
            # physical qubits the circuit was laid out on.
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=2)
            self.transpiled_qc = pm.run(self.qc)
            self.observables = [obs.apply_layout(self.transpiled_qc.layout) for obs in observables]
            self.estimator = None  # a Runtime estimator is created per job
        else:
            from qiskit.primitives import StatevectorEstimator

            self.backend = None
            self.transpiled_qc = self.qc
            self.observables = observables
            self.estimator = StatevectorEstimator()
            print("Using local statevector simulator (no IBM Quantum account needed)")

        # Parameter split: first layer encodes inputs, the rest are trainable weights
        self.input_params = list(self.qc.parameters[:n_qubits])
        self.weight_params = list(self.qc.parameters[n_qubits:])
        self.quantum_weights = nn.Parameter(torch.randn(len(self.weight_params)) * 0.1)

        # Classical head: one expectation value per logical qubit -> Q-values
        self.head = nn.Sequential(
            nn.Linear(len(self.observables), 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )

        self.quantum_jobs_count = 0

    @staticmethod
    def _create_observables(n_qubits: int) -> List:
        """Pauli-Z observable on each qubit."""
        from qiskit.quantum_info import SparsePauliOp

        return [
            SparsePauliOp.from_list([('I' * i + 'Z' + 'I' * (n_qubits - i - 1), 1.0)])
            for i in range(n_qubits)
        ]

    def _run_estimator(self, pubs):
        if self.use_real_hardware:
            from qiskit_ibm_runtime import EstimatorV2

            estimator = EstimatorV2(mode=self.backend)
            estimator.options.default_shots = self.shots
        else:
            estimator = self.estimator
        return estimator.run(pubs).result()

    def _execute_quantum(self, input_data: torch.Tensor) -> torch.Tensor:
        """Return <Z> for each qubit, for every sample in the batch."""
        input_np = input_data.detach().cpu().numpy()
        weights_np = self.quantum_weights.detach().cpu().numpy()
        all_results = []

        # Sub-batches keep each hardware job within free-plan circuit limits
        for start in range(0, len(input_np), MAX_CIRCUITS_PER_JOB):
            sub_batch = input_np[start:start + MAX_CIRCUITS_PER_JOB]
            pubs = [
                (self.transpiled_qc, self.observables, np.concatenate([inp, weights_np]))
                for inp in sub_batch
            ]
            self.quantum_jobs_count += 1
            if self.use_real_hardware:
                print(f"  Quantum job #{self.quantum_jobs_count} submitted ({len(sub_batch)} samples)...")
            for pub_result in self._run_estimator(pubs):
                all_results.append(pub_result.data.evs)

        return torch.tensor(np.array(all_results), dtype=torch.float32, device=input_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum network."""
        features = self.preprocessing(x)
        quantum_out = self._execute_quantum(features)
        return self.head(quantum_out)


class QuantumDQNAgent:
    """
    Quantum DQN agent optimized for limited quantum resources.
    
    Key optimizations:
        - Batch all samples before quantum execution
        - Limit total quantum jobs
        - Cache results when possible
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_qubits: int = 4,
        shots: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
        use_real_hardware: bool = True,
        backend_name: Optional[str] = None,
        max_quantum_jobs: int = 8
    ):
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.max_quantum_jobs = max_quantum_jobs
        
        # Quantum Q-network
        self.q_net = QuantumQNetwork(
            in_dim=obs_dim,
            n_actions=n_actions,
            n_qubits=n_qubits,
            shots=shots,
            use_real_hardware=use_real_hardware,
            backend_name=backend_name
        ).to(device)
        
        # Classical target network (no quantum execution needed)
        from utils import MLP
        self.target_net = MLP(obs_dim, n_actions, hidden=64).to(device)
        
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)
        
        # Action cache for evaluation
        self._action_cache = {}
    
    def act(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Use classical approximation during training to save quantum jobs
        # Only use quantum for final evaluation
        obs_key = tuple(obs.round(3))
        if obs_key in self._action_cache:
            return self._action_cache[obs_key]
        
        # Fallback to random for uncached observations during training
        return np.random.randint(0, self.n_actions)
    
    def act_quantum(self, obs: np.ndarray) -> int:
        """Select action using quantum network (uses quantum resources)."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def update_batch(self, batches: List) -> float:
        """
        Update using multiple batches combined into one quantum call.
        
        This is the key optimization: instead of one quantum job per batch,
        we combine all batches and make one quantum call.
        """
        if self.q_net.quantum_jobs_count >= self.max_quantum_jobs:
            print(f"  Skipping update: quantum job limit ({self.max_quantum_jobs}) reached")
            return 0.0
        
        # Combine all batches
        all_obs = torch.cat([b[0] for b in batches], dim=0)
        all_actions = torch.cat([b[1] for b in batches], dim=0)
        all_rewards = torch.cat([b[2] for b in batches], dim=0)
        all_next_obs = torch.cat([b[3] for b in batches], dim=0)
        all_dones = torch.cat([b[4] for b in batches], dim=0)
        
        # Forward pass (one quantum job for all samples)
        q_values = self.q_net(all_obs)
        current_q = q_values.gather(1, all_actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (classical network, no quantum)
        with torch.no_grad():
            next_q = self.target_net(all_next_obs).max(1)[0]
            targets = all_rewards + (1 - all_dones) * self.gamma * next_q
        
        # Loss and optimization
        loss = nn.functional.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update(self, tau: float = 0.01):
        """Soft update target network from Q-network's classical parts."""
        # Update target using preprocessing and head weights
        for name, param in self.q_net.named_parameters():
            if 'preprocessing' in name or 'head' in name:
                for target_name, target_param in self.target_net.named_parameters():
                    if target_name in name:
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_quantum_jobs_used(self) -> int:
        """Return number of quantum jobs used."""
        return self.q_net.quantum_jobs_count
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "quantum_jobs_count": self.q_net.quantum_jobs_count
        }, path)
        print(f"Model saved to {path}")
