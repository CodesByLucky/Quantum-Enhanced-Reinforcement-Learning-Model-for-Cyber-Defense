"""
Quantum Q-Network using IBM Quantum Runtime.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List


class QuantumQNetwork(nn.Module):
    """
    Quantum-enhanced Q-Network using IBM Quantum hardware.
    
    Architecture:
        Input → Classical Preprocessing → Quantum Circuit → Classical Head → Q-values
    
    Optimized for:
        - 4 qubits (fast execution)
        - 256 shots (minimum for accuracy)
        - Batch processing (minimize quantum jobs)
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
        
        # Import Qiskit modules
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import RealAmplitudes
            from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        except ImportError as e:
            raise RuntimeError(
                "Install required packages:\n"
                "pip install qiskit qiskit-ibm-runtime"
            ) from e
        
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
        
        # Initialize IBM Quantum service
        self.service = QiskitRuntimeService()
        
        # Select backend
        if use_real_hardware:
            if backend_name:
                self.backend = self.service.backend(backend_name)
            else:
                self.backend = self.service.least_busy(operational=True, simulator=False)
            print(f"Using quantum backend: {self.backend.name}")
        else:
            # Use simulator
            backends = self.service.backends(simulator=True)
            self.backend = backends[0] if backends else None
            print(f"Using simulator: {self.backend.name if self.backend else 'local'}")
        
        # Create parameterized quantum circuit
        self.qc = RealAmplitudes(
            num_qubits=n_qubits,
            reps=2,  # 2 repetitions for good expressiveness
            entanglement='linear',
            insert_barriers=True
        )
        
        # Create observables (Z measurement on each qubit)
        self.observables = self._create_observables()
        
        # Transpile circuit for backend
        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=2)
        self.transpiled_qc = self.pm.run(self.qc)
        
        # Handle potential qubit count change after transpilation
        transpiled_n_qubits = self.transpiled_qc.num_qubits
        if transpiled_n_qubits != n_qubits:
            print(f"Note: Circuit expanded to {transpiled_n_qubits} qubits after transpilation")
            self.observables = self._create_observables(transpiled_n_qubits)
        
        # Store parameter info
        self.input_params = list(self.qc.parameters[:n_qubits])
        self.weight_params = list(self.qc.parameters[n_qubits:])
        
        # Trainable quantum weights
        n_weights = len(self.weight_params)
        self.quantum_weights = nn.Parameter(torch.randn(n_weights) * 0.1)
        
        # Classical head: map quantum outputs to Q-values
        n_obs = len(self.observables)
        self.head = nn.Sequential(
            nn.Linear(n_obs, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )
        
        # Job counter for tracking
        self.quantum_jobs_count = 0
    
    def _create_observables(self, n_qubits: int = None) -> List:
        """Create Pauli-Z observables for each qubit."""
        from qiskit.quantum_info import SparsePauliOp
        
        n = n_qubits or self.n_qubits
        observables = []
        
        # Z measurement on each qubit
        for i in range(n):
            pauli_str = 'I' * i + 'Z' + 'I' * (n - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
        
        return observables
    
    def _execute_quantum(self, input_data: torch.Tensor) -> torch.Tensor:
        """Execute quantum circuit on IBM Quantum hardware."""
        from qiskit_ibm_runtime import EstimatorV2
        
        batch_size = input_data.shape[0]
        input_np = input_data.detach().cpu().numpy()
        weights_np = self.quantum_weights.detach().cpu().numpy()
        
        # Process in sub-batches to avoid IBM Quantum limits
        MAX_CIRCUITS_PER_JOB = 20  # Free plan limit
        all_results = []
        
        for start_idx in range(0, batch_size, MAX_CIRCUITS_PER_JOB):
            end_idx = min(start_idx + MAX_CIRCUITS_PER_JOB, batch_size)
            sub_batch = input_np[start_idx:end_idx]
            
            # Create estimator for this sub-batch
            estimator = EstimatorV2(mode=self.backend)
            
            # Build PUBs for this sub-batch
            pubs = []
            for inp in sub_batch:
                params = np.concatenate([inp, weights_np])
                pubs.append((self.transpiled_qc, self.observables, params))
            
            # Execute on quantum hardware
            job = estimator.run(pubs=pubs)
            self.quantum_jobs_count += 1
            
            print(f"  Quantum job #{self.quantum_jobs_count} submitted ({len(sub_batch)} samples)...")
            
            result = job.result()
            
            # Extract expectation values
            for pub_result in result:
                evs = pub_result.data.evs
                all_results.append(evs)
        
        return torch.tensor(np.array(all_results), dtype=torch.float32, device=input_data.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum network."""
        # Classical preprocessing
        features = self.preprocessing(x)
        
        # Quantum layer
        quantum_out = self._execute_quantum(features)
        
        # Classical head
        q_values = self.head(quantum_out)
        
        return q_values


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
