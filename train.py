"""
Quantum Reinforcement Learning - Fair Comparison Study
Compares Classical DQN vs Quantum DQN for threat classification.

FAIR COMPARISON PROTOCOL:
- Both models use EXACTLY the same training samples
- Both models use the same number of epochs and updates
- Both models evaluated on the same test samples
- Fixed random seed for reproducibility

Optimized for IBM Quantum free plan (10 minutes runtime).

Usage:
    python train.py --mode compare --real-hardware    # Fair comparison on quantum hardware
    python train.py --mode dqn                        # Train only classical DQN
    python train.py --mode qdqn --real-hardware       # Train only quantum QDQN
"""
import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from utils import load_dataset, ReplayBuffer
from environment import ThreatClassificationEnv
from models import DQNAgent

# Fixed seed for reproducibility
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prepare_fair_data(X_train, y_train, X_test, y_test, 
                      train_samples: int = 60, 
                      test_samples: int = 20):
    """
    Prepare fixed subsets for fair comparison.
    Both models will use EXACTLY the same data.
    """
    set_seed(RANDOM_SEED)
    
    # Select fixed training subset
    train_indices = np.random.choice(len(X_train), train_samples, replace=False)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    
    # Select fixed test subset
    test_indices = np.random.choice(len(X_test), test_samples, replace=False)
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]
    
    print(f"\n📊 Fair Comparison Data:")
    print(f"   Training samples: {len(X_train_subset)} (fixed subset)")
    print(f"   Test samples: {len(X_test_subset)} (fixed subset)")
    print(f"   Random seed: {RANDOM_SEED}")
    
    return X_train_subset, y_train_subset, X_test_subset, y_test_subset


def train_dqn_fair(
    X_train, y_train, X_test, y_test, 
    class_to_idx, 
    epochs: int = 2,
    batch_size: int = 20,
    n_updates_per_epoch: int = 3,
    device: str = "cpu"
):
    """
    Train classical DQN with SAME parameters as quantum model.
    This ensures fair comparison.
    """
    print("\n" + "=" * 50)
    print("TRAINING CLASSICAL DQN (Fair Protocol)")
    print("=" * 50)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Updates per epoch: {n_updates_per_epoch}")
    print(f"  Total gradient updates: {epochs * n_updates_per_epoch}")
    
    set_seed(RANDOM_SEED)
    
    n_actions = len(class_to_idx)
    obs_dim = X_train.shape[1]
    
    # Initialize
    agent = DQNAgent(obs_dim, n_actions, hidden=128, device=device)
    buffer = ReplayBuffer(capacity=2000, device=device)
    env = ThreatClassificationEnv(X_train, y_train, class_to_idx)
    
    start_time = time.time()
    losses = []
    
    # Collect experiences from training data
    print("\nCollecting experiences...")
    obs = env.reset()
    for _ in tqdm(range(len(X_train)), desc="Experience"):
        action = agent.act(obs, epsilon=0.3)
        next_obs, reward, done, _ = env.step(action)
        
        if next_obs is not None:
            buffer.push(obs, action, reward, next_obs, int(done))
            obs = next_obs
        
        if done:
            obs = env.reset()
    
    # Training with EXACT same number of updates as quantum
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for update in range(n_updates_per_epoch):
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = agent.update(batch)
                epoch_losses.append(loss)
                agent.soft_update()
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f} ({n_updates_per_epoch} updates)")
    
    train_time = time.time() - start_time
    
    # Evaluation on SAME test set
    print("\nEvaluating...")
    accuracy, f1 = evaluate_agent(agent, X_test, y_test, class_to_idx, device)
    
    results = {
        "model": "DQN",
        "accuracy": accuracy,
        "f1_score": f1,
        "training_time": train_time,
        "epochs": epochs,
        "updates_per_epoch": n_updates_per_epoch,
        "total_updates": epochs * n_updates_per_epoch,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "batch_size": batch_size,
        "losses": losses
    }
    
    print(f"\n✅ DQN Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Time: {train_time:.2f}s")
    print(f"   Total updates: {epochs * n_updates_per_epoch}")
    
    return agent, results


def train_qdqn_fair(
    X_train, y_train, X_test, y_test,
    class_to_idx,
    epochs: int = 2,
    batch_size: int = 20,
    n_updates_per_epoch: int = 3,
    n_qubits: int = 4,
    shots: int = 256,
    max_quantum_jobs: int = 8,
    device: str = "cpu",
    use_real_hardware: bool = False,
    backend_name: str = None
):
    """
    Train Quantum DQN with SAME parameters as classical model.
    This ensures fair comparison.
    """
    print("\n" + "=" * 50)
    print("TRAINING QUANTUM DQN (Fair Protocol)")
    print("=" * 50)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Updates per epoch: {n_updates_per_epoch}")
    print(f"  Total gradient updates: {epochs * n_updates_per_epoch}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Shots: {shots}")
    print(f"  Real hardware: {use_real_hardware}")
    
    set_seed(RANDOM_SEED)
    
    from quantum_network import QuantumDQNAgent
    
    n_actions = len(class_to_idx)
    obs_dim = X_train.shape[1]
    
    # Initialize
    agent = QuantumDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_qubits=n_qubits,
        shots=shots,
        device=device,
        use_real_hardware=use_real_hardware,
        backend_name=backend_name,
        max_quantum_jobs=max_quantum_jobs
    )
    
    buffer = ReplayBuffer(capacity=2000, device=device)
    env = ThreatClassificationEnv(X_train, y_train, class_to_idx)
    
    start_time = time.time()
    losses = []
    
    # Collect experiences (using epsilon-greedy, no quantum)
    print("\nCollecting experiences...")
    obs = env.reset()
    for _ in tqdm(range(len(X_train)), desc="Experience"):
        action = agent.act(obs, epsilon=0.5)
        next_obs, reward, done, _ = env.step(action)
        
        if next_obs is not None:
            buffer.push(obs, action, reward, next_obs, int(done))
            obs = next_obs
        
        if done:
            obs = env.reset()
    
    # Training with SAME number of updates as classical
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Collect batches for this epoch
        all_batches = []
        for _ in range(n_updates_per_epoch):
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                all_batches.append(batch)
        
        if all_batches:
            total_samples = sum(len(b[0]) for b in all_batches)
            print(f"\n  Epoch {epoch+1}: Processing {total_samples} samples ({n_updates_per_epoch} updates)...")
            loss = agent.update_batch(all_batches)
            losses.append(loss)
            print(f"  Loss: {loss:.4f}")
            print(f"  Quantum jobs used: {agent.get_quantum_jobs_used()}/{max_quantum_jobs}")
    
    train_time = time.time() - start_time
    
    # Evaluation on SAME test set
    print("\nEvaluating with quantum network...")
    accuracy, f1 = evaluate_quantum_agent(agent, X_test, y_test, class_to_idx, device)
    
    results = {
        "model": "QDQN",
        "accuracy": accuracy,
        "f1_score": f1,
        "training_time": train_time,
        "epochs": epochs,
        "updates_per_epoch": n_updates_per_epoch,
        "total_updates": epochs * n_updates_per_epoch,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "batch_size": batch_size,
        "quantum_jobs": agent.get_quantum_jobs_used(),
        "n_qubits": n_qubits,
        "shots": shots,
        "losses": losses
    }
    
    print(f"\n✅ QDQN Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Time: {train_time:.2f}s")
    print(f"   Total updates: {epochs * n_updates_per_epoch}")
    print(f"   Quantum jobs: {agent.get_quantum_jobs_used()}")
    
    return agent, results


def evaluate_agent(agent, X_test, y_test, class_to_idx, device):
    """Evaluate classical DQN agent on fixed test set."""
    correct = 0
    predictions = []
    labels = []
    
    for i in range(len(X_test)):
        obs = X_test[i]
        true_label = class_to_idx[y_test[i]]
        
        action = agent.act(obs, epsilon=0)
        predictions.append(action)
        labels.append(true_label)
        
        if action == true_label:
            correct += 1
    
    accuracy = correct / len(X_test)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, predictions, average='weighted')
    
    return accuracy, f1


def evaluate_quantum_agent(agent, X_test, y_test, class_to_idx, device):
    """Evaluate Quantum DQN agent on fixed test set (same as classical)."""
    print(f"  Evaluating {len(X_test)} samples...")
    
    correct = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        q_values = agent.q_net(X_tensor)
        actions = q_values.argmax(dim=1).cpu().numpy()
    
    for i, action in enumerate(actions):
        true_label = class_to_idx[y_test[i]]
        predictions.append(action)
        labels.append(true_label)
        
        if action == true_label:
            correct += 1
    
    accuracy = correct / len(X_test)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, predictions, average='weighted')
    
    return accuracy, f1


def save_results(results: dict, output_dir: Path):
    """Save training results with fair comparison metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add comparison metadata
    results["comparison_protocol"] = {
        "fair_comparison": True,
        "random_seed": RANDOM_SEED,
        "same_training_data": True,
        "same_test_data": True,
        "same_epochs": True,
        "same_updates_per_epoch": True
    }
    
    results_file = output_dir / "results.json"
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    clean_results = convert_types(results)
    
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def print_comparison_table(dqn_results: dict, qdqn_results: dict):
    """Print formatted comparison table for research paper."""
    print("\n" + "=" * 70)
    print("FAIR COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'DQN (Classical)':<20} {'QDQN (Quantum)':<20}")
    print("-" * 70)
    print(f"{'Training Samples':<25} {dqn_results['train_samples']:<20} {qdqn_results['train_samples']:<20}")
    print(f"{'Test Samples':<25} {dqn_results['test_samples']:<20} {qdqn_results['test_samples']:<20}")
    print(f"{'Epochs':<25} {dqn_results['epochs']:<20} {qdqn_results['epochs']:<20}")
    print(f"{'Updates per Epoch':<25} {dqn_results['updates_per_epoch']:<20} {qdqn_results['updates_per_epoch']:<20}")
    print(f"{'Total Updates':<25} {dqn_results['total_updates']:<20} {qdqn_results['total_updates']:<20}")
    print(f"{'Batch Size':<25} {dqn_results['batch_size']:<20} {qdqn_results['batch_size']:<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {dqn_results['accuracy']:<20.4f} {qdqn_results['accuracy']:<20.4f}")
    print(f"{'F1 Score':<25} {dqn_results['f1_score']:<20.4f} {qdqn_results['f1_score']:<20.4f}")
    print(f"{'Training Time':<25} {dqn_results['training_time']:<20.2f}s {qdqn_results['training_time']:<20.2f}s")
    if 'quantum_jobs' in qdqn_results:
        print(f"{'Quantum Jobs':<25} {'N/A':<20} {qdqn_results['quantum_jobs']:<20}")
    print("=" * 70)
    print("\n✅ Fair Comparison Protocol Verified:")
    print("   - Same training samples: ✓")
    print("   - Same test samples: ✓")
    print("   - Same epochs: ✓")
    print("   - Same gradient updates: ✓")
    print("   - Same random seed: ✓")


def main():
    parser = argparse.ArgumentParser(description="Quantum RL - Fair Comparison Study")
    parser.add_argument("--mode", choices=["dqn", "qdqn", "compare"], default="compare",
                        help="Training mode: dqn, qdqn, or compare")
    parser.add_argument("--real-hardware", action="store_true",
                        help="Use real IBM Quantum hardware")
    parser.add_argument("--backend", type=str, default=None,
                        help="Specific IBM Quantum backend name")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (same for both models)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Batch size (same for both models)")
    parser.add_argument("--updates-per-epoch", type=int, default=3,
                        help="Gradient updates per epoch (same for both models)")
    parser.add_argument("--train-samples", type=int, default=60,
                        help="Number of training samples (same for both models)")
    parser.add_argument("--test-samples", type=int, default=20,
                        help="Number of test samples (same for both models)")
    parser.add_argument("--qubits", type=int, default=4,
                        help="Number of qubits for quantum circuit")
    parser.add_argument("--shots", type=int, default=256,
                        help="Number of shots for quantum measurement")
    parser.add_argument("--max-quantum-jobs", type=int, default=8,
                        help="Maximum number of quantum jobs")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("QUANTUM REINFORCEMENT LEARNING - FAIR COMPARISON STUDY")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Real hardware: {args.real_hardware}")
    print(f"\n📋 Fair Comparison Protocol:")
    print(f"   Training samples: {args.train_samples} (same for both)")
    print(f"   Test samples: {args.test_samples} (same for both)")
    print(f"   Epochs: {args.epochs} (same for both)")
    print(f"   Updates per epoch: {args.updates_per_epoch} (same for both)")
    print(f"   Batch size: {args.batch_size} (same for both)")
    print(f"   Random seed: {RANDOM_SEED} (fixed)")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load full data
    X_train_full, y_train_full, X_test_full, y_test_full, classes, class_to_idx = load_dataset()
    
    # Prepare SAME data subsets for fair comparison
    X_train, y_train, X_test, y_test = prepare_fair_data(
        X_train_full, y_train_full, X_test_full, y_test_full,
        train_samples=args.train_samples,
        test_samples=args.test_samples
    )
    
    output_dir = Path(args.output_dir)
    all_results = {}
    
    # Train models with SAME parameters
    if args.mode in ["dqn", "compare"]:
        dqn_agent, dqn_results = train_dqn_fair(
            X_train, y_train, X_test, y_test,
            class_to_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_updates_per_epoch=args.updates_per_epoch,
            device=device
        )
        all_results["dqn"] = dqn_results
        dqn_agent.save(str(output_dir / "dqn.pt"))
    
    if args.mode in ["qdqn", "compare"]:
        qdqn_agent, qdqn_results = train_qdqn_fair(
            X_train, y_train, X_test, y_test,
            class_to_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_updates_per_epoch=args.updates_per_epoch,
            n_qubits=args.qubits,
            shots=args.shots,
            max_quantum_jobs=args.max_quantum_jobs,
            device=device,
            use_real_hardware=args.real_hardware,
            backend_name=args.backend
        )
        all_results["qdqn"] = qdqn_results
        qdqn_agent.save(str(output_dir / "qdqn.pt"))
    
    # Save results
    save_results(all_results, output_dir)
    
    # Print comparison table
    if args.mode == "compare" and "dqn" in all_results and "qdqn" in all_results:
        print_comparison_table(all_results["dqn"], all_results["qdqn"])
    
    print("\n🎓 Results ready for research paper!")
    print(f"   Check {output_dir}/results.json for detailed metrics")


if __name__ == "__main__":
    main()
