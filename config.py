"""
IBM Quantum Configuration Utility.
Saves credentials for IBM Quantum Runtime access (only needed for --real-hardware).

Reads IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE from the environment if set,
otherwise prompts for them.
"""
import os

from qiskit_ibm_runtime import QiskitRuntimeService


def setup_credentials(token: str = None, instance: str = None):
    """
    Save IBM Quantum credentials.
    
    Args:
        token: IBM Quantum API token
        instance: Instance CRN (for IBM Cloud)
    """
    token = token or os.getenv("IBM_QUANTUM_TOKEN")
    instance = instance or os.getenv("IBM_QUANTUM_INSTANCE")

    if not token:
        token = input("Enter your IBM Quantum API token: ").strip()

    if not instance:
        instance = input("Enter your Instance CRN (or press Enter to skip): ").strip() or None
    
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_cloud",
            token=token,
            instance=instance,
            overwrite=True
        )
        print("✅ Credentials saved successfully!")
        return True
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")
        return False


def verify_connection():
    """Verify connection to IBM Quantum and list available backends."""
    try:
        service = QiskitRuntimeService()
        
        # Get available backends
        backends = service.backends(operational=True, simulator=False)
        
        print("✅ Connected to IBM Quantum!")
        print(f"\nAvailable backends ({len(backends)}):")
        for backend in backends:
            status = backend.status()
            print(f"  • {backend.name}: {backend.num_qubits} qubits, {status.pending_jobs} pending jobs")
        
        # Get least busy
        least_busy = service.least_busy(operational=True, simulator=False)
        print(f"\n🎯 Recommended backend: {least_busy.name}")
        
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def get_backend_info(backend_name: str = None):
    """Get information about a specific backend or least busy one."""
    service = QiskitRuntimeService()
    
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)
    
    status = backend.status()
    
    return {
        "name": backend.name,
        "qubits": backend.num_qubits,
        "pending_jobs": status.pending_jobs,
        "operational": status.operational
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_connection()
    else:
        print("IBM Quantum Configuration")
        print("=" * 40)
        setup_credentials()
        print("\nVerifying connection...")
        verify_connection()
