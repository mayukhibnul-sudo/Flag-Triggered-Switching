import stim
import scipy.sparse as sp
import numpy as np

def get_repetition_code_circuit(distance: int, rounds: int, error_rate: float) -> stim.Circuit:
    """
    Generates a fault-tolerant 1D repetition code circuit.
    
    Parameters:
    -----------
    distance : int
        The code distance (number of data qubits).
    rounds : int
        Number of noisy syndrome measurement rounds to simulate.
    error_rate : float
        The probability of physical faults (applied to gates, resets, and measurements).
    """
    return stim.Circuit.generated(
        "repetition_code:test",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate
    )

def get_surface_code_circuit(distance: int, rounds: int, error_rate: float) -> stim.Circuit:
    """
    Generates a fault-tolerant 2D rotated surface code circuit (X-basis/Z-basis memory).
    This provides a highly realistic, loop-heavy check matrix for your BP+OSD decoder.
    
    Parameters:
    -----------
    distance : int
        The code distance (d x d grid of data qubits).
    rounds : int
        Number of noisy syndrome measurement rounds.
    error_rate : float
        The uniform physical error rate for noise insertion.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate
    )




def get_cyclic_shift_matrix(l: int) -> sp.csr_matrix:
    """Generates a basic l x l cyclic shift matrix (permutation matrix)."""
    data = np.ones(l)
    rows = np.arange(l)
    cols = (np.arange(l) + 1) % l
    return sp.csr_matrix((data, (rows, cols)), shape=(l, l), dtype=np.uint8)

def make_generalized_bicycle_code(l: int, a_powers: list, b_powers: list):
    """
    Constructs a Generalized Bicycle (GB) code from two polynomials A and B.
    Defined by shift powers, e.g., A = S^0 + S^1 + S^3
    """
    S = get_cyclic_shift_matrix(l)
    
    # Construct classical component matrices A and B
    A = sp.csr_matrix((l, l), dtype=np.uint8)
    for p in a_powers:
        A = (A + (S ** p)) % 2
        
    B = sp.csr_matrix((l, l), dtype=np.uint8)
    for p in b_powers:
        B = (B + (S ** p)) % 2

    # CSS construction for Generalized Bicycle codes
    # Block matrix: H_X = [A, B], H_Z = [B.T, A.T]
    HX = sp.hstack([A, B], format="csr")
    HZ = sp.hstack([B.T, A.T], format="csr")
    
    return HX, HZ

def make_bivariate_bicycle_144_12_12():
    """
    Constructs the canonical Bravyi et al. [[144, 12, 12]] Bivariate Bicycle Code.
    Uses bivariate polynomials over l=12, m=6 (n = 2 * l * m = 144).
    """
    l, m = 12, 6
    N_block = l * m
    
    # Define independent shift matrices for the 2D torus grid
    X_shift = get_cyclic_shift_matrix(l)
    Y_shift = get_cyclic_shift_matrix(m)
    
    # Kronecker products map shifts to the full l*m bivariate space
    I_l = sp.eye(l, dtype=np.uint8)
    I_m = sp.eye(m, dtype=np.uint8)
    
    x = sp.kron(X_shift, I_m, format="csr")
    y = sp.kron(I_l, Y_shift, format="csr")
    
    # Canonical Bravyi Polynomials:
    # A = x^3 + y^1 + y^2
    # B = y^3 + x^1 + x^2
    A = (x**3 + y**1 + y**2)
    A.data %= 2
    B = (y**3 + x**1 + x**2)
    B.data %= 2
    
    # CSS assembly
    HX = sp.hstack([A, B], format="csr")
    HZ = sp.hstack([B.T, A.T], format="csr")
    
    return HX, HZ