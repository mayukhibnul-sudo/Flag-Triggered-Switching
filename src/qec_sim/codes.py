import stim

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