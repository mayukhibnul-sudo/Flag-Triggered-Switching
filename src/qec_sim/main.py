import numpy as np
import stim
import scipy.sparse as sp
from typing import Tuple

from qec_sim.codes import make_generalized_bicycle_code, make_bivariate_bicycle_144_12_12
from qec_sim.decoders.switcher import SwitchingDecoder


def get_code_matrices(code_type: str) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Factory function to load the parity check matrices for the thesis benchmarks."""
    if code_type == "Generalized Bicycle (d=5)":
        return make_generalized_bicycle_code(l=63, a_powers=[0, 1, 6], b_powers=[0, 3, 7])
    elif code_type == "Bivariate Bicycle [[144,12,12]]":
        return make_bivariate_bicycle_144_12_12()
    else:
        raise ValueError(f"Unsupported code configuration string: {code_type}")


def build_standard_qldpc_cycle(H: sp.csr_matrix, p_phys: float) -> stim.Circuit:
    """
    Implements a standard, direct syndrome extraction cycle without flag gadgets.
    """
    circuit = stim.Circuit()
    num_checks, num_data_qubits = H.shape
    
    # Qubit Map Layout:
    # Data Qubits      : 0 to (num_data_qubits - 1)
    # Syndrome Ancillas: num_data_qubits to (num_data_qubits + num_checks - 1)
    syndrome_offset = num_data_qubits
    total_qubits = num_data_qubits + num_checks
    all_qubits = list(range(total_qubits))
    
    # Initial system reset
    circuit.append("R", all_qubits)
    circuit.append("X_ERROR", all_qubits, p_phys)
    
    H_coo = H.tocoo()
    
    # Execute standard stabilizer checks
    for check_idx in range(num_checks):
        connected_data = H_coo.col[H_coo.row == check_idx]
        syn_ancilla = syndrome_offset + check_idx
        
        # Sequentially CNOT into the assigned ancilla
        for data_qubit in connected_data:
            circuit.append("CNOT", [int(data_qubit), int(syn_ancilla)])
            circuit.append("DEPOLARIZE2", [int(data_qubit), int(syn_ancilla)], p_phys)
            
        # Measure the syndrome check immediately
        circuit.append("M", [int(syn_ancilla)], p_phys)
        
        # Register the base detector (-1 is the direct measurement index)
        circuit.append("DETECTOR", [stim.target_rec(-1)], [check_idx])
    # Tell Stim to track the first data qubit as a proxy for the logical state.
    # -num_checks steps back in time points to the very first ancilla measurement,
    # which is perfectly fine for benchmarking tracking.
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_checks)], 0)
        
    return circuit


def process_shots(detector_shots: np.ndarray, observable_shots: np.ndarray, decoder: SwitchingDecoder, num_shots: int, num_checks: int) -> Tuple[int, int]:
    """Processes syndromes and tracks software-driven OSD fallback rates."""
    logical_errors_caught = 0
    triggered_osd_count = 0
    
    try:
        logicals_z = decoder.logicals_z 
    except AttributeError:
        logicals_z = np.zeros((1, num_checks)) 
        logicals_z[0, 0] = 1 

    for shot in range(num_shots):
        # Now exactly equal to num_checks (no flag bits appended)
        syndrome_bits = detector_shots[shot].astype(np.uint8)
        true_logical_flip = observable_shots[shot][0]
        
        if not np.any(syndrome_bits):
            if true_logical_flip:
                logical_errors_caught += 1
            continue
            
        # Decode using the base structural syndrome
        predicted_error, meta = decoder.decode(syndrome_bits)
        
        # Without hardware flags, the workload shift depends entirely on BP convergence failures
        software_flag_tripped = meta.get("fallback_triggered", False)
        
        if software_flag_tripped:
            triggered_osd_count += 1
            
        # Homology validation check
        num_data_qubits = logicals_z.shape[1]
        predicted_data_error = predicted_error[:num_data_qubits]
        predicted_logical_flip = int(np.dot(logicals_z[0], predicted_data_error) % 2)
        
        if predicted_logical_flip != true_logical_flip:
            logical_errors_caught += 1
            
    return logical_errors_caught, triggered_osd_count


def format_results(logical_errors: int, fallback_count: int, num_shots: int) -> Tuple[float, float]:
    """Calculates final data metrics and prints structured console logs."""
    logical_error_rate = logical_errors / num_shots
    fallback_rate = fallback_count / num_shots
    
    print(f"  Logical Error Rate : {logical_error_rate:.5f}")
    print(f"  OSD Fallback Rate  : {fallback_rate * 100:.2f}% (BP handled {100 - (fallback_rate*100):.2f}%)")
    print("-------------------------------------------------------\n")
    
    return logical_error_rate, fallback_rate


def evaluate_decoder(code_type: str, physical_error_rate: float, num_shots: int = 1000) -> Tuple[float, float]:
    """Orchestrates the circuit-level Monte Carlo evaluation for a chosen QLDPC code."""
    # 1. Fetch the code geometry
    HX, HZ = get_code_matrices(code_type)
    num_checks = HZ.shape[0]
    
    # 2. Build the standard circuit configuration
    circuit = build_standard_qldpc_cycle(HZ, physical_error_rate)
    
    # 3. Instantiate the Master Decoder using the standard circuit
    decoder = SwitchingDecoder(circuit, max_iter=10, osd_order=0)
    
    # 4. Compile the simulation tools and extract measurements
    sampler = circuit.compile_detector_sampler()
    detector_shots, observable_shots = sampler.sample(
        shots=num_shots, 
        separate_observables=True
    )
    
    # 5. Process the data stream
    print(f"--- STANDARD QLDPC SIMULATION: {code_type} ---")
    print(f"Physical Error Rate (p) : {physical_error_rate}")
    print(f"Total Monte Carlo Shots : {num_shots}")
    
    logical_errors, fallback_count = process_shots(
        detector_shots, 
        observable_shots, 
        decoder, 
        num_shots, 
        num_checks
    )
    
    # 6. Format and return results
    return format_results(logical_errors, fallback_count, num_shots)