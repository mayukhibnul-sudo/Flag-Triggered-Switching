import numpy as np
import stim
from qec_sim.codes import get_repetition_code_circuit, get_surface_code_circuit
from qec_sim.decoders.switcher import SwitchingDecoder

def evaluate_decoder(distance: int, rounds: int, physical_error_rate: float, num_shots: int = 1000):
    """
    Runs a Monte Carlo simulation loop for a given code configuration.
    """
    # 1. Generate the Stim Circuit (Let's use a Surface Code for realistic loop structures)
    circuit = get_surface_code_circuit(distance=distance, rounds=rounds, error_rate=physical_error_rate)
    
    # 2. Extract the blueprint to check final logical execution correctness
    dem = circuit.detector_error_model(decompose_errors=True)
    
    # 3. Initialize our Master Switching Decoder
    decoder = SwitchingDecoder(circuit, max_iter=30, osd_order=5)
    
    # 4. Compile a high-speed sampler from Stim
    sampler = circuit.compile_detector_sampler()
    
    # detector_shots: shape (num_shots, num_detectors)
    # actual_logical_flips: shape (num_shots, num_logical_observables)
    detector_shots, actual_logical_flips = sampler.sample(shots=num_shots, separate_observables=True)
    
    logical_errors_caught = 0
    
    print(f"--- Simulating d={distance}, r={rounds}, p={physical_error_rate} ({num_shots} shots) ---")
    
    # 5. Process each sampled shot through our decoding pipeline
    for shot in range(num_shots):
        syndrome = detector_shots[shot]
        true_flip = actual_logical_flips[shot]
        
        # Feed the syndrome to the switcher (BP runs first, drops to OSD if needed)
        predicted_errors, meta = decoder.decode(syndrome)
        
        # Map our predicted error vector to see if it implies a logical flip
        # Use the matrix we extracted inside the switcher's BP stage
        predicted_logical_flip = (decoder.bp_stage.obs_matrix @ predicted_errors) % 2
        
        # If our prediction doesn't match the actual logical flip, it's a decoding failure
        if not np.array_equal(predicted_logical_flip, true_flip):
            logical_errors_caught += 1

    # 6. Output simulation metrics
    logical_error_rate = logical_errors_caught / num_shots
    switching_ratio = decoder.get_switching_ratio()
    
    print(f"  Logical Error Rate : {logical_error_rate:.5f}")
    print(f"  OSD Fallback Rate  : {switching_ratio * 100:.2f}% (BP handled {100 - (switching_ratio*100):.2f}%)")
    print("-------------------------------------------------------\n")
    
    return logical_error_rate, switching_ratio

if __name__ == "__main__":
    # Run a quick diagnostic simulation
    # Distance 3, 3 rounds of measurement, 3% physical error rate, 200 shots
    evaluate_decoder(distance=3, rounds=3, physical_error_rate=0.03, num_shots=200)     