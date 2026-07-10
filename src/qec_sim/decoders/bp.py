import numpy as np
import stim
from ldpc import BpDecoder  # Capitalized for v2

class BPDecoder:
    def __init__(self, circuit: stim.Circuit, max_iter: int = 100):
        """
        Belief Propagation Decoder initialized directly from a Stim Circuit.
        """
        self.dem = circuit.detector_error_model(decompose_errors=True).flattened()
        self.H, self.error_rates, self.obs_matrix = self._extract_dem_matrices(self.dem)
        self.max_iter = max_iter
        
        # v2 uses BpDecoder and takes H as an initialization argument
        self.bpd = BpDecoder(
            self.H,
            max_iter=self.max_iter,
            bp_method="minimum_sum",
            channel_probs=list(self.error_rates)
        )
        
    def _extract_dem_matrices(self, dem: stim.DetectorErrorModel):
        num_detectors = dem.num_detectors
        num_errors = dem.num_errors
        num_observables = dem.num_observables

        H = np.zeros((num_detectors, num_errors), dtype=np.int8)
        obs_matrix = np.zeros((num_observables, num_errors), dtype=np.int8)
        error_rates = np.zeros(num_errors, dtype=float)

        error_idx = 0
        for instruction in dem:
            if instruction.type == "error":
                prob = instruction.args_copy()[0]
                error_rates[error_idx] = prob

                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        det_id = target.val
                        H[det_id, error_idx] ^= 1
                    elif target.is_logical_observable_id():
                        obs_id = target.val
                        obs_matrix[obs_id, error_idx] ^= 1
                error_idx += 1

        return H, error_rates, obs_matrix
        
    def decode(self, syndrome: np.ndarray):
        """
        Decodes a flat binary syndrome vector.
        """
        # Run decode
        decoding = self.bpd.decode(syndrome)
        
        # In ldpc v2, the convergence indicator status is tracked by 'converge'
        # 1 means converged successfully, 0 means it failed to converge
        converged = bool(self.bpd.converge == 1)
        
        return decoding, converged

    def get_log_likelihoods(self) -> np.ndarray:
        # Expose soft prob information
        return self.bpd.log_prob_ratios