import numpy as np
from qec_sim.decoders.bp import BPDecoder
from qec_sim.decoders.osd import OSDDecoder

class SwitchingDecoder:
    def __init__(self, circuit, max_iter: int = 100, osd_order: int = 10):
        """
        The Master Switching Decoder that coordinates BP and OSD.
        """
        # 1. Initialize the BP Decoder directly using the Stim circuit
        self.bp_stage = BPDecoder(circuit, max_iter=max_iter)
        
        # 2. Extract the parsed matrices from the BP stage to feed into the OSD stage
        # This ensures both stages are looking at the exact same space-time check matrix
        self.osd_stage = OSDDecoder(
            parity_check_matrix=self.bp_stage.H,
            error_rates=self.bp_stage.error_rates,
            max_iter=max_iter,
            osd_order=osd_order
        )
        
        # Metrics to track switching statistics
        self.total_decodes = 0
        self.bp_success_count = 0
        self.osd_fallback_count = 0

    def decode(self, syndrome: np.ndarray):
        """
        Decodes a syndrome by attempting BP first, and falling back to OSD if necessary.
        
        Returns:
        --------
        final_prediction : np.ndarray
            The binary error correction vector.
        meta_data : dict
            Diagnostic metrics containing which decoder was used.
        """
        self.total_decodes += 1
        
        # Step 1: Attempt low-overhead Belief Propagation
        bp_prediction, bp_converged = self.bp_stage.decode(syndrome)
        
        if bp_converged:
            self.bp_success_count += 1
            return bp_prediction, {"decoder_used": "BP", "switched": False}
        
        # Step 2: If BP stalls, log the failure and trigger the OSD fallback
        self.osd_fallback_count += 1
        
        # OSD uses its internal matrix solver to guarantee a valid correction match
        osd_prediction = self.osd_stage.decode_with_soft_hints(syndrome)
        
        return osd_prediction, {"decoder_used": "OSD", "switched": True}

    def get_switching_ratio(self) -> float:
        """Returns the fraction of times the decoder had to fall back to OSD."""
        if self.total_decodes == 0:
            return 0.0
        return self.osd_fallback_count / self.total_decodes