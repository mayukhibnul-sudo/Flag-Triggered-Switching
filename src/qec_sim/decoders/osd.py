import numpy as np
from ldpc import BpOsdDecoder  # Capitalized for v2

class OSDDecoder:
    def __init__(self, parity_check_matrix: np.ndarray, error_rates: list, max_iter: int = 100, osd_order: int = 10):
        """
        Ordered Statistics Decoding (OSD) Companion Decoder.
        """
        self.H = parity_check_matrix
        self.error_rates = error_rates
        self.osd_order = osd_order
        
        # Initialize modern BpOsdDecoder wrapper
        self.bposd = BpOsdDecoder(
            self.H,
            max_iter=max_iter,
            bp_method="minimum_sum",
            channel_probs=self.error_rates,
            osd_method="osd_cs",
            osd_order=self.osd_order
        )

    def decode_with_soft_hints(self, syndrome: np.ndarray):
        """
        Runs the full OSD post-processing pass using the underlying matrix layout.
        """
        decoding = self.bposd.decode(syndrome)
        return decoding