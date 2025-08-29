import numpy as np

class RealTimePitchogram:
    def __init__(self, num_channels=71, sai_width=400, future_lags=100, num_triggers=4):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.future_lags = future_lags
        self.num_triggers = num_triggers
        
        self.mask = np.ones((num_channels, sai_width))
        self.output_buffer = np.zeros((num_channels, sai_width))
        
        # smoothing for continuous frames
        self.cgram = np.zeros(num_channels)
        self.vowel_matrix = None  # optional for F1-F2 embedding

    def set_vowel_matrix(self, vowel_matrix):
        self.vowel_matrix = vowel_matrix

    def apply_mask(self, sai_frame):
        return sai_frame * self.mask

    def run_frame(self, sai_frame):
        """
        sai_frame: shape (num_channels, sai_width)
        returns: pitchogram output of same shape
        """
        masked = self.apply_mask(sai_frame)
        
        # simple frame averaging for demo (replace with proper lag correlations if needed)
        self.output_buffer = masked.mean(axis=1, keepdims=True) * np.ones_like(masked)
        
        # optional vowel embedding
        if self.vowel_matrix is not None:
            self.cgram = 0.2 * masked.mean(axis=1) + 0.8 * self.cgram
            vowel_coords = self.vowel_matrix @ self.cgram
            # You could map vowel_coords to color later

        return self.output_buffer.copy()
