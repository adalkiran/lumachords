from dataclasses import dataclass
import cv2
import numpy as np

from lumachords.preferences import Preferences
from lumachords.utils import Utils

@dataclass
class FFTAnalyzerOutput:
    im_transformed: np.ndarray
    dominant_freqs: np.ndarray
    normalized_freqs: np.ndarray
    normalized_peakiness: np.ndarray
    fft_magnitudes_no_dc: np.ndarray
    frequencies: np.ndarray
    dominant_freq_indices: np.ndarray
    row_peakiness_ratios: np.ndarray

    num_rows: int

class FFTAnalyzer:
    def __init__(self, pref: Preferences):
        self.pref = pref
    
    def perform_fft_rows(self, im_gray, scharr=True, normalize_and_center=True, return_image=False):
        # --- Preprocessing ---        
        if scharr:
            im_gradient_x = cv2.Scharr(im_gray, cv2.CV_32F, 1, 0)
            im_gradient_x = np.abs(im_gradient_x)
            if normalize_and_center:
                im_gradient_x = im_gradient_x / (im_gradient_x.mean(axis=1, keepdims=True) + 1e-6)
                im_transformed = im_gradient_x - im_gradient_x.mean(axis=1)[:, None]
            else:
                im_transformed = im_gradient_x
        else:
            im_transformed = im_gray

        num_rows, num_cols = im_transformed.shape

        # --- Row-wise FFT ---
        num_freq_bins = num_cols // 2 + 1
        fft_magnitudes = np.empty((num_rows, num_freq_bins), dtype=np.float32)
        for r in range(num_rows):
            fft_spectrum = np.fft.rfft(im_transformed[r, :])
            fft_magnitudes[r] = np.abs(fft_spectrum)

        if self.pref.engine.keybed_max_fft_bins is not None:
            num_freq_bins = min(num_freq_bins, self.pref.engine.keybed_max_fft_bins)
            fft_magnitudes = fft_magnitudes[:, :num_freq_bins]

        frequencies = np.fft.rfftfreq(num_cols, d=1.0)[:fft_magnitudes.shape[1]]

        # --- Dominant frequency & peakiness per row ---
        fft_magnitudes_no_dc = fft_magnitudes.copy()
        fft_magnitudes_no_dc[:, 0] = 0  # ignore DC

        dominant_freq_indices = fft_magnitudes_no_dc.argmax(axis=1)
        dominant_freqs = frequencies[dominant_freq_indices]        # cycles/pixel
        row_peak_magnitudes = fft_magnitudes_no_dc[np.arange(num_rows), dominant_freq_indices]
        row_energy_sums = fft_magnitudes_no_dc.sum(axis=1) + 1e-9
        row_peakiness_ratios = (row_peak_magnitudes / row_energy_sums).astype(np.float32)

        # --- Feature matrix for clustering ---
        normalized_freqs = Utils.normalize(dominant_freqs)
        normalized_peakiness = Utils.normalize(row_peakiness_ratios)

        if im_transformed.max() < 10 or (im_transformed.min() < 1 and im_transformed.max() > 1):
            im_transformed = cv2.normalize(im_transformed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return FFTAnalyzerOutput(
            im_transformed=im_transformed if return_image else None,
            dominant_freqs=dominant_freqs,
            normalized_freqs=normalized_freqs,
            normalized_peakiness=normalized_peakiness,
            fft_magnitudes_no_dc=fft_magnitudes_no_dc,
            frequencies=frequencies,
            dominant_freq_indices=dominant_freq_indices,
            row_peakiness_ratios=row_peakiness_ratios,
            num_rows=num_rows,
        )
