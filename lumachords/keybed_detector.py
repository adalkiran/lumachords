from dataclasses import dataclass
import json
import typing
import cv2
import numpy as np
from collections import OrderedDict

from lumachords.processing_state import ProcessingState
from lumachords.runtime_config import AppMode, LogLevel, ProdMode, RuntimeConfig
from lumachords.fft_analyzer import FFTAnalyzer
from lumachords.image_input import ImagePreprocessor, KeybedImageInput, KeybedInternalImageInput
from lumachords.preferences import Preferences
from lumachords.rendering import KeybedRenderer
from lumachords.utils import DetectionException, Utils
from lumachords.artifact_sink import ArtifactConfigEntry, ArtifactSink

@dataclass
class KeybedDetectorOutput:
    keybed_bounds: 'typing.Any'
    all_keys_data: 'typing.Any'
    all_keys_edge: 'typing.Any'
    white_key_default_width: 'typing.Any'
    black_key_default_width: 'typing.Any'
    evaluation_result: 'typing.Any'

    def find_fitting_slot_overlaps(self, rect: np.void|tuple[int, int]|int) -> tuple[int, np.ndarray]:
        """Finds the best matching keyboard key for a rectangle."""
        if isinstance(rect, np.void):
            x0, x1 = rect[["x0", "x1"]]
        elif isinstance(rect, tuple) and len(rect) == 2:
            x0, x1 = rect
        else:
            x0, x1 = rect, rect + 1
        edges = self.all_keys_edge
        rect  = np.asarray([x0, x1],  int)   # (2,)

        # overlap length with rect for each edge
        overlap = np.minimum(edges[:, 1], rect[1]) - np.maximum(edges[:, 0], rect[0])

        idx = int(np.argmax(overlap))
        if overlap[idx] <= 0:
            return -1, None
        return idx, overlap

    def find_best_fitting_slot(self, rect: np.void|tuple[int, int]|int) -> int:
        idx, _ = self.find_fitting_slot_overlaps(rect)
        return idx
    
    def find_multiple_fitting_slots(self, rect: np.void|tuple[int, int]|int) -> np.ndarray:
        idx, overlap = self.find_fitting_slot_overlaps(rect)
        if idx < 0:
            return []
        left = idx
        while left > 0 and overlap[left - 1] > 0:
            left -= 1
        right = idx
        while right + 1 < overlap.shape[0] and overlap[right + 1] > 0:
            right += 1
        fitting_overlap = overlap[left:right +1]
        fitting_widths = np.array([self.all_keys_data[idx]["width"] for idx in range(left, right +1)])
        fitting_overlap_rates = fitting_overlap / fitting_widths
        max_fitting_overlap_rate = fitting_overlap_rates.max()
        if max_fitting_overlap_rate >= 0.95:
            # if any of overlapping keys fully overlaps, the tolerance is 0.9 overlap if there are multiple fitting keys.
            local_fitting_indices = np.flatnonzero(fitting_overlap_rates >= 0.9)
            result_overlap_rates = fitting_overlap_rates[local_fitting_indices]
        else:
            # if there's no fully overlap, the result is argmax, it's assigned to idx already.
            local_fitting_indices = np.array([idx - left])
            result_overlap_rates = np.array([max_fitting_overlap_rate])
        # convert local indices to global indices
        fitting_indices = left + local_fitting_indices
        # return edge x0 and x1 coordinates fitting key slots: array[[x0, x1], [x0, x1], ...]
        result_edges = self.all_keys_edge[fitting_indices]
        result_key_metadatas = [(idx, self.all_keys_data[idx]["color"] == "b", overlap_rate) for idx, overlap_rate in zip(fitting_indices, result_overlap_rates)]
        return result_edges, result_key_metadatas
    
    def get_transpose_suggestion(self) -> int:
        possible_keybed_sizes = [
            (25, 3),
            (32, 3),
            (37, 3),
            (44, 2),
            (49, 2),
            (54, 2),
            (61, 1),
            (73, 0),
            (76, 0),
            (88, 0),
        ]
        key_count = len(self.all_keys_data)
        nearest_idx = int(np.argmin(np.abs(np.array([size for size, _ in possible_keybed_sizes]) - key_count)))
        return int(possible_keybed_sizes[nearest_idx][1])
        


class KeybedDetector:
    STATE_IMAGE_OUTPUT = 0
    STATE_IMAGE_FFT_ANALYSIS = 1
    STATE_IMAGE_KEYBED = 2


    def __init__(self, pref: Preferences, runtime_config: RuntimeConfig):
        self.pref = pref
        self.runtime_config = runtime_config

        self.fft_analyzer = FFTAnalyzer(self.pref)
        self.state = None
        
        self.init_artifact_sink()
        self.init_clustering()

    def init_clustering(self):
        # Set a constant random seed to make KMeans++ random selection near deterministic.
        cv2.setRNGSeed(42)
        # Call the KMeans clustering with dummy matrix to initiate random-state of the cv2.kmeans
        rnd_feature_matrix = np.random.random((10, 2)).astype(np.float32)
        rnd_feature_matrix = Utils.normalize(rnd_feature_matrix)
        self.cluster_kmeans(rnd_feature_matrix)
    
    def init_artifact_sink(self):
        enable_extra_panels = (self.runtime_config.app_mode in [AppMode.GUI_ADVANCED, AppMode.NOTEBOOK])
        artifact_config = {
            "Output": ArtifactConfigEntry(
                ProdMode.PROD, 
                None, 
                emit_fn=lambda data:self.state.set_state_image(__class__.STATE_IMAGE_OUTPUT, data),
            ),
            "FFT Analysis": ArtifactConfigEntry(
                ProdMode.PROD,
                None,
                emit_fn=lambda data:self.state.set_state_image(__class__.STATE_IMAGE_FFT_ANALYSIS, data),
                filename="data/debug/fft_analysis.png",
                enabled=enable_extra_panels,
            ),
            "Keybed": ArtifactConfigEntry(
                ProdMode.PROD,
                None,
                emit_fn=lambda data:self.state.set_state_image(__class__.STATE_IMAGE_KEYBED, data),
                filename="data/debug/keybed.png",
                enabled=enable_extra_panels,
            ),
            "uncorrected_white_keys": ArtifactConfigEntry(
                ProdMode.DEBUG, 
                LogLevel.LOGLEVEL_VERBOSE,
                filename="data/debug/uncorrected_white_keys.png",
            ),
            "corrected_white_keys": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_VERBOSE,
                filename="data/debug/corrected_white_keys.png",
            ),
            "all_keys_on_sobel_x": ArtifactConfigEntry(ProdMode.DEBUG, LogLevel.LOGLEVEL_VERBOSE),
            "debug_print_verbose": ArtifactConfigEntry(ProdMode.DEBUG, LogLevel.LOGLEVEL_VERBOSE),
            "Final Keys Data": ArtifactConfigEntry(ProdMode.DEBUG, LogLevel.LOGLEVEL_DEBUG),
            "Final Keys Edge": ArtifactConfigEntry(ProdMode.DEBUG, LogLevel.LOGLEVEL_DEBUG),
            "Keybed Detection Result": ArtifactConfigEntry(ProdMode.DEBUG, LogLevel.LOGLEVEL_INFO),
        }
        self.artifact_sink = ArtifactSink(artifact_config, self.runtime_config)                

    def set_runtime_config(self, new_val: RuntimeConfig) -> ProcessingState:
        self.runtime_config = new_val
        self.init_artifact_sink()
        return self.init_state()

    def init_state(self):
        panel_titles = ["OUTPUT", "FFT Analysis", "Found Keybed"]
        if self.runtime_config.app_mode == AppMode.GUI_BASIC:
            panel_titles = panel_titles[:1]
        self.state = ProcessingState.from_existing_state(self.state, panel_titles)
        return self.state
    
    def calculate_key_default_width(self, key_widths):
        key_default_width = Utils.mad_filter(key_widths)

        # Detect the anomalies on possible key widths
        key_widths_copy = key_widths.copy()
        if key_widths_copy[0] < key_default_width:
            key_widths_copy[0] = key_default_width
        if key_widths_copy[-1] < key_default_width:
            key_widths_copy[-1] = key_default_width

        outlier_indices = np.where((np.abs(key_widths_copy - np.mean(key_widths_copy)) >= 2 * np.std(key_widths_copy)))[0]
        if not len(outlier_indices):
            return key_default_width
        
        non_outlier_mask = np.ones(len(key_widths), dtype=bool)
        non_outlier_mask[outlier_indices] = False

        non_outlier_widths = key_widths[non_outlier_mask]
        key_default_width = np.median(non_outlier_widths)
        return key_default_width

    def calculate_key_widths(self, possible_white_keys_x):
        possible_key_widths = np.diff(possible_white_keys_x)
        key_default_width = self.calculate_key_default_width(possible_key_widths)
        return key_default_width, possible_key_widths

    def detect_outlier_widths(self, key_widths, tolerance=0.15):
        key_default_width = self.calculate_key_default_width(key_widths)
        return key_default_width, 1 + np.where((np.abs(key_widths[1:-1] - key_default_width) >= tolerance * key_default_width))[0]

    def remove_outlier_white_keys(self, possible_white_keys_x, tolerance=0.15):
        white_key_default_width, possible_white_key_widths = self.calculate_key_widths(possible_white_keys_x)
        are_widths_in_range = np.abs(possible_white_key_widths - white_key_default_width) < tolerance * white_key_default_width
        first_widths_in_range_idx = max(1, np.argmax(are_widths_in_range))
        first_x_in_range_idx = first_widths_in_range_idx
        last_x_in_range = possible_white_keys_x[first_x_in_range_idx]
        keep_x = []
        for i in range(first_x_in_range_idx + 1, len(possible_white_keys_x) - 1):
            current_x = possible_white_keys_x[i]
            distance_by_first_x_in_range = current_x - last_x_in_range
            split_count = distance_by_first_x_in_range / white_key_default_width
            
            split_count_ratio = np.divide(split_count, np.round(split_count), out=np.array([-1], dtype=float), where=np.round(split_count)!=0)

            if split_count_ratio != -1 and np.abs(split_count_ratio - np.round(split_count_ratio)) < 0.1:
                keep_x.append(current_x)
                last_x_in_range = current_x
        keep_x.append(possible_white_keys_x[-1])
        return np.concatenate((possible_white_keys_x[:first_x_in_range_idx + 1], keep_x))

    def split_merged_white_keys(self, possible_white_keys_x, tolerance=0.15):
        white_key_default_width, possible_white_key_widths = self.calculate_key_widths(possible_white_keys_x)
        outlier_indices = 1 + np.where((np.abs(possible_white_key_widths[1:-1] - white_key_default_width) >= tolerance * white_key_default_width))[0]
        if not len(outlier_indices):
            return possible_white_key_widths, white_key_default_width, []

        

        corrected, splits = [], []
        for i, possible_white_key_width in enumerate(possible_white_key_widths):
            if i not in outlier_indices:
                corrected.append(int(possible_white_key_width))
                continue

            split_count = int(round(possible_white_key_width / white_key_default_width))
            if split_count >= 2 and abs(possible_white_key_width - split_count*white_key_default_width) <= tolerance * split_count * white_key_default_width:
                base = possible_white_key_width // split_count
                remaining  = possible_white_key_width - base * split_count
                corrected.extend([base + 1] * remaining + [base] * (split_count - remaining))
                splits.append((i, int(possible_white_key_width), split_count))
            elif split_count == 0:
                raise DetectionException(f"Found an insufficient white key width value {possible_white_key_width} at key {i + 1}. Cannot continue.")
            else:
                corrected.append(int(possible_white_key_width))

        return np.array(corrected, dtype=int), white_key_default_width, splits
    
    def correct_white_keys(self, possible_white_keys_x, tolerance=0.15):
        possible_white_keys_x = self.remove_outlier_white_keys(possible_white_keys_x, tolerance=tolerance)
        corrected_widths, white_key_default_width, splits = self.split_merged_white_keys(possible_white_keys_x, tolerance=tolerance)
        shifted_idx = 0
        corrected_xs = possible_white_keys_x.copy()
        for split_start_idx, split_merged_width, split_count in splits:
            split_start_x = possible_white_keys_x[split_start_idx]
            split_end_idx = split_start_idx + split_count - 1
            splitted_widths = corrected_widths[split_start_idx + shifted_idx:split_end_idx + shifted_idx]
            corrected_local_diffs = np.cumsum(splitted_widths)
            
            corrected_prev = corrected_xs[:split_start_idx + shifted_idx + 1]
            corrected_insert = split_start_x + corrected_local_diffs
            corrected_next = corrected_xs[split_start_idx + shifted_idx + 1:]
            
            corrected_xs = np.concatenate((
                corrected_prev, 
                corrected_insert, 
                corrected_next
            ))
            shifted_idx += len(corrected_local_diffs)
        return corrected_xs, corrected_widths, white_key_default_width

    def enhance_keybed_segment_selection(self, seg_scores: np.ndarray, best_idx: int):
        is_valid_segment = seg_scores > -1
        if not is_valid_segment[best_idx]:
            result_idx_range = (best_idx, best_idx)
        else:
            invalid_indices = np.flatnonzero(~is_valid_segment)
            top_limit_idx = (invalid_indices[invalid_indices < best_idx].max() + 1) if np.any(invalid_indices < best_idx) else 0
            bottom_limit_idx = (invalid_indices[invalid_indices > best_idx].min() - 1) if np.any(invalid_indices > best_idx) else seg_scores.size - 1

            strong_segments = (seg_scores >= 2) & is_valid_segment
            strong_indices = np.flatnonzero(strong_segments)
            strong_indices = strong_indices[(strong_indices >= top_limit_idx) & (strong_indices <= bottom_limit_idx)]
            if strong_indices.size:
                result_idx_range = (strong_indices.min(), strong_indices.max())
            else:
                result_idx_range = (top_limit_idx, bottom_limit_idx)
        return result_idx_range

    async def select_keybed_segment(self, mask: np.ndarray, row_magnitude: np.ndarray, dominant_freqs: np.ndarray, return_debug: bool = False):
        """
        Pick the best horizontal segment (0-runs and 1-runs both considered).
        score(segment) = segment_height * mean(row_magnitude within segment)

        mask: (H, W) bool/uint8
        row_magnitude: (H,) float (e.g., energy or peakiness*energy per row)
        """
        best_mask = None
        H, W = mask.shape
        row_mask = mask.any(axis=1).astype(np.int8)  # 1D rows: 0/1

        # Run-length segments for BOTH values (0 and 1)
        # edges: where value changes, plus start/end
        edges = np.r_[0, np.flatnonzero(np.diff(row_mask) != 0) + 1, H]
        starts, ends = edges[:-1], edges[1:]

        # Score each segment: height * mean(row_magnitude[y0:y1])
        seg_scores = []
        for y0, y1 in zip(starts, ends):
            if y1 <= y0:  # safety
                seg_scores.append(-np.inf)
                continue
            height = y1 - y0
            mean_freq = float(dominant_freqs[y0:y1].mean()) if height > 0 else 0.0
            if mean_freq < self.pref.engine.keybed_min_selected_freqs_mean:
                mean_mag = -np.inf
            else:
                mean_mag = float(row_magnitude[y0:y1].mean()) if height > 0 else 0.0
            seg_scores.append(height * mean_mag)
        seg_scores = np.array(seg_scores)
        if len(seg_scores) == 0:
            if return_debug:
                best_mask = np.zeros_like(mask, dtype=bool)
            return (0, 0, W, 0), best_mask

        best_idx = int(np.argmax(seg_scores))
        if seg_scores[best_idx] < -1:
            return None, None, None
        result_idx_range = self.enhance_keybed_segment_selection(seg_scores, best_idx)
        y0, y1 = int(starts[result_idx_range[0]]), int(ends[result_idx_range[1]])

        await self.artifact_sink.emit_lazy_async(
            "debug_print_verbose",
            lambda: ("\n".join(
                [
                    f"best idx: {best_idx}",
                    f"result_idx_range: {result_idx_range}",
                    "segments:",
                ] + 
                [f"{i}: {t}" for i, t in enumerate(zip(starts, ends, seg_scores))]
            ))
        )

        if return_debug:
            best_rows = np.zeros(H, dtype=bool)
            best_rows[y0:y1] = True
            best_mask = np.repeat(best_rows[:, None], W, axis=1)

        best_segment_rate = (y1 - y0) / H

        return (0, y0, W, y1 - y0), best_mask, best_segment_rate
    
    def cluster_kmeans(self, feature_matrix):
        kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        _, row_cluster_labels, cluster_centers = cv2.kmeans(
            feature_matrix, K=2, bestLabels=None,
            criteria=kmeans_criteria, attempts=1,
            flags=cv2.KMEANS_PP_CENTERS
        )
        row_cluster_labels = row_cluster_labels.ravel()
        return row_cluster_labels, cluster_centers

    async def find_keybed(self, im_gray: np.ndarray, return_debug: bool = False):
        """
        Returns a binary mask (H, W) via KMeans clustering on per-row FFT features.
        Picks the cluster with low frequency + high peakiness.
        """

        # --- feature matrix for clustering ---
        num_rows, num_cols = im_gray.shape
        keybed_min_bottom_pos = int(np.ceil(num_rows * self.pref.engine.keybed_min_bottom_rate))
        fft_analyzer_output = self.fft_analyzer.perform_fft_rows(im_gray)
        feature_matrix = np.vstack([
            fft_analyzer_output.normalized_freqs, 
            fft_analyzer_output.normalized_peakiness
            ]).T.astype(np.float32)

        # --- KMeans clustering ---
        row_cluster_labels, cluster_centers = self.cluster_kmeans(feature_matrix)

        # Pick the cluster with low freq + high peakiness
        cluster_scores = (-cluster_centers[:, 0] + cluster_centers[:, 1])
        keybed_cluster_idx = np.argmax(cluster_scores)

        row_keybed_mask = (row_cluster_labels == keybed_cluster_idx).astype(np.uint8)

        # --- cleanup mask ---
        row_keybed_mask = row_keybed_mask.reshape(-1, 1).astype(np.float32)
        row_keybed_mask_cpy = row_keybed_mask.copy().astype(bool)
        row_keybed_mask = row_keybed_mask.astype(bool)

        keybed_mask = np.repeat(row_keybed_mask[:, None], num_cols, axis=1).squeeze(-1)
        keybed_bounds, keybed_mask, best_segment_rate = await self.select_keybed_segment(keybed_mask, fft_analyzer_output.normalized_peakiness, fft_analyzer_output.dominant_freqs, return_debug=return_debug)

        pre_evaluation_result = None
        if keybed_bounds is None:
            pre_evaluation_result = "No keybed has been detected."
        elif best_segment_rate > self.pref.engine.keybed_min_segment_rate:
            pre_evaluation_result = f"Keybed has been detected with best_segment_rate={best_segment_rate}, which is inappropriate for a valid keybed."
        else:
            selected_freqs = fft_analyzer_output.dominant_freqs[keybed_bounds[1]:keybed_bounds[1]+keybed_bounds[3]]
            selected_freqs_mean = np.mean(selected_freqs)
            if selected_freqs_mean < self.pref.engine.keybed_min_selected_freqs_mean:
                pre_evaluation_result = f"Keybed has been detected with selected_freqs_mean={selected_freqs_mean}, which is inappropriate for a valid keybed."
        if keybed_bounds is not None and pre_evaluation_result is None:
            # Check the position of the keybed: expected is to be positioned at below of the image
            _, keybed_top, _, keybed_height = keybed_bounds
            keybed_bottom_pos = keybed_top + keybed_height
            if keybed_bottom_pos < keybed_min_bottom_pos:
                pre_evaluation_result = f"Keybed has been detected with bottom y={keybed_bounds[1]}, which is inappropriate for a valid keybed, expected minimum bottom y={keybed_min_bottom_pos}."

        debug = None
        if return_debug:
            debug = {
                "row_keybed_mask_cpy": row_keybed_mask_cpy,
                "keybed_mask": keybed_mask,
                "im_gray": im_gray,
                "fft_analyzer_output": fft_analyzer_output,
                "keybed_bounds": keybed_bounds,
            }

        return keybed_bounds, pre_evaluation_result, debug
    
    def find_boundaries_part(self, keybed_edges_part: np.ndarray):
        keybed_width = keybed_edges_part.shape[1]

        # Column-wise vertical-edge energy
        column_edge_energy = keybed_edges_part.sum(axis=0).astype(np.float32)

        # Light smoothing (box filter) to reduce noise
        smooth_len = max(5, keybed_width // 200)
        box_kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
        column_edge_energy = np.convolve(column_edge_energy, box_kernel, mode="same")

        # Normalize to [0,1] and pick strong boundary columns with an adaptive threshold
        col_min = float(column_edge_energy.min())
        col_range = float(np.ptp(column_edge_energy) + 1e-6)
        column_edge_score = (column_edge_energy - col_min) / col_range
        boundary_threshold = float(column_edge_score.mean() + 0.3 * column_edge_score.std()) #0.7
        boundary_mask = (column_edge_score >= boundary_threshold).astype(np.uint8)

        # Collapse contiguous "true" runs to single boundary x-positions (midpoints)
        diff = np.diff(np.r_[0, boundary_mask.astype(np.int8), 0])
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1

        if starts.size == 0 or ends.size == 0:
            boundary_edges_x = np.array([], dtype=int)
            boundary_edges_score = np.array([], dtype=column_edge_score.dtype)
        else:
            if starts.size != ends.size:
                m = min(starts.size, ends.size)
                starts = starts[:m]
                ends = ends[:m]
            boundary_edges_x = (starts + ends) // 2
            prefix = np.concatenate(([0.0], np.cumsum(column_edge_score)))
            counts = (ends - starts + 1).astype(np.float32)
            boundary_edges_score = (prefix[ends + 1] - prefix[starts]) / counts
        
        if not len(boundary_edges_x):
            raise DetectionException("No boundary found")

        min_boundary_separation = int(0.006 * keybed_width)

        boundary_edges_x = np.asarray(boundary_edges_x)
        boundary_edges_score = np.asarray(boundary_edges_score)
        possible_prune_mask = np.concatenate([[True], np.diff(boundary_edges_x) >= min_boundary_separation])
        possible_prune_indices = np.where(~possible_prune_mask)[0]
        prune_mask = np.ones(possible_prune_mask.shape, dtype=bool)

        for possible_prune_idx in possible_prune_indices:
            current_scoore = boundary_edges_score[possible_prune_idx]
            prev_score = boundary_edges_score[possible_prune_idx - 1]
            prune_idx = possible_prune_idx if current_scoore < prev_score else (possible_prune_idx - 1)
            prune_mask[prune_idx] = False
        boundary_edges_x = boundary_edges_x[prune_mask == 1]
        boundary_edges_score = boundary_edges_score[prune_mask == 1]
        boundary_score_mean = boundary_edges_score.mean()

        add_left_edge = boundary_edges_x[0] - 0 > min_boundary_separation
        add_right_edge = keybed_width - boundary_edges_x[-1] > min_boundary_separation
        boundary_edges_x = np.concatenate([
            np.array([0] if add_left_edge else [], dtype=boundary_edges_x.dtype), 
            boundary_edges_x, 
            np.array([keybed_width - 1] if add_right_edge else [], dtype=boundary_edges_x.dtype)
        ])
        boundary_edges_score = np.concatenate([
            np.array([1] if add_left_edge else [], dtype=boundary_edges_score.dtype), 
            boundary_edges_score, 
            np.array([1] if add_right_edge else [], dtype=boundary_edges_score.dtype)
        ])

        return boundary_edges_x, boundary_edges_score, boundary_score_mean

    def find_and_correct_full_whites(self, white_keys_dict):
        white_keys_x = list(white_keys_dict.keys())
        if len(white_keys_x) < 7:
            raise DetectionException(f"Not enough white keys found: {len(white_keys_x)}. Expected more than 7.")
        actual_full_indices = np.array([i for i, v in enumerate(white_keys_dict.values()) if v['is_full']])

        if not len(actual_full_indices):
            raise DetectionException("No full white key found. Cannot continue.")
        start_index = actual_full_indices[0]
        focused_num_keys = len(white_keys_x) - start_index + 1

        # Geneate octave patterns for broader octave range back and forth.
        # Because while cumulative summing of larger negative numbers may cancel necessary positive numbers.
        # In further lines, unnecessary items will be filtered out.
        negative_octave_count = int(np.ceil((start_index + 1) / 7) + 1)
        positive_octave_count = int(np.ceil(focused_num_keys / 7 + negative_octave_count + 1))
        
        # Generate ideal order starting from note C
        deltas_c_start = np.concatenate((
            np.tile([-4, -3], negative_octave_count),
            np.tile([3, 4], positive_octave_count),
            #np.array([3])
        ))
        ideal_indices_c = np.unique(start_index + np.cumsum(np.insert(deltas_c_start, 0, 0)))
        ideal_indices_c = ideal_indices_c[(ideal_indices_c >= 0) & (ideal_indices_c < focused_num_keys + start_index)]

        # Generate ideal order starting from note F
        deltas_f_start = np.concatenate((
            np.tile([-3, -4], negative_octave_count),
            np.tile([4, 3], positive_octave_count),
            #np.array([4])
        ))
        ideal_indices_f = np.unique(start_index + np.cumsum(np.insert(deltas_f_start, 0, 0)))
        ideal_indices_f = ideal_indices_f[(ideal_indices_f >= 0) & (ideal_indices_f < focused_num_keys + start_index)]

        # Score two alternatives, select the best fitting one
        score_c = np.sum(np.isin(actual_full_indices, ideal_indices_c))
        score_f = np.sum(np.isin(actual_full_indices, ideal_indices_f))

        if score_c != score_f:
            ideal_indices_list = [ideal_indices_c if score_c >= score_f else ideal_indices_f]
            white_keys_dict_cpy = None
        else:
            ideal_indices_list = [ideal_indices_c, ideal_indices_f]
            white_keys_dict_cpy = white_keys_dict.copy()
        
        ideal_indices_success = False
        for i_try, ideal_indices in enumerate(ideal_indices_list):
            if i_try > 0:
                if ideal_indices_success:
                    break
                white_keys_dict = white_keys_dict_cpy
            ideal_indices_success = True
            is_first_run = True
            mismatched_indices_of_indices = []
            while is_first_run or len(mismatched_indices_of_indices):
                is_first_run = False
                white_keys_x = list(white_keys_dict.keys())
                actual_full_indices = np.array([i for i, v in enumerate(white_keys_dict.values()) if v['is_full']])
                matching_mask = np.isin(ideal_indices, actual_full_indices)
                mismatched_indices_of_indices = np.where(matching_mask == False)[0] # noqa: E712
                matching_mask = np.isin(actual_full_indices, ideal_indices)
                mismatched_indices_of_indices = np.sort(np.unique(np.concatenate((mismatched_indices_of_indices, np.where(matching_mask == False)[0])))) # noqa: E712

                if not len(mismatched_indices_of_indices):
                    break

                mismatched_idx_of_indices = mismatched_indices_of_indices[0]
                if mismatched_idx_of_indices >= len(ideal_indices):
                    break
                ideal_idx = ideal_indices[mismatched_idx_of_indices]
                if mismatched_idx_of_indices < len(actual_full_indices):
                    actual_full_idx = actual_full_indices[mismatched_idx_of_indices]
                    if ideal_idx < actual_full_idx: # A missing is_full flag in actual_full_indices
                        if len(ideal_indices_list) > 0 and i_try < len(ideal_indices_list) - 1:
                            if len(white_keys_dict[white_keys_x[int(ideal_idx)]]["black_keys_x"]) < 2:
                                white_keys_dict[white_keys_x[int(ideal_idx)]]["is_full"] = True
                            else:
                                ideal_indices_success = False
                                break
                        else:
                            white_keys_dict[white_keys_x[int(ideal_idx)]]["is_full"] = True
                    else: # An extra is_full flag in actual_full_indices
                        white_keys_dict[white_keys_x[int(actual_full_idx)]]["is_full"] = False
                    continue
                else:
                    if ideal_idx < len(white_keys_x):
                        white_keys_dict[white_keys_x[int(ideal_idx)]]["is_full"] = True
                    else:
                        ideal_indices_success = False
                        break

    def find_and_correct_black_keys(self, white_keys_dict, note_names):
        blacks_per_white_dict = [
            (False, True), # C, Left=No black key, Right=Black key exists
            (True, True),  # D,
            (True, False),  # E,
            (False, True),  # F,
            (True, True),  # G,
            (True, True),  # A,
            (True, False),  # B,
            ]
        
        black_keys_per_white_dict = {}
        for white_key_idx, (white_key_x, white_key_data) in enumerate(white_keys_dict.items()):
            black_left_expected, black_right_expected = blacks_per_white_dict[white_key_data["note_num"]]
            if black_right_expected:
                black_keys_per_white_dict[white_key_idx] = {
                    "note_num": white_key_data["note_num"],
                    "note_name": white_key_data["note_name"] + "#",
                    "x_left": None,
                    "x_right": None
                }
            elif white_key_idx == 0 and black_left_expected:
                note_num = (white_key_data["note_num"] - 1) % len(note_names)
                black_keys_per_white_dict[white_key_idx] = {
                    "note_num": note_num,
                    "note_name": note_names[note_num] + "#",
                    "x_left": None,
                    "x_right": None
                }

        for white_key_idx, (white_key_x, white_key_data) in enumerate(white_keys_dict.items()):
            white_key_width = white_key_data["width"]
            black_left_expected, black_right_expected = blacks_per_white_dict[white_key_data["note_num"]]
            black_keys_x = white_key_data["black_keys_x"]
            black_keys_local_x = [black_key_x - white_key_x for black_key_x in black_keys_x]
            black_keys_local_side = {}
            if len(black_keys_local_x) == 1 and (black_left_expected + black_right_expected) == 1:
                black_keys_local_side["single"] = black_keys_x[0]
            else:
                black_keys_local_side = {("left" if abs(local_x-0) <= abs(local_x-white_key_width + 3) else "right"): global_x for local_x, global_x in zip(black_keys_local_x, black_keys_x)}
            if black_left_expected and (black_keys_local_side.get("left", None) or black_keys_local_side.get("single", None)):
                if white_key_idx > 0:
                    black_keys_per_white_dict[white_key_idx - 1]["x_right"] = black_keys_local_side.get("left", None) or black_keys_local_side.get("single", None)
                else:
                    black_keys_per_white_dict[white_key_idx]["x_left"] = max(0, white_key_x - white_key_width // 2)
                    black_keys_per_white_dict[white_key_idx]["x_right"] = black_keys_local_side.get("left", None) or black_keys_local_side.get("single", None)

            if black_right_expected and (black_keys_local_side.get("right", None) is not None or black_keys_local_side.get("single", None) is not None):
                black_keys_per_white_dict[white_key_idx]["x_left"] = black_keys_local_side.get("right", None) or black_keys_local_side.get("single", None)

        black_keys_width = []
        for black_key_idx, (white_key_idx, black_key_data) in enumerate(black_keys_per_white_dict.items()):
            if not (black_key_data["x_left"] is not None or black_key_data["x_right"] is not None):
                if black_key_idx in [0, len(black_keys_per_white_dict) -1]:
                    black_keys_per_white_dict[white_key_idx] = None
                else:
                    raise DetectionException(f"No x-coordinate could not been determined for black key idx={black_key_idx}, note_name={black_key_data["note_name"]}")
            elif black_key_data["x_left"] is not None and black_key_data["x_right"] is not None:
                black_keys_width.append(black_key_data["x_right"] - black_key_data["x_left"])

        black_key_default_width = int(Utils.mad_filter(black_keys_width))

        black_key_widths = []
        for black_key_idx, (white_key_idx, black_key_data) in enumerate(black_keys_per_white_dict.items()):
            if not black_key_data:
                continue
            if not (black_key_data["x_left"] is not None and black_key_data["x_right"] is not None):
                if black_key_data["x_left"]:
                    black_key_data["x_right"] = black_key_data["x_left"] + black_key_default_width
                else:
                    black_key_data["x_left"] = black_key_data["x_right"] - black_key_default_width
            black_key_data["width"] = black_key_data["x_right"] - black_key_data["x_left"]
            black_key_widths.append(int(black_key_data["width"]))
        
        black_key_widths = np.array(black_key_widths)
        
        outlier_width_indices = np.where((np.abs(black_key_widths - black_key_default_width) >= 0.20 * black_key_default_width))[0]
        if len(outlier_width_indices) > 0:
            black_key_xs = list(black_keys_per_white_dict.keys())
            for outlier_width_idx in outlier_width_indices:
                black_key_data = black_keys_per_white_dict[black_key_xs[outlier_width_idx]]
        return black_keys_per_white_dict, black_key_default_width

    def find_octaves(self, white_keys_dict):
        note_names = ["C", "D", "E", "F", "G", "A", "B"]
        note_all_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        self.find_and_correct_full_whites(white_keys_dict)

        actual_full_indices = [i for i, v in enumerate(white_keys_dict.values()) if v['is_full']]

        start_full_index = actual_full_indices[0]
        next_full_index = actual_full_indices[1]
        start_note_num = 0 if (next_full_index - start_full_index) == 3 else 3 # Starting from note C or note F

        for white_key_idx, (white_key_x, white_key_data) in enumerate(white_keys_dict.items()):
            note_num = (white_key_idx - start_full_index + start_note_num) % len(note_names)
            white_key_data["note_num"] = note_num
            white_key_data["note_name"] = note_names[note_num]
        

        black_keys_per_white_dict, black_key_default_width = self.find_and_correct_black_keys(white_keys_dict, note_names)

        all_keys_edge = []
        all_keys_data = []

        for white_key_x, white_key_data in white_keys_dict.items():
            all_keys_data.append({
                "color": "w",
                "width": white_key_data["width"],
                "is_full": white_key_data["is_full"],
                "note_num": white_key_data["note_num"],
                "note_name": white_key_data["note_name"],
            })
            all_keys_edge.append([white_key_x, white_key_x + white_key_data["width"]])

        for _, black_key_data in black_keys_per_white_dict.items():
            if not black_key_data:
                continue
            all_keys_data.append({
                "color": "b",
                "width": black_key_data["width"],
                "note_num": black_key_data["note_num"],
                "note_name": black_key_data["note_name"],
            })
            all_keys_edge.append([black_key_data["x_left"], black_key_data["x_right"]])
        all_keys_data = np.asarray(all_keys_data)
        all_keys_edge = np.asarray(all_keys_edge, int)
        sorted_idx = np.argsort(all_keys_edge, axis=0)[:,0]
        all_keys_data = all_keys_data[sorted_idx]
        all_keys_edge = all_keys_edge[sorted_idx]

        note_c_num = note_names.index("C")
        midi_num_c0 = 12
        octave_num = 0
        last_note_c_idx = -2
        for i, key_data in enumerate(all_keys_data):
            if key_data["note_num"] == note_c_num and last_note_c_idx != i - 1:
                last_note_c_idx = i
                if i > 0:
                    octave_num += 1
            key_data["octave"] = octave_num
            local_note_num = note_all_names.index(key_data["note_name"])
            key_data["midi_num"] = midi_num_c0 + octave_num * len(note_all_names) + local_note_num
            key_data["full_note_name"] = f"{key_data["note_name"]}{octave_num}"

        return all_keys_data, all_keys_edge, black_key_default_width

    def find_keys(self, kb_internal_image_input: KeybedInternalImageInput, return_debug: bool = False):
        all_keys_data, all_keys_edge, black_key_default_width, error = None, None, 0, None
        keybed_edges_parts = kb_internal_image_input.keybed_edges_parts
        boundary_edges_x_top, _, _ = self.find_boundaries_part(keybed_edges_parts[0])
        boundary_edges_x_bottom, _, _ = self.find_boundaries_part(keybed_edges_parts[1])
        try:
            white_keys_corrected_x, white_keys_corrected_width, white_keys_corrected_default_width = self.correct_white_keys(boundary_edges_x_bottom)
        except Exception as e:
            error = e
            return all_keys_data, all_keys_edge, 0, 0, error, None
        debug = None
        if return_debug:
            debug = {
                "boundary_edges_x_bottom": boundary_edges_x_bottom,
                "white_keys_corrected_x": white_keys_corrected_x,
            }

        white_keys_dict = OrderedDict((white_key_x, {
            "width": white_key_width, 
            "is_full": False, 
            "black_keys_x": []
            })
            for white_key_x, white_key_width in zip(white_keys_corrected_x.tolist(), white_keys_corrected_width.tolist()
        ))

        # Assume we have:
        # boundary_edges_x_top =                       array([   0,   25,   59,   87,  114,  148, ...]) # x values of possible black keys
        # white_keys_corrected_x =                     array([   0,   33,   84,  133,  184,  234, ...]) # x values of possible white keys
        #
        # we expect the following to be:
        # expected mapping :                                    (0 to 0), (25 to 0), (59 to 33), (87 to 84), (114 to 84), (148 to 133), ...
        # (boundary_edges_x_top to white_keys_corrected_x)

        # black_key_rightest_white_key_idx =           array([   0,    0,    1,    2,    2,    3, ...])
        # black_key_rightest_white_key_x =             array([   0,    0,   33,   84,   84,  133, ...])

        black_key_x_treshold = float(white_keys_corrected_default_width) / 10
        black_key_rightest_white_key_idx = np.searchsorted(white_keys_corrected_x, boundary_edges_x_top, side='right') - 1 # -1 here shifts the result one step left
        black_key_rightest_white_key_x = white_keys_corrected_x[black_key_rightest_white_key_idx]

        for black_key_edge_x, rightest_white_key_x in zip(boundary_edges_x_top[1:-1].tolist(), black_key_rightest_white_key_x[1:-1]):
            if black_key_edge_x - rightest_white_key_x < black_key_x_treshold:
                white_keys_dict[rightest_white_key_x]["is_full"] = True
                continue
            white_keys_dict[rightest_white_key_x]["black_keys_x"].append(black_key_edge_x)
        try:
            all_keys_data, all_keys_edge, black_key_default_width = self.find_octaves(white_keys_dict)
        except Exception as e:
            error = e
        return all_keys_data, all_keys_edge, int(white_keys_corrected_default_width), int(black_key_default_width), error, debug

    def evaluate_detected_keys(self, all_keys_data): 
        actual_order = np.array([(key_data["note_name"], key_data["color"]) for key_data in all_keys_data])

        ideal_order_note_names = np.array([("C", "w"), ("C#", "b"), ("D", "w"), ("D#", "b"), ("E", "w"), ("F", "w"), ("F#", "b"), ("G", "w"), ("G#", "b"), ("A", "w"), ("A#", "b"), ("B", "w")])
        ideal_order_start = actual_order[0]
        ideal_start_idx = np.argmax(np.all(ideal_order_note_names == ideal_order_start, axis=1))
        # ideal_order_note_names has 12 items
        # Assume all_keys_data has 88 items, and the first item of all_keys_data index is ideal_order_start=9
        # Roll items of ideal_order_note_names starting from 9th item, traverse left to right until 
        # filling out ideal_order list with 88 items.
        ideal_order = ideal_order_note_names[(np.arange(len(all_keys_data)) + ideal_start_idx) % len(ideal_order_note_names)]

        white_keys_width = np.array([[key_data["width"], i] for i, key_data in enumerate(all_keys_data) if key_data["color"] == "w"])
        black_keys_width = np.array([[key_data["width"], i] for i, key_data in enumerate(all_keys_data) if key_data["color"] == "b"])
        
        white_key_default_width, white_outlier_local_indices = self.detect_outlier_widths(white_keys_width[:, 0], tolerance=0.2)
        black_key_default_width, black_outlier_local_indices = self.detect_outlier_widths(black_keys_width[:, 0], tolerance=0.2)
        
        all_outlier_messages = []
        if len(white_outlier_local_indices) or len(black_outlier_local_indices):
            white_outlier_global_indices = white_keys_width[white_outlier_local_indices][:,1]
            black_outlier_global_indices = black_keys_width[black_outlier_local_indices][:,1]
            white_outliers_data = all_keys_data[white_outlier_global_indices]
            black_outliers_data = all_keys_data[black_outlier_global_indices]
            if len(white_outliers_data):
                all_outlier_messages.append(f"Width outliers in the white keys. Expected average width: {white_key_default_width}, but found:\n   * " + "\n   * ".join([f"index: {white_outlier_global_idx}, note name: {white_outlier_data["note_name"]}, width: {white_outlier_data["width"]}" for white_outlier_global_idx, white_outlier_data in zip(white_outlier_global_indices, white_outliers_data)]))
            if len(black_outliers_data):
                all_outlier_messages.append(f"Width outliers in the black keys. Expected average width: {black_key_default_width}, but found:\n   * " + "\n   * ".join([f"index: {black_outlier_global_idx}, note name: {black_outlier_data["note_name"]}, width: {black_outlier_data["width"]}" for black_outlier_global_idx, black_outlier_data in zip(black_outlier_global_indices, black_outliers_data)]))
        
        mismatch_idx  = np.flatnonzero(np.all(actual_order != ideal_order, axis=1))
        mismatch_pairs = list(zip(mismatch_idx, actual_order[mismatch_idx], ideal_order[mismatch_idx]))
        if len(mismatch_pairs):
            all_outlier_messages.append("Mismatched key order found:\n   * " + "\n   * ".join([str(pair) for pair in mismatch_pairs]))
        return "Detected keys were evaluated and failed:\n- " + "\n".join(all_outlier_messages) if all_outlier_messages else None

    async def detect(self, kb_image_input: KeybedImageInput) -> KeybedDetectorOutput:
        self.artifact_sink.emit("Output", kb_image_input.im_bgr) # Original frame
        evaluation_result_is_system_exception = False
        try:
            is_keybed_bounds_valid = False
            want_debug_data = self.artifact_sink.wants("FFT Analysis")
            keybed_bounds, pre_evaluation_result, keybed_debug = await self.find_keybed(kb_image_input.im_gray, return_debug=want_debug_data)
            self.artifact_sink.emit("debug_print_verbose", f"keybed_bounds {keybed_bounds}")
            if pre_evaluation_result is not None:
                evaluation_result = pre_evaluation_result
            else:
                is_keybed_bounds_valid = True
                kb_internal_image_input = await ImagePreprocessor.preprocess_for_keybed_internal(kb_image_input, keybed_bounds)
                all_keys_data, all_keys_edge, white_key_default_width, black_key_default_width, error, keys_debug = self.find_keys(kb_internal_image_input, return_debug=want_debug_data)
                if keybed_debug and keys_debug:
                    keys_debug["keybed_bounds"] = keybed_bounds
                if error is not None and isinstance(error, Exception):
                    evaluation_result = error
                else:
                    evaluation_result = self.evaluate_detected_keys(all_keys_data)
        except Exception as e:
            is_keybed_bounds_valid = False
            evaluation_result = e
            evaluation_result_is_system_exception = isinstance(e, Exception) and not isinstance(evaluation_result, DetectionException)
        if want_debug_data and not evaluation_result_is_system_exception:
            im_vis_bgr = kb_image_input.im_bgr.copy()
            # im_vis_bgr status: Copy of original frame
            await self.artifact_sink.emit_lazy_async(
                "Keybed",
                lambda: KeybedRenderer.render_keybed_detection(
                    im_vis_bgr,
                    keybed_debug["keybed_bounds"],
                    self.pref,
                    evaluation_result is None,
                ),
            )
            # im_vis_bgr status: Keybed rectangle drawn frame
            await self.artifact_sink.emit_lazy_async(
                "FFT Analysis",
                lambda: KeybedRenderer.render_fft_analysis(
                    keybed_debug["row_keybed_mask_cpy"],
                    keybed_debug["keybed_mask"],
                    keybed_debug["im_gray"],
                    keybed_debug["fft_analyzer_output"],
                )
            )

        if not is_keybed_bounds_valid:
            return KeybedDetectorOutput(
                keybed_bounds=keybed_bounds if not evaluation_result_is_system_exception else None,
                all_keys_data=None,
                all_keys_edge=None,
                white_key_default_width=None, 
                black_key_default_width=None, 
                evaluation_result=evaluation_result,
            )

        if self.artifact_sink.wants("uncorrected_white_keys"):
            im_vis_uncorrected_white_keys = im_vis_bgr.copy()
            KeybedRenderer.render_white_key_edges(
                im_vis_uncorrected_white_keys,
                keys_debug["boundary_edges_x_bottom"],
                keys_debug["keybed_bounds"],
                self.pref,
            )
            self.artifact_sink.emit("uncorrected_white_keys", im_vis_uncorrected_white_keys)
            del im_vis_uncorrected_white_keys

        if self.artifact_sink.wants("corrected_white_keys"):
            im_vis_corrected_white_keys = im_vis_bgr.copy()
            KeybedRenderer.render_white_key_edges(
                im_vis_corrected_white_keys,
                keys_debug["white_keys_corrected_x"],
                keys_debug["keybed_bounds"],
                self.pref,
            )
            KeybedRenderer.render_white_key_text(
                im_vis_corrected_white_keys,
                keys_debug["white_keys_corrected_x"],
                keys_debug["keybed_bounds"],
                self.pref,
            )
            # Artifact: Corrected white keys (VERBOSE)
            self.artifact_sink.emit("corrected_white_keys", im_vis_corrected_white_keys)
            del im_vis_corrected_white_keys

        if isinstance(evaluation_result, DetectionException):
            return KeybedDetectorOutput(
                keybed_bounds=None,
                all_keys_data=None,
                all_keys_edge=None,
                white_key_default_width=None, 
                black_key_default_width=None, 
                evaluation_result=str(evaluation_result),
            )

        keybed_output = KeybedDetectorOutput(
            keybed_bounds=keybed_bounds,
            all_keys_data=all_keys_data,
            all_keys_edge=all_keys_edge,
            white_key_default_width=white_key_default_width, 
            black_key_default_width=black_key_default_width, 
            evaluation_result=evaluation_result,
        )


        if evaluation_result is None:
            if self.artifact_sink.wants("all_keys_on_sobel_x"):
                im_sobel_x = np.vstack(kb_internal_image_input.keybed_edges_parts)
                im_vis_all_keys_on_sobel_x = (im_sobel_x * 255).astype(np.uint8)
                im_vis_all_keys_on_sobel_x = cv2.cvtColor(im_vis_all_keys_on_sobel_x, cv2.COLOR_GRAY2BGR)
                del im_sobel_x
            else:
                im_vis_all_keys_on_sobel_x = None
            if self.artifact_sink.wants("Keybed") or self.artifact_sink.wants("all_keys_on_sobel_x"):
                im_vis_all_keys = im_vis_bgr.copy()
                KeybedRenderer.render_keys(im_vis_all_keys, im_vis_all_keys_on_sobel_x, keybed_output, self.pref)
                self.artifact_sink.emit("all_keys_on_sobel_x", im_vis_all_keys_on_sobel_x)
                self.artifact_sink.emit("Keybed", im_vis_all_keys)

            await self.artifact_sink.emit_lazy_async("Final Keys Data", lambda: json.dumps(keybed_output.all_keys_data.tolist(), indent=4))
            await self.artifact_sink.emit_lazy_async("Final Keys Edge", lambda: json.dumps(keybed_output.all_keys_edge.tolist(), indent=4))
        elif isinstance(evaluation_result, Exception):
            raise evaluation_result

        await self.artifact_sink.emit_lazy_async("Keybed Detection Result", lambda: self.report_keybed_detection_result(keybed_output))
        return keybed_output

    def report_keybed_detection_result(self, keybed_output: KeybedDetectorOutput):
        if keybed_output.evaluation_result:
            return f"Keybed detection has been failed!\n{keybed_output.evaluation_result}" 
        messages = [
            f"Total key count: {len(keybed_output.all_keys_data)}",
            f"White key count: {len([key_data for key_data in keybed_output.all_keys_data if key_data["color"] == "w"])}",
            f"Black key count: {len([key_data for key_data in keybed_output.all_keys_data if key_data["color"] == "b"])}",
            f"Starts with note {keybed_output.all_keys_data[0]["note_name"]}",
            f"Ends with note {keybed_output.all_keys_data[-1]["note_name"]}",
            f"White key default width: {keybed_output.white_key_default_width}",
            f"Black key default width: {keybed_output.black_key_default_width}",
        ]
        return f"Keybed detection has been succeeded!\n{"\n".join(messages)}"
