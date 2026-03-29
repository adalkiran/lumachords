from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable
import cv2
import numpy as np

from lumachords.data_types import as_structured_array
from lumachords.preferences import Preferences
from lumachords.artifact_sink import ArtifactSink
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.utils import Utils


@dataclass
class HandsDetectorOutputRangeItem:
    x: int
    key_idx: int
    midi_num: int
    note_name: str

@dataclass
class HandMidiNumRange:
    start_midi_num: int
    end_midi_num: int

@dataclass
class HandsMidiNumRangesPerTime:
    pts_time_array = np.array([], dtype=float)
    items: list[list[HandMidiNumRange]] = field(default_factory=list)

    def append(self, pts_time: float, hands_midi_num_ranges: list[HandMidiNumRange], check_if_last_same: bool=True):
        if check_if_last_same and len(self.items):
            last = self.items[-1]
            if len(last) == len(hands_midi_num_ranges) and all([last_item == new_item for last_item, new_item in zip(last, hands_midi_num_ranges)]):
                return
        self.pts_time_array = np.append(self.pts_time_array, pts_time)
        self.items.append(hands_midi_num_ranges)
    
    def find_index(self, pts_time: float):
        idx = np.searchsorted(self.pts_time_array, pts_time, side="right") - 1
        return int(idx)

    def find(self, pts_time: float):
        idx = self.find_index(pts_time)
        return self.items[idx] if idx > -1 else None
    
    def get_latest_two_hands(self):
        for rev_i, item in enumerate(reversed(self.items)):
            if len(item) == 2:
                return len(self.items) - 1 - rev_i
        return len(self.items) - 1

@dataclass
class HandsDetectorOutputRanges:
    items: list[tuple[HandsDetectorOutputRangeItem, HandsDetectorOutputRangeItem]]

    def to_midi_num_ranges(self) -> list[HandMidiNumRange]:
        return [(HandMidiNumRange(hand[0].midi_num, hand[1].midi_num) if hand is not None else None) for hand in self.items]


class HandsColorTracker:
    def __init__(self):
        self.color_ranges = np.full((2, 2), -1, dtype=float)
        self.hand_color_ids = None

    def _midpoints(self, mask=None):
        ranges = self.color_ranges if mask is None else self.color_ranges[mask]
        return ranges.mean(axis=1)

    def _is_new_hand(self, new_mid, existing_mids):
        if len(existing_mids) == 0:
            return True
        if len(existing_mids) == 1:
            # With only one tracked hand there is no spread estimate, so use a
            # fixed hue-distance threshold to decide whether this is a new hand.
            return np.abs(existing_mids[0] - new_mid) > 10

        dists = np.abs(existing_mids - new_mid)
        threshold = existing_mids.std() + dists.mean()
        return dists.min() > threshold

    def update(self, new_ranges) -> np.ndarray:
        """
        Update color_ranges with new_ranges and return the index assignments.
        
        Args:
            new_ranges: list of (min, max) tuples, one per detected hand
        Returns:
            indices: np.ndarray of shape (N,), mapping each new_range to its slot (0 or 1)
        """
        new_ranges = np.array(new_ranges)
        indices = np.full(len(new_ranges), -1, dtype=int)

        for i, new in enumerate(new_ranges):
            active = self.color_ranges[:, 0] != -1

            if not active.any():
                self.color_ranges[0] = new
                indices[i] = 0

            elif not active.all():
                existing_mids = self._midpoints(active)
                new_mid = new.mean()
                if self._is_new_hand(new_mid, existing_mids):
                    empty_idx = np.where(~active)[0][0]
                    self.color_ranges[empty_idx] = new
                    indices[i] = empty_idx
                else:
                    closest_idx = np.where(active)[0][np.argmin(np.abs(existing_mids - new_mid))]
                    self.color_ranges[closest_idx, 0] = min(self.color_ranges[closest_idx, 0], new[0])
                    self.color_ranges[closest_idx, 1] = max(self.color_ranges[closest_idx, 1], new[1])
                    indices[i] = closest_idx

            else:
                existing_mids = self._midpoints()
                new_mid = new.mean()
                closest_idx = np.argmin(np.abs(existing_mids - new_mid))
                self.color_ranges[closest_idx, 0] = min(self.color_ranges[closest_idx, 0], new[0])
                self.color_ranges[closest_idx, 1] = max(self.color_ranges[closest_idx, 1], new[1])
                indices[i] = closest_idx

        return indices

    def update_by_values(self, hue_vals_list: list[np.ndarray]) -> list[np.ndarray]:
        hue_ranges = []
        for hue_vals in hue_vals_list:
            if len(hue_vals):
                hue_ranges.append((hue_vals.min(), hue_vals.max()))
        return self.update(hue_ranges)

    def get_color_id(self, hue_value: int):
        active = self.color_ranges[:, 0] != -1
        if not active.any():
            return -1

        existing_mids = self._midpoints(active)
        closest_local_idx = np.argmin(np.abs(existing_mids - hue_value))
        return int(np.where(active)[0][closest_local_idx])

    def __repr__(self):
        def fmt(r):
            return f"({int(r[0])}, {int(r[1])})" if r[0] != -1 else "unset"
        return f"Slot 0: {fmt(self.color_ranges[0])}\nSlot 1: {fmt(self.color_ranges[1])}"


@dataclass
class HandsDetectorOutput:
    skin_mask: np.ndarray
    hands_mask: np.ndarray
    hands_bgr: np.ndarray
    ranges: HandsDetectorOutputRanges


DT_HUE_RANGE = np.dtype([
    ('x_start',   'i2'),
    ('x_end',   'i2'),
    ('hue_start',   'i2'),
    ('hue_end',   'i2'),
    ('range_count_sum',   'i2'),
    ('frequent_hue',   'i2'),
    ('frequent_count',   'i2'),
    ('frequent_density',   'float'),
    ('range_score',   'float'),
])


class HandsType(IntEnum):
    HANDS = 0
    COLORS = 1

    def __str__(self):
        return "HANDS" if self == HandsType.HANDS else "COLORS"

class HandsDetector:
    def __init__(
            self,
            pref: Preferences,
            keybed_output: KeybedDetectorOutput,
            artifact_sink: ArtifactSink,
            keep_largest=2,           # keep N biggest connected components (2 hands)
            blur_ksize=5,             # 0 to disable
            morph_ksize=5,
            filter_components: bool=True,
            hands_type_callback_fn: Callable[[HandsType], None]=None,
    ):
        self.pref = pref
        self.keybed_output = keybed_output
        self.artifact_sink = artifact_sink
        self.keep_largest = keep_largest
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize
        self.filter_components = filter_components
        self.hands_type_callback_fn = hands_type_callback_fn

        self.color_tracker = HandsColorTracker()
        self.known_hands_type: HandsType = None
        self.known_hands_type_hist: list[HandsType] = []

    def darken_top_linear(self, im_bgr: np.ndarray, top_gain: float = 0.2, y_end: float = 0.6) -> np.ndarray:
        """
        top_gain: brightness multiplier at very top (0..1). e.g. 0.6 = 40% darker.
        y_end: fraction of image height where fade ends (bottom part stays unchanged).
            e.g. 0.6 means from y=0..0.6H fades, and 0.6H..H is unchanged.
        """
        h, w = im_bgr.shape[:2]
        y_end_px = int(h * y_end)

        # build per-row gain: top_gain -> 1.0, then clamp to 1.0 for bottom
        g = np.ones(h, np.float32)
        if y_end_px > 0:
            g[:y_end_px] = np.linspace(top_gain, 1.0, y_end_px, dtype=np.float32)

        # apply to BGR (broadcast over width and channels)
        out = (im_bgr.astype(np.float32) * g[:, None, None]).clip(0, 255).astype(np.uint8)
        return out

    def merge_by_vertical_dilate(self, mask_u8: np.ndarray, gap_thr: int) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_thr * 2 + 1))
        dil = cv2.dilate(mask_u8, k, iterations=1)
        # optionally erode back a bit if you don't want thickening:
        out = cv2.erode(dil, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_thr * 2 + 1)), iterations=1)
        return out

    def extract_hands(
        self,
        im_crop_keybed_bgr: np.ndarray,
        crop_keybed_extra_height: int
    ):
        """
        Skin-color segmentation in YCrCb.
        Returns: hands_mask (uint8 0/255 full image), hands_bgr (BGR with bg removed), raw_skin_mask
        """
        im_crop_keybed_bgr = self.darken_top_linear(im_crop_keybed_bgr)
        h, w = im_crop_keybed_bgr.shape[:2]

        x0 = y0 = 0
        im = im_crop_keybed_bgr

        # --- Optional blur to reduce noise ---
        if self.blur_ksize and self.blur_ksize > 1:
            im_blur = cv2.GaussianBlur(im, (self.blur_ksize, self.blur_ksize), 0)
        else:
            im_blur = im

        # --- YCrCb threshold ---
        ycrcb = cv2.cvtColor(im_blur, cv2.COLOR_BGR2YCrCb)
        # Assumed skin color ranges, differs by lighting/camera.
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([173, 173, 127], dtype=np.uint8)

        skin_mask = cv2.inRange(ycrcb, lower, upper)  # 0/255

        # --- Cleanup: morphology ---
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k, iterations=2)

        k_break = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))  # wide & short
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k_break, iterations=1)

        skin_mask = self.merge_by_vertical_dilate(skin_mask, int(h *0.1))

        num, labels, stats, _ = cv2.connectedComponentsWithStats((skin_mask > 0).astype(np.uint8), connectivity=8)

        if self.filter_components:
            stats1 = stats[1:]
            keybed_height = h - crop_keybed_extra_height
            edge_left = w * 0.1
            edge_right = w - edge_left
            edge_bottom_individually = keybed_height * 0.6
            edge_bottom_with_top_clause = keybed_height * 0.8

            if num > 4:
                remove_ids_mask = (stats1[:, cv2.CC_STAT_TOP] + stats1[:, cv2.CC_STAT_HEIGHT] < edge_bottom_individually)
                remove_ids_mask |= (
                    (
                        (stats1[:, cv2.CC_STAT_LEFT] < edge_left) |
                        (stats1[:, cv2.CC_STAT_LEFT] + stats1[:, cv2.CC_STAT_WIDTH] > edge_right)
                    ) &
                    (stats1[:, cv2.CC_STAT_HEIGHT] > keybed_height * 0.7)
                )
            else:
                remove_ids_mask = (stats1[:, cv2.CC_STAT_WIDTH] > w // 2)
                remove_ids_mask |= (
                    (stats1[:, cv2.CC_STAT_TOP] + stats1[:, cv2.CC_STAT_HEIGHT] < edge_bottom_individually) |
                    (
                        (stats1[:, cv2.CC_STAT_TOP] < 5) & 
                        (stats1[:, cv2.CC_STAT_TOP] + stats1[:, cv2.CC_STAT_HEIGHT] < edge_bottom_with_top_clause)
                    )
                )

            keep_ids = np.flatnonzero(~remove_ids_mask) + 1  # label ids in [1..num-1]

            # Apply filtering to the mask
            skin_mask = (np.isin(labels, keep_ids).astype(np.uint8) * 255)

        else:
            # Keep everything except background
            keep_ids = np.arange(1, num, dtype=np.int32)      # label ids in [1..num-1]
            # skin_mask stays as-is

        # Now stats_kept always matches keep_ids
        stats_kept = stats[keep_ids]  # shape (len(keep_ids), 5)

        # --- Keep largest connected components (usually 2 hands) ---
        if self.keep_largest is not None and self.keep_largest > 0 and keep_ids.size > 0:
            areas = stats_kept[:, cv2.CC_STAT_AREA]
            ok = areas > 0.01 * w * h
            keep_ids2 = keep_ids[ok]
            areas2 = areas[ok]

            order = np.argsort(areas2)  # ascending
            chosen = keep_ids2[order[-min(self.keep_largest, keep_ids2.size):]]  # label ids
            hands_mask = (np.isin(labels, chosen).astype(np.uint8) * 255)
        else:
            hands_mask = skin_mask

        # --- Paste back into full-size mask if ROI was used ---
        full_hands_mask = np.zeros((h, w), dtype=np.uint8)
        full_skin_mask = np.zeros((h, w), dtype=np.uint8)
        full_hands_mask[y0:y0 + hands_mask.shape[0], x0:x0 + hands_mask.shape[1]] = hands_mask
        full_skin_mask[y0:y0 + skin_mask.shape[0], x0:x0 + skin_mask.shape[1]] = skin_mask

        # --- Apply mask to original image ---
        bg = (128, 128, 128)  # BGR background color you want

        m = (full_hands_mask > 0)[..., None]          # (H,W,1) boolean
        hands_bgr = np.where(m, im_crop_keybed_bgr, np.array(bg, dtype=im_crop_keybed_bgr.dtype))

        return full_skin_mask, full_hands_mask, hands_bgr

    def detect_hands(
        self,
        im_crop_keybed_bgr: np.ndarray,
        crop_keybed_extra_height: int,
        transpose_octaves: int,
        return_debug: bool,
    ) -> HandsDetectorOutput:
        skin_mask, hands_mask, hands_bgr = self.extract_hands(im_crop_keybed_bgr, crop_keybed_extra_height)
        hands_mask_bool = (hands_mask > 0)
        column_mask = hands_mask_bool.any(axis=0).astype(np.uint8)

        # Collapse contiguous "true" runs to single boundary x-positions (midpoints)
        diff = np.diff(np.r_[0, column_mask.astype(np.int8), 0])
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1

        ranges = HandsDetectorOutputRanges([])
        for x_parts in zip(starts, ends):
            range_item_part_list = []
            for x_part in x_parts:
                key_idx = self.keybed_output.find_best_fitting_slot(x_part)
                midi_num = self.keybed_output.all_keys_data[key_idx]["midi_num"] + transpose_octaves * 12
                note_name = Utils.midi_num_to_name(midi_num)
                range_item_part_list.append(HandsDetectorOutputRangeItem(x_part, key_idx, midi_num, note_name))
            ranges.items.append((range_item_part_list[0], range_item_part_list[1]))

        return HandsDetectorOutput(
            skin_mask=skin_mask if return_debug else None,
            hands_mask=hands_mask if return_debug else None,
            hands_bgr=hands_bgr,
            ranges=ranges,
        )
    
    def group_hue_ranges(self, hue_roi: np.ndarray) -> np.ndarray:
        vals, cnts = np.unique(hue_roi, return_counts=True)

        keep = cnts >= cnts.mean()
        vals, cnts = vals[keep], cnts[keep]

        gap = 3  # merge if next_range_min - prev_range_max < 3  (i.e. break only if >= 3)

        starts = np.r_[0, np.flatnonzero(np.diff(vals) >= gap) + 1]
        ends   = np.r_[starts[1:] - 1, vals.size - 1]
        sum_cnts = np.add.reduceat(cnts, starts)

        gid = np.cumsum(np.isin(np.arange(vals.size), starts)) - 1
        order = np.lexsort((vals, -cnts, gid))  # per-group max cnt (tie -> smaller hue)
        first = np.r_[0, np.flatnonzero(np.diff(gid[order])) + 1]
        top_i = order[first]

        ranges = np.c_[
            np.full(sum_cnts.size, -1, dtype=np.int32),
            np.full(sum_cnts.size, -1, dtype=np.int32),
            vals[starts], 
            vals[ends], 
            sum_cnts,
            vals[top_i], 
            cnts[top_i], 
            np.full(sum_cnts.size, -1, dtype=np.int32), 
            np.full(sum_cnts.size, -1, dtype=np.int32),
        ]
        ranges = as_structured_array(ranges, DT_HUE_RANGE)
        return ranges
    
    def find_dominant_hue_range(self, hue_roi: np.ndarray, x_start: int, x_end: int) -> np.ndarray:
        ranges = self.group_hue_ranges(hue_roi)
        range_width = x_end - x_start + 1
        dominant_range = ranges[np.argmax(ranges["range_count_sum"])]
        dominant_range["x_start"] = x_start
        dominant_range["x_end"] = x_end
        dominant_range["frequent_hue"]
        frequent_density = dominant_range["frequent_density"] = dominant_range["range_count_sum"] / hue_roi.size
        frequent_exp_density = np.exp(4.0 * frequent_density)
        dominant_range["range_score"] = frequent_exp_density * range_width
        return dominant_range

    def detect_colors(
        self,
        im_crop_keybed_bgr: np.ndarray,
        crop_keybed_extra_height: int,
        transpose_octaves: int,
        return_debug: bool,
    ) -> HandsDetectorOutput:
        H, W, _ = im_crop_keybed_bgr.shape
        im_crop_keybed_bgr = im_crop_keybed_bgr.copy()
        hsv = cv2.cvtColor(im_crop_keybed_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        val = hsv[..., 2]

        # Keep only sufficiently saturated and non-dark pixels.
        keep_mask = (sat > 50) & (val > 40)
        sum_horiz = np.sum(keep_mask, axis=0)
        raw_horiz_mask = sum_horiz >= H*0.33
        diff = np.diff(np.r_[False, raw_horiz_mask, False].astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        min_true_run = int(self.keybed_output.black_key_default_width * 0.6)
        valid_runs = (ends - starts) >= min_true_run
        valid_starts = starts[valid_runs]
        valid_ends = ends[valid_runs]
        raw_horiz_mask[:] = False
        processed_horiz_mask = np.zeros_like(raw_horiz_mask)
        
        dominant_hue_ranges_list = []
        for i, (start, end) in enumerate(zip(valid_starts, valid_ends)):
            hue_roi = hue[:, start:end][keep_mask[:, start:end]]            
            dominant_hue_ranges_list.append(self.find_dominant_hue_range(hue_roi, start, end))
        del keep_mask

        dominant_hue_ranges = np.array(dominant_hue_ranges_list, dtype=DT_HUE_RANGE)
        
        hue_horiz_means = hue.mean(axis=0)
        hue_mean_thr = Utils.k_sigma_threshold(hue_horiz_means, k=1)

        if len(dominant_hue_ranges):
            dominant_hue_ranges_mean = np.r_[dominant_hue_ranges["hue_start"], dominant_hue_ranges["hue_end"]].mean()
            if dominant_hue_ranges_mean < hue_mean_thr:
                dominant_hue_ranges = dominant_hue_ranges[dominant_hue_ranges["hue_start"] > hue_mean_thr]
        
        frequent_densities = dominant_hue_ranges["frequent_density"]
        if np.any(frequent_densities < 0.8):
            range_scores = dominant_hue_ranges["range_score"]
            thr = Utils.k_sigma_threshold(range_scores, k=-0.3)
            high_score_ranges = dominant_hue_ranges[range_scores > thr]
        else:
            high_score_ranges = dominant_hue_ranges
        
        high_score_hue_vals = high_score_ranges["frequent_hue"].copy()
        if len(high_score_hue_vals) > 1:
            high_score_hue_vals.sort()
            gaps = np.diff(high_score_hue_vals)
            threshold = np.maximum(10, gaps.mean() + 1 * gaps.std())
            gap_split = np.argmax(gaps)
            if gaps[gap_split] < threshold:
                split_hue_vals = [high_score_hue_vals]
            else:
                split_hue_vals = [high_score_hue_vals[:gap_split + 1], high_score_hue_vals[gap_split + 1:]]
        else:
            split_hue_vals = [high_score_hue_vals]
        self.color_tracker.update_by_values(split_hue_vals)

        color_ids = np.array([self.color_tracker.get_color_id(hue_range["frequent_hue"]) for hue_range in dominant_hue_ranges], dtype=int)
        range_scores = dominant_hue_ranges["range_score"].copy()
        merge_colors = True
 
        for start, end in zip(valid_starts, valid_ends):
            raw_horiz_mask[start:end] = True
 
        if len(color_ids):
            if merge_colors:
                merged_breaks = np.r_[0, 1 + np.flatnonzero(np.diff(color_ids) != 0), len(color_ids)]
                range_scores = np.add.reduceat(range_scores, merged_breaks[:-1])
                valid_starts = valid_starts[merged_breaks[:-1]]
                valid_ends = valid_ends[merged_breaks[1:] - 1]
                color_ids = color_ids[merged_breaks[:-1]]
            if merge_colors and len(color_ids) > 2:
                thr = Utils.k_sigma_threshold(range_scores, k=-0.5)
                keep_idx = np.flatnonzero(range_scores >= thr)
                valid_starts = valid_starts[keep_idx]
                valid_ends = valid_ends[keep_idx]
                color_ids = color_ids[keep_idx]
                range_scores = range_scores[keep_idx]


                merged_breaks = np.r_[0, 1 + np.flatnonzero(np.diff(color_ids) != 0), len(color_ids)]
                range_scores = np.add.reduceat(range_scores, merged_breaks[:-1])
                valid_starts = valid_starts[merged_breaks[:-1]]
                valid_ends = valid_ends[merged_breaks[1:] - 1]
                color_ids = color_ids[merged_breaks[:-1]]


                keep_idx = np.argsort(range_scores)[-2:]
                keep_idx.sort()
                valid_starts = valid_starts[keep_idx]
                valid_ends = valid_ends[keep_idx]
                color_ids = color_ids[keep_idx]
                range_scores = range_scores[keep_idx]
        else:
            valid_starts, valid_ends = [], []
        for start, end in zip(valid_starts, valid_ends):
            processed_horiz_mask[start:end] = True

        if return_debug:
            raw_mask = (np.repeat(raw_horiz_mask[None, :], im_crop_keybed_bgr.shape[0], axis=0).astype(np.uint8) * 255)
            processed_mask = (np.repeat(processed_horiz_mask[None, :], im_crop_keybed_bgr.shape[0], axis=0).astype(np.uint8) * 255)
        else:
            raw_mask = None
            processed_mask = None

        colors_bgr = np.zeros_like(im_crop_keybed_bgr)
        colors_bgr[:, processed_horiz_mask] = im_crop_keybed_bgr[:, processed_horiz_mask]

        for start, end in zip(valid_starts, valid_ends):
            raw_horiz_mask[start:end] = True

        ranges = HandsDetectorOutputRanges([])
        for x_parts in zip(valid_starts, valid_ends):
            range_item_part_list = []
            for x_part in x_parts:
                key_idx = self.keybed_output.find_best_fitting_slot(x_part)
                midi_num = self.keybed_output.all_keys_data[key_idx]["midi_num"] + transpose_octaves * 12
                note_name = Utils.midi_num_to_name(midi_num)
                range_item_part_list.append(HandsDetectorOutputRangeItem(x_part, key_idx, midi_num, note_name))
            ranges.items.append((range_item_part_list[0], range_item_part_list[1]))

        if len(color_ids) == 2:
            self.color_tracker.hand_color_ids = color_ids
        elif len(color_ids) == 1 and self.color_tracker.hand_color_ids is not None:
            single_color_id = color_ids[0]
            if self.color_tracker.hand_color_ids[0] == single_color_id:
                ranges.items.append(None)
            else:
                ranges.items.insert(0, None)

        return HandsDetectorOutput(
            skin_mask=raw_mask,
            hands_mask=processed_mask,
            hands_bgr=colors_bgr,
            ranges=ranges,
        )
    

    async def detect(
        self,
        im_crop_keybed_bgr: np.ndarray,
        crop_keybed_extra_height: int,
        transpose_octaves: int,
        return_debug: bool=False,
    ) -> HandsDetectorOutput:
        if self.known_hands_type is None:
            current_hands_type = None
            hands_output = self.detect_hands(im_crop_keybed_bgr, crop_keybed_extra_height, transpose_octaves, return_debug)
            if len(hands_output.ranges.items):
                current_hands_type = HandsType.HANDS
                final_output = hands_output
            else:
                colors_output = self.detect_colors(im_crop_keybed_bgr, crop_keybed_extra_height, transpose_octaves, return_debug)
                if len(colors_output.ranges.items):
                    current_hands_type = HandsType.COLORS
                # if colors_output is success, output it. If not success, hands_output was already failed, so there's no difference between outputting any of them.
                final_output = colors_output
            if current_hands_type is not None:
                self.known_hands_type_hist.append(current_hands_type)
            if len(self.known_hands_type_hist) > 1 and self.known_hands_type_hist[-2] != self.known_hands_type_hist[-1]:
                self.known_hands_type_hist = []
            if len(self.known_hands_type_hist) == 10:
                self.known_hands_type = self.known_hands_type_hist[-1]
                self.known_hands_type_hist = []
                if self.hands_type_callback_fn is not None:
                    self.hands_type_callback_fn(self.known_hands_type)
        elif self.known_hands_type == HandsType.HANDS:
            final_output = self.detect_hands(im_crop_keybed_bgr, crop_keybed_extra_height, transpose_octaves, return_debug)
        else:
            final_output = self.detect_colors(im_crop_keybed_bgr, crop_keybed_extra_height, transpose_octaves, return_debug)
        if self.artifact_sink.wants("Lines"):
            color = self.pref.appearance.floating_box_border_color_bgr
            for range_item in final_output.ranges.items:
                if range_item is None:
                    continue
                range_item_start, range_item_end = range_item
                x_start, x_end = range_item_start.x, range_item_end.x
                text_start, text_end = range_item_start.note_name, range_item_end.note_name
                cv2.line(final_output.hands_bgr, (x_start, 0), (x_start, final_output.hands_bgr.shape[0]), color, 2)
                cv2.line(final_output.hands_bgr, (x_end, 0), (x_end, final_output.hands_bgr.shape[0]), color, 2)

                cv2.putText(final_output.hands_bgr, f"{text_start} - {text_end}", (x_start + 3, int(final_output.hands_bgr.shape[0] * 0.2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        return final_output
