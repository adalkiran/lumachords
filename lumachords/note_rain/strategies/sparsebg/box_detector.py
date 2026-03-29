
import cv2
import numpy as np

from lumachords.artifact_sink import ArtifactSink
from lumachords.data_types import DT_RECT
from lumachords.image_input import NoteRainImageInput
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.preferences import Preferences
from lumachords.rendering import EdgeRenderer
from lumachords.utils import Utils

from ...note_rain_utils import NoteRainUtils


class BoxDetector:
    """Detects note rain boxes directly from sparse images."""

    def __init__(self, pref: Preferences, artifact_sink: ArtifactSink, keybed_output: KeybedDetectorOutput):
        self.pref = pref
        self.artifact_sink = artifact_sink
        self.keybed_output = keybed_output
        self.keybed_top_y = keybed_output.keybed_bounds[1]
        self.note_rain_boundary_limits = NoteRainUtils.calculate_boundary_limits(pref, keybed_output)

    async def preprocess_image(self, im_crop_luma):
        H, W = im_crop_luma.shape

        cv2.rectangle(
            im_crop_luma,
            (0, 0),
            (W-1, H-1),
            (255),
            2,
        )

        min_dim = min(H, W)
        original_mask = (im_crop_luma > 0).astype(np.uint8)

        # Remove thin horizontal/vertical line noise while preserving note boxes.
        kernel_size = max(5, int(round(min_dim * 0.007)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        clean = cv2.morphologyEx(original_mask * 255, cv2.MORPH_OPEN, open_kernel)

        # Drop any remaining long, thin components from the cleaned image itself.
        clean_num_labels, _, clean_stats, _ = cv2.connectedComponentsWithStats((clean > 0).astype(np.uint8), connectivity=8)
        thin_limit = max(7, kernel_size + 2)
        long_limit = max(32, int(round(min_dim * 0.25)))
        for comp_id in range(1, clean_num_labels):
            x, y, w, h, _ = clean_stats[comp_id]
            is_vertical_line = w <= thin_limit and h >= long_limit
            is_horizontal_line = h <= thin_limit and w >= long_limit
            if is_vertical_line or is_horizontal_line:
                clean[y:y+h, x:x+w] = 0

        clean_mask = (clean > 0)
        row_sums = np.zeros((clean_mask.shape[0] + 1))
        row_sums[:-1] = np.sum(clean_mask, axis=1)
        row_sums[row_sums < W *0.75] = 0
        diff = np.diff(row_sums)
        starts = np.flatnonzero(diff > 0)
        ends = np.flatnonzero(diff < 0)
        if starts.size > 0 and starts[-1] > H - 1:
            starts = starts[:-1]
        if ends.size > 0 and ends[-1] > H - 1:
            ends[-1] = ends[-1] - 1

        if not (starts.size == 0 and ends.size == 0):
            if starts.size != ends.size:
                m = min(starts.size, ends.size)
                starts = starts[:m]
                ends = ends[:m]
            for start, end in zip(starts, ends):
                length = end - start +1
                if length < 0:
                    continue
                local_mask = np.ones((length, W), dtype=bool)
                if start > 0:
                    local_mask[:, clean_mask[start-1]] = False
                if end < H - 1:
                    local_mask[:, clean_mask[end+1]] = False
                clean_mask[start:end+1, :] = ~local_mask


        im_crop_luma[~clean_mask] = 0
        im_crop_vis_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        
        # Scharr gradient magnitude
        sX = cv2.Scharr(im_crop_luma, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_CONSTANT)
        sY = cv2.Scharr(im_crop_luma, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_CONSTANT)

        for grads in [sX, sY]:
            abs_grads = np.abs(grads)
            thr = Utils.k_sigma_threshold(abs_grads, self.pref.engine.gradient_threshold_standard_deviation_factor)
            if thr >= abs_grads.max():
                thr = 0.9 * float(abs_grads.max())
            grads[abs_grads < thr] = 0

        mag = np.sqrt(sX**2 + sY**2)
        mag_mask = (mag > 0).astype(np.uint8)

        return im_crop_luma, mag_mask, im_crop_vis_bgr
        
    def find_drammatic_change(self, diff: np.ndarray, only_negatives: bool = True):
        if only_negatives:
            focus_diff = diff[diff < 0]
        else:
            focus_diff = diff
        if not focus_diff.size:
            return None
        first_diff = focus_diff[0]
        if focus_diff.size > 1 and not any(item for item in focus_diff[1:] if item != first_diff):
            return None
        mu, sigma = focus_diff.mean(), focus_diff.std()
        k = 1.0 if len(focus_diff) > 10 else 0.4
        if only_negatives:
            thr = mu - k * sigma  # note: thr is negative
            idxs = np.flatnonzero(diff <= thr)
            idx_first = idxs[0] if idxs.size else None
        else:
            thr = mu + k * sigma
            idxs = np.flatnonzero(diff >= thr)
            idx_first = idxs[0] if idxs.size else None
        return idx_first

    def find_edges(self, grads_mask: np.ndarray, box_candidate: tuple[int, int, int, int], vert_tol: int):
        pad = 1
        H, W = grads_mask.shape
        x0, y0, x1, y1 = box_candidate
        drammatic_change_x_left, drammatic_change_y_top, drammatic_change_x_right, drammatic_change_y_bottom = None, None, None, None
        grads_horiz = grads_mask[y0:y1+1, :].copy()

        sum_horiz = np.sum(grads_horiz, axis=0)
        sum_horiz_limit = np.minimum(np.max(sum_horiz) * 0.5, H*0.2)
        sum_horiz[sum_horiz < sum_horiz_limit] = 0
        grads_vert = grads_mask[:, x0:x1+1].copy()
        sum_vert = np.sum(grads_vert, axis=1)
        sum_vert_limit = np.minimum(np.max(sum_vert) * 0.4, W*0.2)
        sum_vert[sum_vert < sum_vert_limit] = 0

        # check towards left
        left_x0 = max(0, x0 - vert_tol)
        left_x1 = min(W, x0 + pad)
        diff_rev_left = np.diff(sum_horiz[left_x0:left_x1][::-1])
        drammatic_change_idx = self.find_drammatic_change(diff_rev_left)
        if drammatic_change_idx is None:
            diff_right = np.diff(sum_horiz[left_x0:W])
            drammatic_change_idx = self.find_drammatic_change(diff_right, only_negatives=True)
            drammatic_change_x_left = np.maximum(0, left_x0+drammatic_change_idx - 1) if drammatic_change_idx is not None else None
        else:
            drammatic_change_x_left = np.maximum(0, left_x1 - 1 - drammatic_change_idx) if drammatic_change_idx is not None else None  
        if drammatic_change_x_left is not None and np.abs(drammatic_change_x_left - x1) < (x1 - x0) * 0.2:
            drammatic_change_x_left = None

        # check towards right
        right_x0 = max(0, x1 - pad)
        right_x1 = min(W, x1 + vert_tol)
        diff_right = np.diff(sum_horiz[right_x0:right_x1])
        drammatic_change_idx = self.find_drammatic_change(diff_right)
        if drammatic_change_idx is None:
            diff_rev_left = np.diff(sum_horiz[:right_x1][::-1])
            drammatic_change_idx = self.find_drammatic_change(diff_rev_left, only_negatives=True)
            drammatic_change_x_right = (right_x1 - 1 - drammatic_change_idx) if drammatic_change_idx is not None else None
        else:
            drammatic_change_x_right = (right_x0 + drammatic_change_idx) if drammatic_change_idx is not None else None
        if drammatic_change_x_right is not None and np.abs(drammatic_change_x_right - x0) < (x1 - x0) * 0.2:
            drammatic_change_x_right = None

        if drammatic_change_x_left is not None or drammatic_change_x_right is not None:
            tmp_x0 = drammatic_change_x_left if drammatic_change_x_left is not None else x0
            tmp_x1 = drammatic_change_x_right if drammatic_change_x_right is not None else x1
            
            grads_horiz = grads_mask[y0:y1+1, tmp_x0:tmp_x1+1].copy()
            sum_horiz = np.sum(grads_horiz, axis=0)
            sum_horiz[sum_horiz < sum_horiz_limit] = 0
            grads_vert = grads_mask[:, tmp_x0:tmp_x1+1].copy()
            sum_vert = np.sum(grads_vert, axis=1)
            sum_vert[sum_vert < sum_vert_limit] = 0

        # check towards top
        top_y0 = max(0, y0 - vert_tol)
        top_y1 = min(H, y0 + pad)
        diff_rev_top = np.diff(sum_vert[top_y0:top_y1][::-1])
        
        drammatic_change_idx = self.find_drammatic_change(diff_rev_top)
        if drammatic_change_idx is None:
            diff_bottom = np.diff(sum_vert[top_y0:H])
            drammatic_change_idx = self.find_drammatic_change(diff_bottom, only_negatives=True)
            drammatic_change_y_top = np.maximum(0, top_y0 + drammatic_change_idx - 1) if drammatic_change_idx is not None else None
        else:
            drammatic_change_y_top = np.maximum(0, top_y1 - 1 - drammatic_change_idx) if drammatic_change_idx is not None else None  
        if drammatic_change_y_top is not None and np.abs(drammatic_change_y_top - y1) < (y1 - y0) * 0.2:
            drammatic_change_y_top = None
        
        # check towards bottom
        bottom_y0 = max(0, y1 - pad)
        bottom_y1 = min(H, y1 + vert_tol*3)
        diff_bottom = np.diff(sum_vert[bottom_y0:bottom_y1])
        drammatic_change_idx = self.find_drammatic_change(diff_bottom)
        if drammatic_change_idx is None:
            diff_rev_top = np.diff(sum_vert[:bottom_y1][::-1])
            drammatic_change_idx = self.find_drammatic_change(diff_rev_top, only_negatives=True)
            drammatic_change_y_bottom = (bottom_y1 - 1 - drammatic_change_idx) if drammatic_change_idx is not None else None
        else:
            drammatic_change_y_bottom = (bottom_y0 + drammatic_change_idx) if drammatic_change_idx is not None else None
        if drammatic_change_y_bottom is not None and np.abs(drammatic_change_y_bottom - y0) < (y1 - y0) * 0.2:
            drammatic_change_y_bottom = None
        #return (None, None, None, None)
        return (drammatic_change_x_left, drammatic_change_y_top, drammatic_change_x_right, drammatic_change_y_bottom)

    def fine_grain_refine_box_candidate(
            self,
            im_crop_luma_part: np.ndarray,
            im_crop_bgr_part: np.ndarray,
            vert_tol: int,
            key_slot_edge_count: int,
            global_x0: int,
            global_y0: int,
            refine_using_grads: bool = True,
        ) -> list[tuple[bool, np.ndarray]]:
        #Utils.imshow(im_crop_bgr_part, f"crop bgr full")


        note_rain_boundary_limits = self.note_rain_boundary_limits
        min_component_area = note_rain_boundary_limits.final_min_width * note_rain_boundary_limits.final_min_height


        mask_part = (im_crop_luma_part > 0)

        # if this part fits in multiple keys and
        # 95% of mask_part is True, this may be false flag, so ignore.
        if key_slot_edge_count > 1 and mask_part.sum() / (mask_part.shape[0] * mask_part.shape[1]) > 0.95:
            return None

        box_candidates_list = []

        part_axis_abs_grads_list = []
        for dx, dy in [(1, 0), (0, 1)]:
            grads = cv2.Scharr(im_crop_luma_part, cv2.CV_64F, dx, dy, borderType=cv2.BORDER_CONSTANT)
            abs_grads = np.abs(grads)
            thr = Utils.k_sigma_threshold(abs_grads, k=1)
            if thr >= abs_grads.max():
                thr = 0.9 * float(abs_grads.max())
            abs_grads[abs_grads < thr] = 0
            part_axis_abs_grads_list.append(abs_grads)

        part_abs_grads = np.maximum(part_axis_abs_grads_list[0], part_axis_abs_grads_list[1])

        part_grads_mask = (part_abs_grads > 0) 

        im_crop_luma_part_to_erode = im_crop_luma_part.copy()

        im_crop_luma_part_to_erode[part_grads_mask] = 0
        eroded_mask_part = (im_crop_luma_part_to_erode > 0)
        del im_crop_luma_part_to_erode

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(eroded_mask_part.astype(np.uint8), connectivity=8)
        del eroded_mask_part
        for comp_id in range(1, num_labels):
            x, y, w, h, _ = stats[comp_id]
    
            area = w*h
            if area < min_component_area:
                continue

            x0, y0, x1, y1 = x, y, x+w, y+h
            if refine_using_grads:
                new_x0, new_y0, new_x1, new_y1 = self.find_edges(part_grads_mask, (x0, y0, x1, y1), vert_tol)
                x0 = new_x0 if new_x0 is not None else x0
                y0 = new_y0 if new_y0 is not None else y0
                x1 = new_x1 if new_x1 is not None else x1
                y1 = new_y1 if new_y1 is not None else y1
                w = x1-x0
                h = y1-y0
            
            if w < note_rain_boundary_limits.final_min_width or h < note_rain_boundary_limits.final_min_height:
                continue

            box_candidates_list.append((-1, global_x0+x0, global_y0+y0, global_x0+x1, global_y0+y1, -1, 1, -999, -999))

        if len(box_candidates_list):
            box_candidates = np.array(box_candidates_list, dtype=DT_RECT)
            order = np.argsort(box_candidates["key_idx"])
            box_candidates = box_candidates[order]
            valid_indices = []
            for i, box in enumerate(box_candidates):
                if not self.check_fitting_key_slot(box):
                    continue
                valid_indices.append(i)
            box_candidates = box_candidates[valid_indices]
            box_candidates = self.drop_covered_box_candidates(box_candidates, intersection_area_limit_multiplier=0.4)
        else:
            box_candidates = np.empty((0,), dtype=DT_RECT)

        return box_candidates
        
    def check_box_brightness_density(self, im_crop_luma: np.ndarray, box: np.ndarray) -> bool:
        x0, y0, x1, y1 = box[["x0", "y0", "x1", "y1"]]
        luma_part = im_crop_luma[y0:y1, x0:x1]
        if luma_part.size == 0:
            return False
        bright_pixel_rate = np.sum(luma_part.ravel() > 0) / luma_part.size
        luma_max = np.max(luma_part)
        result = bright_pixel_rate >= 0.7 and luma_max >= 80
        return result
    
    def check_fitting_key_slot(self, box: np.ndarray) -> bool:
        key_slot_edges, key_slot_metadata = self.keybed_output.find_multiple_fitting_slots((box["x0"], box["x1"]))
        # allow box to fit in max. 2 slots if one is black, one is white
        if len(key_slot_edges) > 3:
            return False
        elif len(key_slot_edges) > 1:
            # allow consecutive fully overlapping black-white-black combination
            # allow consecutive fully overlapping white-black + partial overlapping white combination (order-independent)
            # not allow consecutive fully overlapping white-white-black
            key_scores = np.array([float(not is_black)*overlap_rate for _, is_black, overlap_rate in key_slot_metadata])
            if np.count_nonzero(key_scores > 0.9) > 1:
                return False
        return True

    def fine_grain_refine_box_candidates(
            self,
            box_candidates: np.ndarray,
            im_crop_luma: np.ndarray,
            im_crop_bgr: np.ndarray
        ) -> np.ndarray:
        boxes_list = []
        vert_tol = int(np.ceil(im_crop_luma.shape[1] * self.pref.engine.note_rain_horizontal_line_vertical_snap_tolerance_rate))

        for box in box_candidates:
            x0, y0, x1, y1 = box[["x0", "y0", "x1", "y1"]]
            # key_slot_edges = array[[x0, x1], [x0, x1], ...]
            key_slot_edges, _ = self.keybed_output.find_multiple_fitting_slots((x0, x1))

            # if there are more than one fitting slots, re-analysis local areas
            im_crop_luma_part = im_crop_luma[y0:y1, x0:x1]
            im_crop_bgr_part = im_crop_bgr[y0:y1, x0:x1]
            split_boxes = self.fine_grain_refine_box_candidate(im_crop_luma_part, im_crop_bgr_part, vert_tol, len(key_slot_edges), x0, y0)
            if split_boxes is None:
                continue
            if len(split_boxes) == 0:
                if not self.check_fitting_key_slot(box):
                    continue
                if not self.check_box_brightness_density(im_crop_luma, box):
                    continue
                boxes_list.extend([box])
            else:
                valid_indices = [] 
                for i, split_box in enumerate(split_boxes):
                    key_slot_edges, key_slot_metadata = self.keybed_output.find_multiple_fitting_slots((split_box["x0"], split_box["x1"]))
                    # allow box to fit in max. 2 slots if one is black, one is white
                    if len(key_slot_edges) > 3:
                        continue
                    elif len(key_slot_edges) > 1:
                        # allow consecutive fully overlapping black-white-black combination
                        # allow consecutive fully overlapping white-black + partial overlapping white combination (order-independent)
                        # not allow consecutive fully overlapping white-white-black
                        key_scores = np.array([float(not is_black)*overlap_rate for _, is_black, overlap_rate in key_slot_metadata])
                        if np.count_nonzero(key_scores > 0.8) > 1:
                            continue
                    if not self.check_box_brightness_density(im_crop_luma, split_box):
                        continue
                    valid_indices.append(i)
                split_boxes = split_boxes[valid_indices]
                boxes_list.extend(split_boxes)
        
        boxes = np.array(boxes_list, dtype=DT_RECT)
        return boxes

    def drop_covered_box_candidates(self, box_candidates: np.ndarray, intersection_area_limit_multiplier:float = 0.8):
        if len(box_candidates) > 1:
            areas = (box_candidates["x1"] - box_candidates["x0"]).astype("i4") * (box_candidates["y1"] - box_candidates["y0"]).astype("i4")
            keep = np.ones(len(box_candidates), dtype=bool)
            for i in np.argsort(-areas):
                if not keep[i]:
                    continue
                for j in np.flatnonzero(keep & (areas < areas[i])):
                    inter_w = max(0, min(box_candidates["x1"][i], box_candidates["x1"][j]) - max(box_candidates["x0"][i], box_candidates["x0"][j]))
                    inter_h = max(0, min(box_candidates["y1"][i], box_candidates["y1"][j]) - max(box_candidates["y0"][i], box_candidates["y0"][j]))
                    if inter_w * inter_h >= areas[j] * intersection_area_limit_multiplier:
                        keep[j] = False
            box_candidates = box_candidates[keep]
        return box_candidates


    async def detect_box_edges(self, im_crop_luma, im_crop_bgr):
        im_crop_luma, mag_mask, im_crop_vis_bgr = await self.preprocess_image(im_crop_luma)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mag_mask, connectivity=8)
        box_candidates_list = []
        note_rain_boundary_limits = self.note_rain_boundary_limits
        min_component_area = note_rain_boundary_limits.final_min_width * note_rain_boundary_limits.final_min_height
        for comp_id in range(1, num_labels):
            x, y, w, h, _ = stats[comp_id]
            area = w * h
            cv2.rectangle(
                im_crop_vis_bgr,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2,
            )
            
            if area < min_component_area or w < note_rain_boundary_limits.final_min_width or h < note_rain_boundary_limits.final_min_height:
                continue

            x0, y0, x1, y1 = x, y, x+w, y+h
            box_candidates_list.append((-1, x0, y0, x1, y1, -1, 1, -999, -999))

        box_candidates = np.array(box_candidates_list, dtype=DT_RECT)
        # DROP COVERED BOXES
        box_candidates = self.drop_covered_box_candidates(box_candidates)

        # ANALYZE BY SCHARR GRADIENTS
        box_candidates = self.fine_grain_refine_box_candidates(box_candidates, im_crop_luma, im_crop_bgr)

        # DROP COVERED BOXES (AGAIN)
        boxes = self.drop_covered_box_candidates(box_candidates)

        # DRAW TEMPORARILY

        for box in boxes:
            cv2.rectangle(
                im_crop_vis_bgr,
                (box["x0"], box["y0"]),
                (box["x1"], box["y1"]),
                self.pref.appearance.end_line_color_bgr,
                4,
            )    
        return boxes, im_crop_vis_bgr


    async def detect_boxes(self, nr_image_input: NoteRainImageInput, hands_bgr: np.array):
        im_crop_luma, im_crop_bgr = nr_image_input.im_crop_luma, nr_image_input.im_crop_bgr
        self.artifact_sink.emit("Crop Luma", im_crop_luma)

        boxes, im_crop_vis_bgr = await self.detect_box_edges(im_crop_luma, im_crop_bgr)

        await self.artifact_sink.emit_lazy_async(
            "Lines",
            lambda: EdgeRenderer.draw_lines_sparsebg(
                im_crop_vis_bgr,
                hands_bgr,
            ),
        )
        return boxes
