import asyncio

import cv2
import numpy as np

from lumachords.preferences import Preferences
from lumachords.processing_state import ProcessingState
from lumachords.image_input import NoteRainImageInput
from lumachords.data_types import AxisType, BackgroundType, as_dt_line
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.rendering import EdgeRenderer
from lumachords.artifact_sink import ArtifactSink
from lumachords.utils import Utils

class EdgeDetector:
    """Detects vertical edges (start/end boundaries) in note rain images using computer vision techniques."""

    def __init__(self, pref: Preferences, artifact_sink: ArtifactSink, keybed_output: KeybedDetectorOutput, state: ProcessingState):
        self.pref = pref
        self.artifact_sink = artifact_sink
        self.keybed_output = keybed_output
        self.state = state
        self.keybed_top_y = keybed_output.keybed_bounds[1]
        
    
    def get_gradient_mask_thresholded(self, im_crop_luma: np.ndarray, background_type: BackgroundType, axis: AxisType, lim_edge_length):
        dx, dy = (0, 1) if axis == AxisType.X else (1, 0)
        grads = cv2.Scharr(im_crop_luma, cv2.CV_32F, dx, dy)
        abs_grads = np.abs(grads)
        thr = Utils.k_sigma_threshold(abs_grads, k=self.pref.engine.gradient_threshold_standard_deviation_factor)
        if thr >= abs_grads.max():
            thr = 0.9 * float(abs_grads.max())
        min_edge_length_req = int(lim_edge_length[0])
        kernel_tickness = self.pref.engine.edge_morphology_kernel_thickness
        ksize_req = (min_edge_length_req, kernel_tickness) if axis == AxisType.X else (kernel_tickness, min_edge_length_req)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize_req)
        
        pos_mask = (grads >=  thr).astype(np.uint8) * 255  # "end" edges
        neg_mask = (grads <  -thr).astype(np.uint8) * 255  # "start" edges

        for mask_item in [pos_mask, neg_mask]:
            mask_item[:] = cv2.morphologyEx(mask_item, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_item[:] = cv2.morphologyEx(mask_item, cv2.MORPH_CLOSE, kernel, iterations=1)
    
        return pos_mask, neg_mask, grads, thr

    def filter_components(self, keep_ids, stats, labels, axis: AxisType):
        remove_ids = []
        for comp_id in keep_ids:
            x, y, w, h, _ = stats[comp_id]
            roi = (labels[y:y+h, x:x+w] == comp_id)
            ys, xs = np.where(roi)
            if xs.size < 2:
                ang = np.nan
            else:
                pts = np.column_stack((xs + x, ys + y)).astype(np.float32)
                vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                ang = float(np.degrees(np.arctan2(vy, vx)))
                ang = (ang + 180.0) % 180.0  # normalize to [0,180)
            is_vertical = np.abs(ang - 90.0) <= 5.0
            is_horizontal = np.minimum(np.abs(ang), np.abs(ang - 180.0)) <= 5.0
            has_appropriate_angle = is_horizontal if axis == AxisType.X else is_vertical
            if not has_appropriate_angle:
                remove_ids.append(comp_id)
        keep_ids = keep_ids[~np.isin(keep_ids, remove_ids)]
        return keep_ids


    def mask_to_components(self, mask, axis: AxisType, lim_edge_tickness, lim_edge_length):
        min_edge_tickness, max_edge_tickness = lim_edge_tickness
        min_edge_length, max_edge_length = lim_edge_length
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        stats1 = stats[1:]
        if axis == AxisType.Y:
            keep_ids = np.where(
                (stats1[:, cv2.CC_STAT_HEIGHT] >= min_edge_length) &
                (stats1[:, cv2.CC_STAT_WIDTH] >= min_edge_tickness) &
                (stats1[:, cv2.CC_STAT_WIDTH] <= max_edge_tickness)
            )[0] + 1
        else:
            keep_ids = np.where(
                (stats1[:, cv2.CC_STAT_WIDTH] >= min_edge_length) &
                (stats1[:, cv2.CC_STAT_WIDTH] <= max_edge_length) &
                (stats1[:, cv2.CC_STAT_HEIGHT] >= min_edge_tickness) &
                (stats1[:, cv2.CC_STAT_HEIGHT] <= max_edge_tickness)
            )[0] + 1
        if keep_ids.size == 0:
            return mask, np.zeros_like(stats), np.array([]), np.zeros_like(labels)

        keep_ids = self.filter_components(keep_ids, stats, labels, axis)
        for comp_id in range(1, num):
            if comp_id not in keep_ids:
                mask[labels == comp_id] = 0  # Artifact generate: mask tuning, only debug
                labels[labels == comp_id] = 0  # Artifact generate: labels tuning, only debug
        stats = stats[keep_ids]
        centroids = centroids[keep_ids]
        return mask, stats, centroids, labels

    def components_to_lines(self, stats, centroids, is_start: bool, axis: AxisType):
        lines = []
        for stats_item, centroid_item in zip(stats, centroids):
            if axis == AxisType.Y:
                y0 = int(stats_item[cv2.CC_STAT_TOP])
                w  = int(stats_item[cv2.CC_STAT_WIDTH])
                h  = int(stats_item[cv2.CC_STAT_HEIGHT])
                y1 = y0 + h - 1
                tickness = w
                x_pos = int(stats_item[cv2.CC_STAT_LEFT]) if is_start else int(round(centroid_item[0] - 1))
                lines.append((-1, x_pos, y0, x_pos, y1, tickness, is_start, axis))
            else:
                x0 = int(stats_item[cv2.CC_STAT_LEFT])
                w  = int(stats_item[cv2.CC_STAT_WIDTH])
                h  = int(stats_item[cv2.CC_STAT_HEIGHT])
                x1 = x0 + w - 1
                tickness = h
                y_pos = int(stats_item[cv2.CC_STAT_TOP]) if is_start else int(round(centroid_item[1] - 1))
                lines.append((-1, x0, y_pos, x1, y_pos, tickness, is_start, axis))
        return as_dt_line(lines, set_ids=True)

    def calculate_edge_limits(self, im_width):
        max_edge_tickness_rate = self.pref.engine.max_edge_tickness_rate
        lim_edge_tickness = (1, int(np.ceil(max_edge_tickness_rate * self.keybed_output.white_key_default_width)))

        min_edge_length_rate_x, max_edge_length_rate_x = self.pref.engine.min_edge_length_rate_x, self.pref.engine.max_edge_length_rate_x
        lim_edge_length_x = (int(np.ceil(min_edge_length_rate_x * im_width)), int(np.ceil(np.maximum(max_edge_length_rate_x * im_width, self.keybed_output.white_key_default_width * 2.55))))

        min_edge_length_rate_y = self.pref.engine.min_edge_length_rate_y
        lim_edge_length_y = (max(4, int(np.floor(min_edge_length_rate_y * im_width))), 10_000)
        
        return lim_edge_tickness, lim_edge_length_x, lim_edge_length_y

    async def process_axis_gradients(self, im_crop_bgr, im_crop_luma, background_type: BackgroundType, axis: AxisType, lim_edge_tickness, lim_edge_length):
        pos_mask, neg_mask, grads, thr = self.get_gradient_mask_thresholded(im_crop_luma, background_type, axis, lim_edge_length)
        pos_mask, pos_stats, pos_centroids, pos_labels = self.mask_to_components(pos_mask, axis, lim_edge_tickness, lim_edge_length)
        neg_mask, neg_stats, neg_centroids, neg_labels = self.mask_to_components(neg_mask, axis, lim_edge_tickness, lim_edge_length)
        
        pos_lines = self.components_to_lines(pos_stats, pos_centroids, True, axis)
        neg_lines = self.components_to_lines(neg_stats, neg_centroids, False, axis)
        axis_lines = np.concatenate((pos_lines, neg_lines))
        axis_lines["id"] = np.arange(len(axis_lines))
                
        im_grads_on_black, im_labels_on_black = None, None
        await self.artifact_sink.emit_lazy_async(
            f"Grads Normalized {axis}",
            lambda: cv2.normalize(grads, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        )
        im_grads_on_black = await self.artifact_sink.emit_lazy_async(
            f"Grads on Black {axis}",
            lambda: EdgeRenderer.visualize_masks(pos_mask, neg_mask, grads),
            return_value=True,
        )
        await self.artifact_sink.emit_lazy_async(
            f"Grads on Image {axis}",
            lambda: EdgeRenderer.visualize_combine_grads(im_grads_on_black, im_crop_bgr),
        )
        self.artifact_sink.emit(
            f"Positive Mask {axis}",
            pos_mask,
        )
        self.artifact_sink.emit(
            f"Negative Mask {axis}",
            neg_mask,
        )
        im_labels_on_black = await self.artifact_sink.emit_lazy_async(
            f"Labels {axis}",
            lambda: EdgeRenderer.visualize_combine_masks(pos_labels, neg_labels, im_crop_bgr),
            return_value=True,
        )
        if self.artifact_sink.wants("Component"):
            pos_grads_scaled = cv2.convertScaleAbs(np.where(grads >= thr, grads, 0))
            neg_grads_scaled = cv2.convertScaleAbs(np.where(grads < -thr, grads, 0))
            for im_combined, caption in EdgeRenderer.visualize_components("Start Line", pos_stats, pos_labels, pos_mask, pos_grads_scaled, im_crop_bgr, im_grads_on_black):
                self.artifact_sink.emit("Component", im_combined, caption=caption)
            for im_combined, caption in EdgeRenderer.visualize_components("End Line", neg_stats, neg_labels, neg_mask, neg_grads_scaled, im_crop_bgr, im_grads_on_black):
                self.artifact_sink.emit("Component", im_combined, caption=caption)
        return axis_lines, im_grads_on_black, im_labels_on_black


    async def detect_lines(self, nr_image_input: NoteRainImageInput, hands_bgr: np.array):
        im_crop_bgr, im_crop_luma, (background_type, _) = nr_image_input.im_crop_bgr, nr_image_input.im_crop_luma, nr_image_input.background_info
        # ======= PREPROCESSING =======
        # Artifact: Cropped luma (VERBOSE)
        self.artifact_sink.emit("Crop Luma", im_crop_luma)
        
        lim_edge_tickness, lim_edge_length_x, lim_edge_length_y = self.calculate_edge_limits(im_crop_bgr.shape[1])
        (
            (lines_y, im_grads_on_black_y, im_labels_on_black_y),
            (lines_x, im_grads_on_black_x, im_labels_on_black_x)
        ) = await asyncio.gather(
            self.process_axis_gradients(im_crop_bgr, im_crop_luma, background_type, AxisType.Y, lim_edge_tickness, lim_edge_length_y),
            self.process_axis_gradients(im_crop_bgr, im_crop_luma, background_type, AxisType.X, lim_edge_tickness, lim_edge_length_x)
        )
        lines_all = np.concatenate((lines_y, lines_x))
        if len(lines_all) > 0:
            lines_all = lines_all[np.lexsort((lines_all["y0"], lines_all["x0"]))]
            lines_all["id"] = np.arange(len(lines_all))
        await self.artifact_sink.emit_lazy_async(
            "Grads on Black Combined",
            lambda: EdgeRenderer.visualize_combine_masks(im_grads_on_black_y, im_grads_on_black_x, im_crop_bgr),
        )
        await self.artifact_sink.emit_lazy_async(
            "Labels Combined",
            lambda: EdgeRenderer.visualize_combine_images(im_labels_on_black_y, im_labels_on_black_x, im_crop_bgr),
        )
        await self.artifact_sink.emit_lazy_async(
            "Lines",
            lambda: EdgeRenderer.draw_lines_texturedbg(
                nr_image_input.im_bgr.copy(),
                hands_bgr,
                self.keybed_top_y,
                lines_all,
                self.pref.appearance.start_line_color_bgr,
                self.pref.appearance.end_line_color_bgr,
            )
        )
        return lines_all, lim_edge_tickness, im_crop_bgr.shape
