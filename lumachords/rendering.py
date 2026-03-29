import cv2
import numpy as np

from lumachords.fft_analyzer import FFTAnalyzerOutput
from lumachords.image_utils import ImageUtils
from lumachords.preferences import Preferences
from lumachords.utils import Utils


class EdgeRenderer:
    @staticmethod
    def draw_lines_texturedbg(vis, hands_bgr: np.ndarray, keybed_top_y: int, lines, start_line_color, end_line_color):
        if hands_bgr is not None and keybed_top_y is not None:
            vis[keybed_top_y:keybed_top_y + hands_bgr.shape[0], :hands_bgr.shape[1], :] = hands_bgr
        for line in lines:
            _, line_x0, line_y0, line_x1, line_y1, _, line_is_start, _ = line
            color = start_line_color if line_is_start else end_line_color
            cv2.line(vis, (line_x0, line_y0), (line_x1, line_y1), color, 1)
            text = f'{"start" if line_is_start else "end"} ({line_x0}, {line_y0}, {line_x1}, {line_y1})'
            cv2.putText(vis, text, (line_x0 + 3, max(30, line_y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return vis
    
    @staticmethod
    def draw_lines_sparsebg(crop_vis, hands_bgr: np.ndarray):
        vis = np.vstack((crop_vis, hands_bgr)) if hands_bgr is not None else crop_vis
        return vis

    @staticmethod
    def visualize_masks(pos_mask, neg_mask, grads):
        im_grads_pos = np.zeros_like(pos_mask, dtype="uint8")
        im_grads_neg = np.zeros_like(neg_mask, dtype="uint8")
        for mask_item, im_grads_item in [(pos_mask, im_grads_pos), (neg_mask, im_grads_neg)]:
            if not (len(mask_item) and len(grads)):
                continue
            im_grads_item[:] = cv2.convertScaleAbs(np.where(mask_item, grads, 0))
        im_grads = np.stack([np.zeros_like(im_grads_neg), im_grads_pos, im_grads_neg], axis=-1)
        return im_grads

    @staticmethod
    def visualize_combine_masks(pos_labels, neg_labels, im_crop_bgr):
        if pos_labels is None and neg_labels is None:
            return None
        im_labels = np.where(cv2.bitwise_or(pos_labels, neg_labels) > 0, 255, 0)
        return __class__.visualize_combine_grads(im_labels, im_crop_bgr)

    @staticmethod
    def visualize_combine_images(pos_labels, neg_labels, im_crop_bgr):
        if pos_labels is None and neg_labels is None:
            return None
        im_labels = cv2.bitwise_or(pos_labels, neg_labels)
        return __class__.visualize_combine_grads(im_labels, im_crop_bgr)

    @staticmethod
    def visualize_combine_grads(im_grads, im_crop_bgr):
        if im_grads is not None and len(im_grads):
            combine_mask = ((im_grads >= 128).astype(np.uint8) * 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            combine_mask = cv2.dilate(combine_mask, kernel, iterations=1)
            if combine_mask.ndim == 2:
                combine_mask = np.repeat(combine_mask[..., None], 3, axis=2)
        else:
            combine_mask = np.zeros_like(im_crop_bgr)
        im_grads_combined = im_crop_bgr | combine_mask
        return im_grads_combined

    @staticmethod
    def visualize_components(caption_prefix: str, stats, labels, mask, grads_scaled, im, im_grads):
        H, W = labels.shape
        comp_ids = np.unique_values(labels)
        comp_ids.sort()
        comp_ids = comp_ids[1:]
        pad = 40
        for i, comp_id in enumerate(comp_ids):
            x, y, w, h, _ = stats[i]
            roi = ((labels[y:y+h, x:x+w] == comp_id) * 255).astype("uint8")
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

            dx0 = min(pad, x)
            dy0 = min(pad, y)
            dx1 = min(pad, W - x+w)
            dy1 = min(pad, H - y+h)

            im_crop = im[y - dy0 : y+h + dy1, x - dx0 : x+w + dx1]
            im_roi = cv2.copyMakeBorder(
                im_crop,
                top   = pad - dy0,
                bottom= pad - dy1,
                left  = pad - dx0,
                right = pad - dx1,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            grads_scaled_crop = grads_scaled[y - dy0 : y+h + dy1, x - dx0 : x+w + dx1]
            grads_scaled_roi = cv2.copyMakeBorder(
                grads_scaled_crop,
                top   = pad - dy0,
                bottom= pad - dy1,
                left  = pad - dx0,
                right = pad - dx1,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            im_grads_crop = im_grads[y - dy0 : y+h + dy1, x - dx0 : x+w + dx1]
            im_grads_roi = cv2.copyMakeBorder(
                im_grads_crop,
                top   = pad - dy0,
                bottom= pad - dy1,
                left  = pad - dx0,
                right = pad - dx1,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            mask_crop = mask[y - dy0 : y+h + dy1, x - dx0 : x+w + dx1]
            mask_roi = cv2.copyMakeBorder(
                mask_crop,
                top   = pad - dy0,
                bottom= pad - dy1,
                left  = pad - dx0,
                right = pad - dx1,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            sep = np.ones((roi.shape[0], 2, 3), dtype=roi.dtype) * 255

            try:
                im_combined = np.hstack((
                    np.repeat(roi[..., None], 3, axis=2),
                    sep,
                    np.repeat(mask_roi[..., None], 3, axis=2),
                    sep,
                    np.repeat(grads_scaled_roi[..., None], 3, axis=2),
                    sep,
                    im_grads_roi,
                    sep,
                    im_roi
                ))
                yield (im_combined, f"{(caption_prefix + ', ') if caption_prefix else ""}id: {comp_id}, bounds: {(int(x), int(y), int(w), int(h))}")
            except Exception:
                pass


class NoteRainRenderer:
    @staticmethod
    def render_pipeline_overlay(im_vis_bgr, boxes, obstacle_lines, tracking_y_band_tuple, play_y, keybed_bounds, pref, draw_box_tickness):
        keybed_left_x, keybed_top_y, keybed_width, keybed_height = keybed_bounds
        if tracking_y_band_tuple is not None:
            ImageUtils.dashed_line(im_vis_bgr, (0, tracking_y_band_tuple[0]), (im_vis_bgr.shape[1], tracking_y_band_tuple[0]), pref.appearance.tracking_band_color_bgr, 3)
            ImageUtils.dashed_line(im_vis_bgr, (0, tracking_y_band_tuple[1]), (im_vis_bgr.shape[1], tracking_y_band_tuple[1]), pref.appearance.tracking_band_color_bgr, 3)
        if play_y is not None:
            cv2.rectangle(im_vis_bgr, (keybed_left_x, keybed_top_y), (keybed_left_x + keybed_width, keybed_top_y + keybed_height), pref.appearance.keybed_border_color_bgr, 10)
            cv2.line(im_vis_bgr, (0, play_y), (im_vis_bgr.shape[1], play_y), pref.appearance.play_edge_color_bgr, 3)

        ImageUtils.draw_boxes(
            im_vis_bgr,
            boxes,
            pref.appearance.floating_box_border_color_bgr,
            estimated_color=pref.appearance.floating_estimated_box_border_color_bgr,
            invalid_color=pref.appearance.floating_invalid_box_border_color_bgr,
            thickness=draw_box_tickness,
        )

        for line in obstacle_lines:
            _, line_x0, line_y0, line_x1, line_y1, _, line_is_start, _ = line.item()
            color = Utils.color_to_bgr((255, 255, 0)) if line_is_start else Utils.color_to_bgr((255, 0, 255))
            cv2.line(im_vis_bgr, (line_x0, line_y0), (line_x1, line_y1), color, 3)
        return im_vis_bgr


class KeybedRenderer:
    @staticmethod
    def visualize_fft_analysis(im_gray, fft_analyzer_output: FFTAnalyzerOutput, im_extra=None, return_as_image=False):
        fft_heatmap_log = np.log1p(fft_analyzer_output.fft_magnitudes_no_dc)
        dpi = 100
        matplotlib = Utils.ensure_matplotlib()
        plt = matplotlib.pyplot
        Figure = matplotlib.figure.Figure
        FigureCanvas = matplotlib.backends.backend_agg.FigureCanvasAgg
        fig = Figure(figsize=(im_gray.shape[1] / dpi, im_gray.shape[0] / dpi), dpi=dpi)
        _ = FigureCanvas(fig) # This must be called, even it's not used.
        plot_axes = fig.subplots(1, 2)
        # Left: grayscale + mask
        plot_axes[0].imshow(
            np.concatenate((
                (im_extra if im_extra is not None else np.zeros((im_gray.shape[0], 0), dtype=im_gray.dtype)) * 255, 
                im_gray
                ), axis=1),
            cmap="gray", aspect="auto", origin="upper"
        )
        plot_axes[0].set_title("Row Mask + Best Mask + Grayscale Image")
        plot_axes[0].set_xlabel("X (columns)")
        plot_axes[0].set_ylabel("Y (rows)")

        # Right: FFT heatmap + overlays
        im = plot_axes[1].imshow(fft_heatmap_log, aspect="auto", origin="upper")
        plot_axes[1].set_title("Per-row rFFT magnitude (log1p) with Overlays")
        plot_axes[1].set_xlabel("Spatial frequency (cycles/pixel)")
        plot_axes[1].set_ylabel("Row (top→bottom)")

        xticks = np.linspace(0, fft_heatmap_log.shape[1]-1, 6).astype(int)
        plot_axes[1].set_xticks(xticks)
        plot_axes[1].set_xticklabels([f"{fft_analyzer_output.frequencies[i]:.3f}" for i in xticks])

        dominant_freq_xcoords = fft_analyzer_output.dominant_freq_indices
        plot_axes[1].plot(dominant_freq_xcoords, np.arange(fft_analyzer_output.num_rows), color="red", linewidth=1.5, label="Dominant freq")

        peakiness_ratio_max = fft_analyzer_output.row_peakiness_ratios.max()
        peakiness_ratio_xcoords = np.divide(
            fft_analyzer_output.row_peakiness_ratios,
            peakiness_ratio_max,
            out=np.zeros_like(fft_analyzer_output.row_peakiness_ratios),
            where=peakiness_ratio_max,
        ) * (fft_heatmap_log.shape[1]-1)
        plot_axes[1].plot(peakiness_ratio_xcoords, np.arange(fft_analyzer_output.num_rows), color="white", linewidth=1.2, alpha=0.9, label="Peakiness ratio")

        peakiness_top_axis = plot_axes[1].twiny()
        peakiness_top_axis.set_xlim(plot_axes[1].get_xlim())
        pk_ticks = np.linspace(0, 1, 6)
        peakiness_top_axis.set_xticks(pk_ticks * (fft_heatmap_log.shape[1]-1))
        peakiness_top_axis.set_xticklabels([f"{t:.2f}" for t in pk_ticks])
        peakiness_top_axis.set_xlabel("Peakiness ratio = max(non-DC)/sum(non-DC)")

        fig.colorbar(im, ax=plot_axes[1], shrink=0.7, label="log1p(|FFT|)")
        plot_axes[1].legend(loc="lower right")
        fig.tight_layout()
        if return_as_image:
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[:,:,:3]
            plt.close()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            plt.show()
            return None

    @staticmethod
    def render_keybed_detection(im_vis_bgr, keybed_bounds, pref: Preferences, is_validated):
        if keybed_bounds is not None:
            keybed_left_x, keybed_top_y, keybed_width, keybed_height = keybed_bounds
            keybed_border_color_bgr = pref.appearance.keybed_border_color_bgr if is_validated else pref.appearance.keybed_border_unvalidated_color_bgr
            cv2.rectangle(im_vis_bgr, (keybed_left_x, keybed_top_y), (keybed_left_x + keybed_width, keybed_top_y + keybed_height), keybed_border_color_bgr, 10)
        return im_vis_bgr

    @staticmethod
    def render_fft_analysis(row_keybed_mask_cpy, keybed_mask, im_gray, fft_analyzer_output):
        keybed_mask_vis = np.hstack((
            np.repeat(row_keybed_mask_cpy, im_gray.shape[1]//2, axis=1),
            keybed_mask[:, :im_gray.shape[1]//2] if keybed_mask is not None else np.zeros((im_gray.shape[0], im_gray.shape[1]//2))
        ))
        im_vis_fft = __class__.visualize_fft_analysis(im_gray, fft_analyzer_output, keybed_mask_vis, return_as_image=True)
        return im_vis_fft

    @staticmethod
    def render_white_key_edges(im_vis_bgr: np.ndarray, boundary_edges_x, keybed_bounds, pref):
        _, keybed_top_y, _, keybed_height = keybed_bounds
        for x in boundary_edges_x:
            cv2.line(im_vis_bgr, (x, 0), (x, keybed_top_y + keybed_height), pref.appearance.white_key_line_color_bgr, 1)

    @staticmethod
    def render_white_key_text(im_vis_bgr: np.ndarray, white_keys_corrected_x, keybed_bounds, pref):
        _, keybed_top_y, _, keybed_height = keybed_bounds
        for i, x in enumerate(white_keys_corrected_x):
            cv2.putText(im_vis_bgr, str(i), (x, keybed_top_y + keybed_height - 60), cv2.FONT_HERSHEY_SIMPLEX,
                        Utils.font_scale(im_vis_bgr, 0.7), pref.appearance.note_text_color_bgr, 1, cv2.LINE_AA)
            cv2.putText(im_vis_bgr, str(x), (x, keybed_top_y + keybed_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        Utils.font_scale(im_vis_bgr, 0.5), pref.appearance.note_text_color_bgr, 1, cv2.LINE_AA)

    @staticmethod
    def render_keys(im_vis_bgr: np.ndarray, im_sobel_x_bgr: np.ndarray, keybed_output, pref: Preferences):
        _, keybed_top_y, _, keybed_height = keybed_output.keybed_bounds

        if im_sobel_x_bgr is not None:
            cv2.rectangle(im_sobel_x_bgr, (0, keybed_top_y), (im_sobel_x_bgr.shape[1], keybed_top_y + int(keybed_height * 0.66)), pref.appearance.black_key_line_color_bgr, 5)
            cv2.putText(im_sobel_x_bgr, "Black key edges", (40, keybed_top_y + 80), cv2.FONT_HERSHEY_SIMPLEX,
                        Utils.font_scale(im_sobel_x_bgr, 1.5), pref.appearance.note_text_color_bgr, 2, cv2.LINE_AA)

        for key_idx, (key_data, key_edge) in enumerate(zip(keybed_output.all_keys_data, keybed_output.all_keys_edge)):
            if key_data["color"] == "w":
                cv2.line(im_vis_bgr, (key_edge[0], 0), (key_edge[0], keybed_top_y + keybed_height), pref.appearance.white_key_line_color_bgr, (5 if key_data["is_full"] else 2))
            else:
                cv2.line(im_vis_bgr, (key_edge[0], 0), (key_edge[0], keybed_top_y + keybed_height), pref.appearance.black_key_line_color_bgr, 2)
                cv2.line(im_vis_bgr, (key_edge[1], 0), (key_edge[1], keybed_top_y + keybed_height), pref.appearance.black_key_line_color_bgr, 2)

                if im_sobel_x_bgr is not None:
                    cv2.line(im_sobel_x_bgr, (key_edge[0], 0), (key_edge[0], keybed_top_y), pref.appearance.black_key_line_color_bgr, 2)
                    cv2.line(im_sobel_x_bgr, (key_edge[1], 0), (key_edge[1], keybed_top_y), pref.appearance.black_key_line_color_bgr, 2)

            cv2.putText(im_vis_bgr, str(key_data["note_name"]), (key_edge[0], (keybed_top_y + keybed_height - 20) if key_data["color"] == "w" else (keybed_top_y + 40)), cv2.FONT_HERSHEY_SIMPLEX,
                        Utils.font_scale(im_vis_bgr, 1), pref.appearance.note_text_color_bgr, 2, cv2.LINE_AA)
            if key_data["color"] == "w":
                cv2.putText(im_vis_bgr, str(key_idx), (key_edge[0], keybed_top_y + keybed_height - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            Utils.font_scale(im_vis_bgr, 0.7), pref.appearance.note_text_color_bgr, 1, cv2.LINE_AA)

class TrackerRenderer:
    @staticmethod
    def track_boxes_once(prev_boxes, cur_boxes, im_vis, match_boxes):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        prev_boxes = prev_boxes[["x0", "y0", "x1", "y1"]]
        cur_boxes = cur_boxes[["x0", "y0", "x1", "y1"]]

        prev_cx = (prev_boxes["x0"] + prev_boxes["x1"]) * 0.5
        prev_cy = (prev_boxes["y0"] + prev_boxes["y1"]) * 0.5
        cur_cx = (cur_boxes["x0"] + cur_boxes["x1"]) * 0.5
        cur_cy = (cur_boxes["y0"] + cur_boxes["y1"]) * 0.5

        prev2cur, cur2prev = match_boxes(prev_cx, prev_cy, cur_cx, cur_cy, 4, 50)
        prev_cx, prev_cy, cur_cx, cur_cy = prev_cx.astype(int), prev_cy.astype(int), cur_cx.astype(int), cur_cy.astype(int)

        prev_only_indices = np.flatnonzero(prev2cur < 0)
        cur_only_indices = np.flatnonzero(cur2prev < 0)

        cand_prev = np.flatnonzero(prev2cur >= 0)
        cand_cur = prev2cur[cand_prev]
        valid = cur2prev[cand_cur] == cand_prev

        matched_prev = cand_prev[valid]
        matched_cur = cand_cur[valid]
        matched_pairs = np.stack([matched_prev, matched_cur], axis=1)

        for i in prev_only_indices:
            x0, y0, x1, y1 = prev_boxes[i]
            cv2.rectangle(im_vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cx, cy = prev_cx[i], prev_cy[i]
            cv2.putText(im_vis, "P", (cx, cy),
                        font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        for j in cur_only_indices:
            x0, y0, x1, y1 = cur_boxes[j]
            cv2.rectangle(im_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cx, cy = cur_cx[j], cur_cy[j]
            cv2.putText(im_vis, "N", (cx, cy),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        for pair_id, (prev_i, cur_i) in enumerate(matched_pairs, start=1):
            r_prev = prev_boxes[prev_i]
            r_cur = cur_boxes[cur_i]

            cv2.rectangle(im_vis, (r_prev[0], r_prev[1]), (r_prev[2], r_prev[3]), (255, 0, 0), 1)
            cv2.rectangle(im_vis, (r_cur[0], r_cur[1]), (r_cur[2], r_cur[3]), (0, 255, 255), 1)

            cx_prev, cy_prev = prev_cx[prev_i], prev_cy[prev_i]
            cx_cur, cy_cur = cur_cx[cur_i], cur_cy[cur_i]

            cv2.line(im_vis, (cx_prev, cy_prev), (cx_cur, cy_cur), (255, 255, 0), 1)

            label = str(pair_id)
            cv2.putText(im_vis, label, (cx_prev, cy_prev),
                        font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(im_vis, label, (cx_cur, cy_cur),
                        font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        velocities = cur_cy[matched_pairs[:, 1]] - prev_cy[matched_pairs[:, 0]]
        velocities_dict = {pair_id: int(vel) for pair_id, vel in enumerate(velocities, start=1)}
        cv2.putText(im_vis, f"Velocities: {velocities_dict}", (30, 30),
                    font, font_scale * 1.5, (255, 32, 32), 4, cv2.LINE_AA)
        return im_vis
