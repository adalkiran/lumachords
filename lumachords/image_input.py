from dataclasses import dataclass
import cv2
import numpy as np

from lumachords.data_types import BackgroundType


@dataclass(frozen=True)
class KeybedImageInput:
    im_bgr: np.ndarray
    im_gray: np.ndarray

@dataclass(frozen=True)
class KeybedInternalImageInput:
    im_bgr: np.ndarray
    im_gray: np.ndarray
    keybed_edges_parts: list[np.ndarray]


@dataclass(frozen=True)
class NoteRainImageInput:
    im_bgr: np.ndarray
    im_crop_bgr: np.ndarray
    im_crop_luma: np.ndarray
    im_crop_keybed_bgr: np.ndarray
    im_crop_keybed_luma: np.ndarray
    crop_keybed_extra_height: int
    background_info: tuple[BackgroundType, int]


class ImagePreprocessor:
    @staticmethod
    async def preprocess_for_keybed(im_bgr: np.ndarray) -> KeybedImageInput:
        im_gray = ImageTransforms.make_im_gray(im_bgr)
        result = KeybedImageInput(im_bgr, im_gray)
        return result

    @staticmethod
    async def preprocess_for_keybed_internal(kb_image_input: KeybedImageInput, keybed_bounds: tuple[int, int, int, int]) -> KeybedInternalImageInput:
        im_gray = kb_image_input.im_gray
        im_crop_gray = ImageTransforms.crop_im_gray(im_gray, keybed_bounds)
        im_gray_normalized = ImageTransforms.make_im_gray_normalized(im_gray)
        im_crop_gray_normalized = ImageTransforms.crop_im_gray(im_gray_normalized, keybed_bounds)
        keybed_height = im_crop_gray.shape[0]
        
        im_crop_gray_part1_normalized = im_crop_gray_normalized[:int(0.33 * keybed_height)] # Important: input is the normalized one
        im_crop_gray_part2 = im_crop_gray[int(0.66 * keybed_height):]
        im_crop_gray_part1_sobel_x_normalized = ImageTransforms.make_im_sobel_x(im_crop_gray_part1_normalized) # Important: input is the normalized one
        im_crop_gray_part2_sobel_x = ImageTransforms.make_im_sobel_x(im_crop_gray_part2)

        keybed_edges_parts = [
            im_crop_gray_part1_sobel_x_normalized,
            im_crop_gray_part2_sobel_x,
        ]
        result = KeybedInternalImageInput(kb_image_input.im_bgr, im_gray, keybed_edges_parts)
        return result

    @staticmethod
    async def preprocess_for_note_rain(im_bgr: np.ndarray, keybed_bounds, known_background_info: tuple[BackgroundType, int]) -> NoteRainImageInput:
        _, keybed_top_y, _, _ = keybed_bounds
        im_crop_bgr, im_crop_luma, background_info = ImageTransforms.make_crop_im_luma(im_bgr, keybed_top_y, known_background_info)
        im_crop_keybed_bgr, im_crop_keybed_luma, crop_keybed_extra_height = ImageTransforms.make_crop_im_keybed(im_bgr, keybed_bounds)
        result = NoteRainImageInput(im_bgr, im_crop_bgr, im_crop_luma, im_crop_keybed_bgr, im_crop_keybed_luma, crop_keybed_extra_height, background_info)
        return result

class ImageTransforms:
    @staticmethod
    def local_contrast_norm(im_gray, win_px=None):
        g = im_gray.astype(np.float32)
        R = int(round(min(g.shape)/20)) if win_px is None else int(win_px)
        R = max(7, R|1)  # odd
        mu = cv2.GaussianBlur(g, (R, R), 0)
        var = cv2.GaussianBlur((g-mu)**2, (R, R), 0)
        std = np.sqrt(var) + 1e-6
        z = (g - mu) / std
        z = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return z

    @staticmethod
    def crop_im_gray(im_gray: np.ndarray, keybed_bounds):
        keybed_left_x, keybed_top_y, keybed_width, keybed_height = keybed_bounds
        im_crop_gray = im_gray[keybed_top_y:keybed_top_y+keybed_height+1, keybed_left_x:keybed_width+1]
        return im_crop_gray

    @staticmethod
    def analyze_background_regions(im_crop_luma: np.ndarray) -> tuple[BackgroundType, int]:
        if im_crop_luma.size == 0:
            return (BackgroundType.TEXTURED, None)
        hist = np.bincount(im_crop_luma.reshape(-1), minlength=256)
        hist_low = hist[:128]
        top_k = min(100, hist_low.size)
        top_indices = np.argpartition(hist_low, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(hist_low[top_indices])[::-1]]
        top_densities = hist_low[top_indices].astype(np.float64) / im_crop_luma.size
        log_top_densities = np.log1p(top_densities)
        keep_count = top_indices.size


        if log_top_densities.size > 1:
            log_gaps = log_top_densities[:-1] - log_top_densities[1:]
            split_idx = int(np.argmax(log_gaps))
            if (log_gaps.max() - log_gaps.min()) > np.std(log_top_densities):
                keep_count = split_idx + 1
        dominant_background_value = int(np.max(top_indices[:keep_count]))

        bg_tolerance = 2
        bg_mask = ((im_crop_luma.astype(np.int16) - dominant_background_value) <= bg_tolerance).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)
        selected_area = 0
        if num_labels > 1:
            min_component_area = int(bg_mask.size * 0.1)
            keep_ids = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_component_area) + 1
            if keep_ids.size:
                max_component_std = 12.0
                consistent_ids = []
                for component_id in keep_ids:
                    x = int(stats[component_id, cv2.CC_STAT_LEFT])
                    y = int(stats[component_id, cv2.CC_STAT_TOP])
                    w = int(stats[component_id, cv2.CC_STAT_WIDTH])
                    h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
                    label_roi = labels[y:y+h, x:x+w]
                    luma_roi = im_crop_luma[y:y+h, x:x+w]
                    component_pixels = luma_roi[label_roi == component_id].astype(np.float32)
                    if component_pixels.size == 0:
                        continue
                    component_std = float(component_pixels.std())
                    if component_std <= max_component_std:
                        consistent_ids.append(component_id)
                if consistent_ids:
                    selected_area = int(stats[np.asarray(consistent_ids, dtype=np.int32), cv2.CC_STAT_AREA].sum())
        largest_ratio = selected_area / float(bg_mask.size)

        min_sparse_ratio = 0.4
        if largest_ratio > min_sparse_ratio:
            robust_count_floor = max(1, int(hist_low.max() * 0.1))
            robust_bins = np.flatnonzero(hist_low >= robust_count_floor)
            robust_background_value = int(robust_bins.max()) if robust_bins.size else dominant_background_value
            return (BackgroundType.SPARSE, robust_background_value)
        return (BackgroundType.TEXTURED, None)

    @staticmethod
    def make_im_luma(im_bgr: np.ndarray):
        lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
        im_luma = lab[:,:,0]
        im_luma = cv2.GaussianBlur(im_luma, (5, 5), 0)
        return im_luma

    @staticmethod
    def make_crop_im_luma(im_bgr: np.ndarray, keybed_top_y: int, known_background_info: tuple[BackgroundType, int]) -> tuple[np.ndarray, np.ndarray, tuple[BackgroundType, int]]:
        im_crop_bgr = im_bgr[:keybed_top_y, :, :]
        im_crop_luma = __class__.make_im_luma(im_crop_bgr)
        background_type, dominant_background_value = known_background_info or (None, None)
        if background_type is None:
            background_info = __class__.analyze_background_regions(im_crop_luma)
            background_type, dominant_background_value = background_info
        else:
            background_info = known_background_info
        if background_type == BackgroundType.SPARSE:
            cutoff = dominant_background_value + 8
            cv2.threshold(im_crop_luma, cutoff, 255, cv2.THRESH_TOZERO, dst=im_crop_luma)
        return im_crop_bgr, im_crop_luma, background_info

    @staticmethod
    def make_crop_im_keybed(im_bgr: np.ndarray, keybed_bounds) -> tuple[np.ndarray, np.ndarray]:
        keybed_left_x, keybed_top_y, keybed_width, keybed_height = keybed_bounds
        crop_bottom_y = int(np.minimum(keybed_top_y+keybed_height * 1.5, im_bgr.shape[0]))
        crop_keybed_extra_height = crop_bottom_y - keybed_top_y - keybed_height
        im_crop_bgr = im_bgr[keybed_top_y:crop_bottom_y, keybed_left_x:keybed_width+1, :]
        im_crop_luma = __class__.make_im_luma(im_crop_bgr)
        return im_crop_bgr, im_crop_luma, crop_keybed_extra_height

    @staticmethod
    def make_im_gray(im_bgr: np.ndarray) -> np.ndarray:
        im_gray  = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        # light contrast boost helps across footage
        clahe = cv2.createCLAHE(12.0, (8,8))
        im_gray = clahe.apply(im_gray)
        return im_gray

    @staticmethod
    def make_im_gray_normalized(im_gray: np.ndarray) -> np.ndarray:
        im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)
        im_gray_normalized = __class__.local_contrast_norm(im_gray)
        return im_gray_normalized

    @staticmethod
    def make_im_sobel_x(im_gray: np.ndarray) -> np.ndarray:
        if not len(im_gray):
            return im_gray
        im_sobel_x = cv2.Sobel(im_gray, cv2.CV_32F, 1, 0, ksize=3)
        im_sobel_x = (np.abs(im_sobel_x) - im_sobel_x.min()) / (np.ptp(im_sobel_x) + 1e-6)
        return im_sobel_x
