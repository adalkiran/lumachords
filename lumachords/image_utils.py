import asyncio
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lumachords.data_types import BoxIsValid
from lumachords.utils import Utils


class ImageUtils:
    @staticmethod
    def imshow(im, caption=None, filename=None, save_only=False):
        if not len(im):
            im = np.zeros((10, 10))
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if not cv2.imwrite(filename, im):
                raise Exception(f'File could not be written: "{filename}"')
        color = im.ndim > 2
        if color:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if not save_only:
            matplotlib = Utils.ensure_matplotlib()
            plt = matplotlib.pyplot
            border_color = (0.1, 0.9, 0.1)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor(border_color)
            ax.set_facecolor(border_color)

            if caption:
                mng = plt.gcf().canvas.manager
                if mng is not None:
                    mng.set_window_title(str(caption))
                if not Utils.HAS_MATPLOTLIB_GUI:
                    plt.title(str(caption))
            plt.imshow(im, cmap=None if color else "gray")
            plt.axis("off")
            plt.show()

    @staticmethod
    def read_image(image_path):
        im_bgr = cv2.imread(image_path)
        return im_bgr

    @staticmethod
    async def imsave(im: np.ndarray, file_path: str):
        if file_path.endswith(".png"):
            await asyncio.to_thread(cv2.imwrite, file_path, im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif file_path.endswith(".npy"):
            await asyncio.to_thread(np.save, file_path, im)
        else:
            await asyncio.to_thread(cv2.imwrite, file_path, im)

    @staticmethod
    def choose_contrast_color_and_pos(
        im_bgr,
        text,
        box_x0,
        box_x1,
        text_y,
        font,
        font_scale,
        thickness,
        light_color=(255, 255, 255),
        dark_color=(0, 0, 0),
    ):
        h, w = im_bgr.shape[:2]
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        box_w = box_x1 - box_x0
        x = box_x0 + (box_w - text_w) // 2
        x0 = max(x, 0)
        y0 = max(text_y - text_h, 0)
        x1 = min(x + text_w, w - 1)
        y1 = min(text_y + baseline, h - 1)

        pos = (x, y1)

        if x0 >= x1 or y0 >= y1:
            return light_color, pos

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        hw = int((x1 - x0) * 0.25 / 2)
        hh = int((y1 - y0) * 0.25 / 2)

        roi = im_bgr[cy - hh:cy + hh, cx - hw:cx + hw]
        b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
        lum = 0.114 * b + 0.587 * g + 0.299 * r
        mean_lum = float(lum.mean())

        lo, hi = float(lum.min()), float(lum.max())
        if hi - lo < 1e-3:
            color = dark_color if mean_lum >= 128 else light_color
        else:
            thresh = lo + 0.5 * (hi - lo)
            color = dark_color if mean_lum > thresh else light_color
        return color, pos

    @staticmethod
    def dashed_line(im_bgr, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA, dash=12, gap=8):
        pt1 = np.array(pt1, dtype=np.float32)
        pt2 = np.array(pt2, dtype=np.float32)
        d = pt2 - pt1
        L = float(np.hypot(d[0], d[1]))
        if L == 0:
            return
        u = d / L
        step = dash + gap
        for t in np.arange(0, L, step):
            a = pt1 + u * t
            b = pt1 + u * min(t + dash, L)
            cv2.line(im_bgr, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, lineType=lineType)

    @staticmethod
    def draw_boxes(im_bgr, boxes, color, estimated_color=None, invalid_color=None, thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        for (id, x0, y0, x1, y1, _, is_valid, _, _) in boxes:
            cur_color = color
            if is_valid == BoxIsValid.Invalid and invalid_color is not None:
                cur_color = invalid_color
            elif is_valid == BoxIsValid.EstimatedValid and estimated_color is not None:
                cur_color = estimated_color
            cv2.rectangle(im_bgr, (x0, y0), (x1, y1), cur_color, thickness)
            label = str(int(id)) if id > -1 else "INV"
            text_y = int((y1 + y0) * 0.5)
            text_color, text_pos = ImageUtils.choose_contrast_color_and_pos(
                im_bgr, label, x0, x1, text_y, font, font_scale, thickness
            )
            cv2.putText(
                im_bgr,
                label,
                text_pos,
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def print_text_on_image(
        im_bgr,
        x: int,
        y: int,
        text: str,
        font_path=None,
        font_scale=0.3,
        color=(255, 255, 255, 255),
        fill_color=None,
        pad_x=0,
        pad_y=0,
        line_spacing: float = 1.2,
    ):
        font_scale_calc = Utils.font_scale(im_bgr, font_scale)
        font_size = int(font_scale_calc * 100)

        h, w = im_bgr.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        pil_overlay = Image.fromarray(overlay, "RGBA")
        draw = ImageDraw.Draw(pil_overlay)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default(font_size)

        line_spacing_val = int(font_size * line_spacing)
        lines = text.split("\n")
        longest_line = lines[np.argmax([len(line) for line in lines])]
        _, _, tw, _ = draw.textbbox((0, 0), longest_line, font)
        y_position = y
        bounds_w = tw + 2 * pad_x
        bounds_h = len(lines) * line_spacing_val + 2 * pad_y
        bounds = (x, y, bounds_w, bounds_h)
        if fill_color is not None:
            draw.rectangle([(x, y), (x + bounds_w, y + bounds_h)], fill=fill_color, outline=color)

        for line in lines:
            if line.strip():
                draw.text((x + pad_x, y_position + pad_y), line, font=font, fill=color)
            y_position += line_spacing_val

        overlay_rgba = np.array(pil_overlay)
        im_rgba = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGBA)
        alpha = overlay_rgba[:, :, 3:4] / 255.0
        blended = (1 - alpha) * im_rgba + alpha * overlay_rgba
        result_bgr = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        return result_bgr, bounds

    @staticmethod
    async def blend_images(im_bgr, overlay_rgba):
        im_rgba = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGBA)

        if overlay_rgba.shape[2] == 3:
            overlay_rgba = np.concatenate(
                [overlay_rgba, 255 * np.ones_like(overlay_rgba[..., :1])], axis=2
            )

        h, w = im_rgba.shape[:2]
        oh, ow = overlay_rgba.shape[:2]
        blend_w = min(w, ow)
        blend_h = min(h, oh)
        if blend_w == 0 or blend_h == 0:
            return im_bgr

        base_roi = im_rgba[:blend_h, :blend_w]
        overlay_roi = overlay_rgba[:blend_h, :blend_w]
        alpha = overlay_roi[:, :, 3:4] / 255.0
        im_rgba[:blend_h, :blend_w] = (1 - alpha) * base_roi + alpha * overlay_roi
        result_bgr = cv2.cvtColor(im_rgba.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        return result_bgr
