from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import datetime

import numpy as np


class CommonVideoUtils:
    @staticmethod
    def time_to_seconds(time_str):
        if time_str is None:
            return 0
        parts = time_str.split(":")

        if len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        elif len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        else:
            raise ValueError("Invalid time format")

        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    @staticmethod
    def time_to_frame_number(time_str, fps):
        return __class__.time_to_seconds(time_str) * fps

    @staticmethod
    def frame_number_to_time(frame_number, fps):
        seconds = frame_number / fps
        datetime_obj = datetime.datetime.fromtimestamp(seconds)
        return datetime_obj.strftime("%H:%M:%S.%f")[:-3]


class BaseVideoReaderBackend(ABC):
    backend_name = "base"

    def __init__(self, fps=0, pix_fmt="bgr24", height_limit=None):
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.height_limit = height_limit
        self.actual_fps = fps

    @abstractmethod
    def load_metadata(self, video_path, fps=0):
        raise NotImplementedError

    @abstractmethod
    async def read_frames(
        self,
        video_path: str,
        stop_event: asyncio.Event,
        seek=None,
        frames=None,
        use_tqdm=True,
        preread_metadata: dict[str, any] = None,
    ):
        raise NotImplementedError


class BaseVideoWriterBackend(ABC):
    backend_name = "base"

    def __init__(self, reader_fps: int, writer_fps: int = 0, pix_fmt="bgra"):
        self.reader_fps = reader_fps
        self.writer_fps = writer_fps
        self.pix_fmt = pix_fmt
        self.writer_actual_fps = writer_fps
        self.audio_supported = False

    def _prepare_overlay_bgra(self, overlay_bgra, width: int, height: int):
        out = np.zeros((height, width, 4), dtype=np.uint8)
        if overlay_bgra is None:
            return out

        if not isinstance(overlay_bgra, np.ndarray):
            overlay_bgra = np.asarray(overlay_bgra)

        if overlay_bgra.ndim != 3 or overlay_bgra.shape[2] not in (3, 4):
            raise ValueError("overlay_bgra must be an HxWx3 or HxWx4 array")

        oh, ow = overlay_bgra.shape[:2]
        blend_h = min(height, oh)
        blend_w = min(width, ow)
        if blend_h == 0 or blend_w == 0:
            return out

        roi_out = out[:blend_h, :blend_w]

        if overlay_bgra.shape[2] == 3:
            roi = overlay_bgra[:blend_h, :blend_w]
            roi_out[..., :3] = roi
            roi_out[..., 3] = 255
        else:
            roi_out[...] = overlay_bgra[:blend_h, :blend_w]
        return out

    def convert_reader_pts_to_writer_pts(self, reader_pts: int):
        if self.reader_fps <= 0 or self.writer_actual_fps <= 0:
            return int(reader_pts)
        t_sec = (reader_pts + 1) / self.reader_fps
        writer_pts = int(t_sec * self.writer_actual_fps) - 1
        return writer_pts

    @abstractmethod
    async def write_frames(
        self,
        progress_overlay: any,
        video_path,
        out_path,
        overlay_queue: asyncio.Queue[tuple[int, np.ndarray]],
        stop_event: asyncio.Event,
        until_reader_pts: int,
        use_tqdm=True,
    ):
        raise NotImplementedError
