from __future__ import annotations

import asyncio
import math
import os

import cv2
import numpy as np
from tqdm import tqdm

from .base_video_backends import BaseVideoReaderBackend, BaseVideoWriterBackend, CommonVideoUtils


class OpenCvVideoUtils:
    @staticmethod
    def load_metadata_opencv(
        video_path: str,
        fps: int = 0,
        height_limit: int | None = None,
        default_fps: int = 30,
    ):
        try:
            if video_path is None or not len(video_path):
                raise Exception("Invalid input_video path value.")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise Exception(f"Input file could not be opened with OpenCV:\n{video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            file_fps = int(cap.get(cv2.CAP_PROP_FPS) or 0)
            file_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            actual_fps = int(fps) if fps and fps > 0 else file_fps
            if actual_fps <= 0:
                actual_fps = default_fps

            if height_limit and height > height_limit and height > 0:
                width = int(height_limit * width / height)
                height = int(height_limit)

            total_duration = (file_frame_count / file_fps) if file_frame_count > 0 and file_fps > 0 else 0.0
            frame_count = math.ceil(total_duration * actual_fps)
            return {
                "width": width,
                "height": height,
                "file_fps": file_fps if file_fps > 0 else actual_fps,
                "actual_fps": actual_fps,
                "total_duration": total_duration,
                "frame_count": frame_count,
            }, None
        except Exception as e:
            return None, e


class OpenCvVideoReaderBackend(BaseVideoReaderBackend):
    backend_name = "opencv"

    def load_metadata(self, video_path, fps=0):
        metadata, err = OpenCvVideoUtils.load_metadata_opencv(video_path, fps=fps, height_limit=self.height_limit)
        if metadata is not None:
            self.actual_fps = metadata["actual_fps"]
        return metadata, err

    async def read_frames(
        self,
        video_path: str,
        stop_event: asyncio.Event,
        seek=None,
        frames=None,
        use_tqdm=True,
        preread_metadata: dict[str, any] = None,
    ):
        if stop_event is None:
            stop_event = asyncio.Event()
        if stop_event.is_set():
            return

        metadata, err = (
            (preread_metadata, None)
            if preread_metadata is not None
            else self.load_metadata(video_path, self.fps)
        )
        if metadata is None:
            raise Exception(err)

        self.actual_fps = metadata["actual_fps"]
        file_fps = metadata["file_fps"] if metadata["file_fps"] > 0 else self.actual_fps
        start_pts = 0
        end_pts = None
        if metadata["frame_count"] > 0 and file_fps > 0:
            end_pts = metadata["frame_count"]

        if seek:
            start_pts = math.ceil(CommonVideoUtils.time_to_frame_number(seek, self.actual_fps))
        if frames is not None:
            end_pts = start_pts + frames

        metadata["start_pts"] = start_pts
        metadata["end_pts"] = end_pts
        metadata["total_duration"] = math.ceil(metadata["total_duration"])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise Exception(f"Input file could not be opened with OpenCV:\n{video_path}")

        start_src_idx = (start_pts * file_fps + self.actual_fps // 2) // self.actual_fps
        if start_src_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_src_idx)

        progress_bar = None
        try:
            total_frame_count = None
            if end_pts is not None:
                total_frame_count = max(end_pts - start_pts, 0)
            progress_bar = tqdm(total=total_frame_count) if use_tqdm and total_frame_count is not None else None

            current_pts = start_pts
            src_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or start_src_idx)
            while not stop_event.is_set() and (end_pts is None or current_pts < end_pts):
                target_src_idx = (current_pts * file_fps + self.actual_fps // 2) // self.actual_fps

                while src_idx < target_src_idx:
                    if not cap.grab():
                        return
                    src_idx += 1

                ret, frame = cap.read()
                if not ret:
                    break
                src_idx += 1

                if self.height_limit and frame.shape[0] > self.height_limit:
                    new_h = self.height_limit
                    new_w = int(new_h * frame.shape[1] / frame.shape[0])
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if progress_bar:
                    progress_bar.update(1)
                    if end_pts is not None:
                        progress_bar.set_postfix(frame=f"{current_pts + 1}/{end_pts}")

                current_secs = current_pts / self.actual_fps if self.actual_fps > 0 else 0.0
                hh = int(current_secs // 3600)
                mm = int((current_secs % 3600) // 60)
                ss = current_secs - (hh * 3600 + mm * 60)
                meta_dict = {
                    "time": f"{hh:02d}:{mm:02d}:{ss:06.3f}",
                    "bitrate": None,
                    "speed": None,
                }
                yield current_pts, meta_dict, frame
                current_pts += 1
        finally:
            if progress_bar:
                progress_bar.close()
            cap.release()


class OpenCvVideoWriterBackend(BaseVideoWriterBackend):
    backend_name = "opencv"

    def __init__(self, reader_fps: int, writer_fps: int = 0, pix_fmt="bgra"):
        super().__init__(reader_fps, writer_fps=writer_fps, pix_fmt=pix_fmt)
        self.audio_supported = False

    def _alpha_blend_overlay(self, base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
        alpha = overlay_bgra[..., 3:4].astype(np.float32) / 255.0
        blended = overlay_bgra[..., :3].astype(np.float32) * alpha + base_bgr.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _create_writer(self, out_path: str, width: int, height: int) -> tuple[cv2.VideoWriter | None, str]:
        base, ext = os.path.splitext(out_path)
        ext = ext.lower()
        candidates: list[tuple[str, str]] = []

        # Order matters: best compression first.
        h264_fourcc = ["avc1", "H264"]  # "X264" usually implies libx264 (often not present under LGPL builds)

        # VPx (often available in LGPL FFmpeg builds via libvpx); needs .webm for best compatibility
        vpx_fourcc = [("VP90", ".webm"), ("VP80", ".webm")]  # VP9, VP8

        candidates: list[tuple[str, str]] = []

        # Try H.264 in MP4 first
        for c in h264_fourcc:
            candidates.append((out_path, c))

        # If H.264 isn't available, try VP9/VP8 in WebM (smaller than mp4v/mjpg)
        for c, e in vpx_fourcc:
            candidates.append((base + e, c))

        # Then try what user asked, and legacy
        candidates.append((out_path, "mp4v"))
        candidates.append((base + ".avi", "MJPG"))
        for path, codec in candidates:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), self.writer_actual_fps, (width, height))
            if writer.isOpened():
                return writer, path
            writer.release()

        return None, out_path

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
        if stop_event is None:
            stop_event = asyncio.Event()
        if stop_event.is_set():
            return

        metadata, err = OpenCvVideoUtils.load_metadata_opencv(
            video_path, fps=self.writer_fps, height_limit=None, default_fps=30
        )
        if metadata is None:
            raise Exception(err)

        self.writer_actual_fps = metadata["actual_fps"]
        width = metadata["width"]
        height = metadata["height"]
        end_pts = metadata["frame_count"]
        if end_pts <= 0:
            end_pts = max(self.convert_reader_pts_to_writer_pts(until_reader_pts) + 1, until_reader_pts + 1)

        out_dir_name = os.path.dirname(out_path)
        if out_dir_name:
            os.makedirs(out_dir_name, exist_ok=True)

        writer, actual_out_path = self._create_writer(out_path, width, height)
        if writer is None:
            raise Exception(f"OpenCV writer could not be opened for output file: {out_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            writer.release()
            cap.release()
            raise Exception(f"Input file could not be opened with OpenCV:\n{video_path}")

        if use_tqdm:
            progress_bar = tqdm(
                total=until_reader_pts,
                desc="Writing video frames (no audio)",
                unit="frame",
                ncols=100,
                leave=False,
            )
        else:
            progress_bar = None

        reader_last_pts = 0
        write_pts = 0
        last_base_frame = None

        try:
            while (not stop_event.is_set() or not overlay_queue.empty()) and write_pts < end_pts:
                reader_target_pts, im_overlay = await overlay_queue.get()
                if reader_target_pts is None:
                    break

                writer_target_pts = min(int(self.convert_reader_pts_to_writer_pts(reader_target_pts)), end_pts - 1)
                if stop_event.is_set():
                    not_written_duration_secs = (end_pts - writer_target_pts) / self.writer_actual_fps
                    if not_written_duration_secs < 5:
                        writer_target_pts = end_pts - 1
                    else:
                        writer_target_pts = min(
                            writer_target_pts + int(self.writer_actual_fps // 4), end_pts - 1
                        )

                im_overlay = self._prepare_overlay_bgra(im_overlay, width, height)

                while (not stop_event.is_set()) and (write_pts <= writer_target_pts and write_pts < end_pts):
                    ret, base_frame = cap.read()
                    if ret:
                        last_base_frame = base_frame
                    elif last_base_frame is not None:
                        base_frame = last_base_frame
                    else:
                        break

                    im_blend = self._alpha_blend_overlay(base_frame, im_overlay)
                    writer.write(im_blend)
                    write_pts += 1
                    await asyncio.sleep(0)

                if progress_bar:
                    progress_bar.update(max(0, reader_target_pts - reader_last_pts))
                if progress_overlay:
                    progress_overlay.set_progress(
                        100 * reader_target_pts / max(until_reader_pts, 1),
                        message=f"Writing video (no audio)... {reader_target_pts}/{until_reader_pts} frames",
                    )
                reader_last_pts = reader_target_pts
        finally:
            if progress_bar:
                progress_bar.close()
                progress_bar.write(f"\n\nVideo file has been created (no audio): {actual_out_path}\n\n")
            writer.release()
            cap.release()
