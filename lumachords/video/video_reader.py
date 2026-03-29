import asyncio

from lumachords.video.backends import (
    BaseVideoReaderBackend,
    FfmpegVideoReaderBackend,
    FfmpegVideoUtils,
    OpenCvVideoReaderBackend,
)


class VideoReader:
    def __init__(self, fps=0, pix_fmt="bgr24", height_limit=None, backend: str | None = None):
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.height_limit = height_limit
        self.backend = self._resolve_backend(backend)
        self._backend_impl: BaseVideoReaderBackend = self._create_backend_impl()
        self.actual_fps = self._backend_impl.actual_fps

    def _resolve_backend(self, backend: str | None) -> str:
        if backend not in (None, "ffmpeg", "opencv"):
            raise ValueError(f"Invalid video reader backend: {backend}")

        if backend is None:
            return "ffmpeg" if FfmpegVideoUtils.has_ffmpeg_binary() else "opencv"

        if backend == "ffmpeg" and not FfmpegVideoUtils.has_ffmpeg_binary():
            return "opencv"

        return backend

    def _create_backend_impl(self) -> BaseVideoReaderBackend:
        if self.backend == "ffmpeg":
            return FfmpegVideoReaderBackend(fps=self.fps, pix_fmt=self.pix_fmt, height_limit=self.height_limit)
        return OpenCvVideoReaderBackend(fps=self.fps, pix_fmt=self.pix_fmt, height_limit=self.height_limit)

    def _sync_from_backend(self):
        self.actual_fps = self._backend_impl.actual_fps

    def load_metadata(self, video_path, fps=0):
        metadata, err = self._backend_impl.load_metadata(video_path, fps=fps)
        self._sync_from_backend()
        return metadata, err

    def initiate_process(self, video_path: str, stop_event: asyncio.Event, seek=None, frames=None, hwaccel_try: bool = True):
        fn = getattr(self._backend_impl, "initiate_process", None)
        if fn is None:
            return None, None, Exception(f"initiate_process is not supported for backend '{self.backend}'")
        return fn(video_path, stop_event, seek=seek, frames=frames, hwaccel_try=hwaccel_try)

    async def initiate_process_with_retries(self, video_path: str, stop_event: asyncio.Event, seek: any, frames: any):
        fn = getattr(self._backend_impl, "initiate_process_with_retries", None)
        if fn is None:
            raise Exception(f"initiate_process_with_retries is not supported for backend '{self.backend}'")
        return await fn(video_path, stop_event, seek, frames)

    def terminate_ffmpeg_process(self, ffmpeg_process):
        fn = getattr(self._backend_impl, "terminate_ffmpeg_process", None)
        if fn is None:
            return
        fn(ffmpeg_process)

    async def read_frames(self, video_path: str, stop_event: asyncio.Event, seek=None, frames=None, use_tqdm=True, preread_metadata: dict[str, any]=None):
        async for current_pts, meta_dict, in_frame in self._backend_impl.read_frames(
            video_path,
            stop_event,
            seek=seek,
            frames=frames,
            use_tqdm=use_tqdm,
            preread_metadata=preread_metadata,
        ):
            self._sync_from_backend()
            yield current_pts, meta_dict, in_frame

    async def read_send(self, video_path: str, stop_event: asyncio.Event, seek=None, frames=None, use_tqdm=True, preread_metadata: dict[str, any]=None):
        async for current_time, meta_dict, frame in self.read_frames(
            video_path,
            stop_event,
            seek=seek,
            frames=frames,
            use_tqdm=use_tqdm,
            preread_metadata=preread_metadata,
        ):
            yield current_time, meta_dict, frame
