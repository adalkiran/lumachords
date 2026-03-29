import asyncio

from lumachords.video.backends import (
    BaseVideoWriterBackend,
    FfmpegVideoUtils,
    FfmpegVideoWriterBackend,
    OpenCvVideoWriterBackend,
)


class VideoWriter:
    def __init__(self, reader_fps: int, writer_fps: int = 0, pix_fmt="bgra", backend: str | None = None):
        self.reader_fps = reader_fps
        self.writer_fps = writer_fps
        self.pix_fmt = pix_fmt
        self.backend = self._resolve_backend(backend)
        self._backend_impl: BaseVideoWriterBackend = self._create_backend_impl()
        self.writer_actual_fps = self._backend_impl.writer_actual_fps
        self.audio_supported = self._backend_impl.audio_supported

    def _resolve_backend(self, backend: str | None) -> str:
        if backend not in (None, "ffmpeg", "opencv"):
            raise ValueError(f"Invalid video writer backend: {backend}")

        if backend is None:
            return "ffmpeg" if FfmpegVideoUtils.has_ffmpeg_binary() else "opencv"

        if backend == "ffmpeg" and not FfmpegVideoUtils.has_ffmpeg_binary():
            return "opencv"

        return backend

    def _create_backend_impl(self) -> BaseVideoWriterBackend:
        if self.backend == "ffmpeg":
            return FfmpegVideoWriterBackend(self.reader_fps, writer_fps=self.writer_fps, pix_fmt=self.pix_fmt)
        return OpenCvVideoWriterBackend(self.reader_fps, writer_fps=self.writer_fps, pix_fmt=self.pix_fmt)

    def _sync_from_backend(self):
        self.writer_actual_fps = self._backend_impl.writer_actual_fps
        self.audio_supported = self._backend_impl.audio_supported

    def initiate_process(self, video_path: str, out_path: str, stop_event: asyncio.Event, seek=None, frames=None):
        fn = getattr(self._backend_impl, "initiate_process", None)
        if fn is None:
            return None, None, Exception(f"initiate_process is not supported for backend '{self.backend}'")
        result = fn(video_path, out_path, stop_event, seek=seek, frames=frames)
        self._sync_from_backend()
        return result

    def _prepare_overlay_bgra(self, overlay_bgra, width: int, height: int):
        return self._backend_impl._prepare_overlay_bgra(overlay_bgra, width, height)

    def convert_reader_pts_to_writer_pts(self, reader_pts: int):
        return self._backend_impl.convert_reader_pts_to_writer_pts(reader_pts)

    async def write_frames(
        self,
        progress_overlay: any,
        video_path,
        out_path,
        overlay_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        until_reader_pts: int,
        use_tqdm=True,
    ):
        await self._backend_impl.write_frames(
            progress_overlay,
            video_path,
            out_path,
            overlay_queue,
            stop_event,
            until_reader_pts,
            use_tqdm=use_tqdm,
        )
        self._sync_from_backend()
