from __future__ import annotations

from asyncio import Event
from collections.abc import Callable
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

from lumachords.utils import Utils


class UserException(Exception):
    pass


class YoutubeDownloadService:
    YOUTUBE_VIDEO_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{11}$")

    @staticmethod
    def normalize_youtube_input(value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("YouTube URL or video ID is required.")
        youtube_id = YoutubeDownloadService.extract_youtube_id(raw)
        if youtube_id:
            return f"https://www.youtube.com/watch?v={youtube_id}"
        parsed = urlparse(raw)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return parsed._replace(query="", fragment="").geturl()
        raise ValueError(
            "Invalid YouTube input. Enter a full URL or an 11-character video ID."
        )
    
    @staticmethod
    def is_valid_youtube_input(value: str) -> bool:
        try:
            __class__.normalize_youtube_input(value)
        except:
            return False
        return True

    @staticmethod
    def extract_youtube_id(value: str) -> str | None:
        raw = (value or "").strip()
        if not raw:
            return None
        if __class__.YOUTUBE_VIDEO_ID_REGEX.fullmatch(raw):
            return raw
        try:
            parsed = urlparse(raw)
            if not parsed.netloc:
                return None
            qs_id = parse_qs(parsed.query).get("v", [None])[0]
            if qs_id and __class__.YOUTUBE_VIDEO_ID_REGEX.fullmatch(qs_id):
                return qs_id
            host = parsed.netloc.lower()
            path_parts = [p for p in parsed.path.split("/") if p]
            if "youtu.be" in host and path_parts:
                yt_id = path_parts[0]
                if __class__.YOUTUBE_VIDEO_ID_REGEX.fullmatch(yt_id):
                    return yt_id
            if path_parts and path_parts[0] in ("shorts", "embed") and len(path_parts) > 1:
                yt_id = path_parts[1]
                if __class__.YOUTUBE_VIDEO_ID_REGEX.fullmatch(yt_id):
                    return yt_id
        except Exception:
            return None
        return None

    @staticmethod
    def _resolve_downloaded_file(info: dict|str, output_dir: Path) -> Path:
        if isinstance(info, str):
            video_id = info
        else:
            video_id = info.get("id")
        if not video_id:
            raise RuntimeError("Failed to resolve downloaded video ID.")
        matches = sorted(
            output_dir.glob(f"*{video_id}*.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise RuntimeError("Download finished but output file could not be found.")
        mp4_file = next((path for path in matches if path.suffix.lower() == ".mp4"), None)
        return (mp4_file or matches[0]).resolve()

    @staticmethod
    def on_progress(stop_event: Event, status: dict, progress_callback: Callable[[float, dict], None] = None) -> None:
        if stop_event.is_set():
            raise UserException("Download cancelled by user.")
        if not progress_callback:
            return
        download_status = status.get("status")
        if download_status == "downloading":
            pct = status.get("_percent")
            if pct is None:
                downloaded = status.get("downloaded_bytes") or 0
                total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
                pct = (downloaded * 100.0 / total) if total else 0.0
            progress_callback(min(100.0, max(0.0, float(pct))), status)
        elif download_status == "finished":
            progress_callback(100.0, status)

    @staticmethod
    def download_youtube_video(
        value: str,
        ffmpeg_binary_path: Path,
        stop_event: Event,
        output_dir: str = "data/videos/youtube",
        progress_callback: Callable[[float, dict], None] = None,
    ) -> str:
        from yt_dlp import YoutubeDL

        normalized_url = YoutubeDownloadService.normalize_youtube_input(value)
        out_dir = Path(output_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except:
            out_dir = Utils.config_path() / output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
        youtube_id = __class__.extract_youtube_id(normalized_url)
        resolved_path: Path | None = None
        try:
            path = __class__._resolve_downloaded_file(youtube_id, out_dir)
            if path.is_file():
                return str(path), True
        except:
            pass

        has_ffmpeg_binary = ffmpeg_binary_path is not None and len(str(ffmpeg_binary_path))
        # specified_ffmpeg_location will be None if ffmpeg_binary_path contains only "ffmpeg" string.
        # This is for PyInstaller executables which cannot access PATH environment variable.
        specified_ffmpeg_location = str(ffmpeg_binary_path) if has_ffmpeg_binary and ffmpeg_binary_path and Path(ffmpeg_binary_path).exists() else None

        # Youtube stores high-res videos separately from audio (DASH). Only with some progressive low-res video files are stored video and audio together.
        # So, in some configurations (like "bestvideo*+bestaudio/best"), YoutubeDL downloads audio (with best quality) and video (with best quality) separately,
        # then merges using FFmpeg. But without FFmpeg, it cannot merge separate parts. 
        # In some other configurations (like "bestvideo[vcodec!=none]"), we tell YoutubeDL to download the video with best quality even if it has not audio.
        #
        # In our case, has_ffmpeg_binary specifies if FFmpeg is available.
        # If not available, we download a silent video file. It's not a problem with us because without FFmpeg,
        # our application uses OpenCV to create video output file which is also silent.
        ydl_opts = {
            "format": "bestvideo*+bestaudio/best" if has_ffmpeg_binary else "bestvideo[vcodec!=none]",
            "merge_output_format": "mp4" if has_ffmpeg_binary else None,
            "ffmpeg_location": specified_ffmpeg_location,
            "outtmpl": str(out_dir / "%(uploader)s - %(title)s [%(id)s].%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "remote_components": ["ejs:github"],
            "js_runtimes": {"deno": {}},
            "progress_hooks": [lambda status: __class__.on_progress(stop_event, status, progress_callback)],
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(normalized_url, download=False)
                if info is None:
                    raise RuntimeError("No downloadable media found for this YouTube input.")
                try:
                    path = __class__._resolve_downloaded_file(info, out_dir)
                    if path.is_file():
                        return str(path), True
                except:
                    pass

                info = ydl.extract_info(normalized_url, download=True)
                if info is None:
                    raise RuntimeError("No downloadable media found for this YouTube input.")
                if isinstance(info, dict) and "entries" in info:
                    entries = [entry for entry in info.get("entries", []) if entry]
                    if not entries:
                        raise RuntimeError("No downloadable media found for this YouTube input.")
                    info = entries[0]
                path = __class__._resolve_downloaded_file(info, out_dir)
                return str(path), False
        except UserException as exc:
            try:
                # When download is interrupted or cancelled by the user, delete the downloaded file
                resolved_path = __class__._resolve_downloaded_file(youtube_id, out_dir)
                if resolved_path and resolved_path.is_file():
                    try:
                        resolved_path.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
            raise exc
        except Exception as exc:
            raise RuntimeError(f"YouTube download failed. You can manually download the video using tools like yt-dlp and open the file with LumaChords.\n{exc}") from exc
