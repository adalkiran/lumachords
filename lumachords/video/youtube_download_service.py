from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse


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
    def download_youtube_video(value: str, has_ffmpeg_binary: bool, output_dir: str = "data/videos/youtube") -> str:
        from yt_dlp import YoutubeDL

        normalized_url = YoutubeDownloadService.normalize_youtube_input(value)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        youtube_id = __class__.extract_youtube_id(normalized_url)
        try:
            path = __class__._resolve_downloaded_file(youtube_id, out_dir)
            if path.is_file():
                return str(path), True
        except:
            pass

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
            "outtmpl": str(out_dir / "%(uploader)s - %(title)s [%(id)s].%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "remote_components": ["ejs:github"],
            "js_runtimes": {"deno": {}},
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
        except Exception as exc:
            raise RuntimeError(f"YouTube download failed. You can manually download the video using tools like yt-dlp and open the file with LumaChords.\n{exc}") from exc
