from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import ffmpeg
import numpy as np
from tqdm import tqdm

from .base_video_backends import BaseVideoReaderBackend, BaseVideoWriterBackend, CommonVideoUtils

def monkey_patch_subprocess():
    if sys.platform != "win32":
        return

    if not isinstance(subprocess.Popen, type):
        return

    _original = subprocess.Popen

    class _PatchedPopen(_original):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("shell", False)
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            si = kwargs.get("startupinfo") or subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
            kwargs["startupinfo"] = si
            super().__init__(*args, **kwargs)

    subprocess.Popen = _PatchedPopen

monkey_patch_subprocess()


class FfmpegVideoUtils:
    FFMPEG_OUT_TEXT_PRESS_KEY = "Press [q] to stop, [?] for help"
    FFMPEG_OUT_TEXT_HWACCEL_DISABLED = "Auto hwaccel disabled"
    FFMPEG_OUT_TEXT_HWACCEL_TYPE = "Using auto hwaccel type"
    FFMPEG_OUT_TEXT_FRAME = "frame="
    FFMPEG_OUT_TEXT_FRAME_REGEX = re.compile(r"([a-zA-Z_]+)\s*\=\s*([^\s=]+)")
    FFMPEG_OUT_TEXT_SHOWINFO = "[Parsed_showinfo_"
    FFMPEG_OUT_TEXT_SHOWINFO_REGEX = re.compile(r"([a-zA-Z_]+)\s*\:\s*(\[[^\]]*\]|[^\s=,]+),?\s*")
    FFMPEG_OUT_TEXT_ERROR_FAILED_KEYWORDS = ("failed", "Failed")
    FFMPEG_OUT_TEXT_ERROR_HW_ACCELERATED_KEYWORDS = ("hardware accelerated", "hwaccel")
    FFMPEG_OUT_TEXT_ERROR_SUBMITTING_PACKET = "Error submitting packet to decoder"

    FFMPEG_BINARY_PATH = None

    @staticmethod
    def calculate_fps_int(stream):
        s = stream.get("avg_frame_rate", "0/0")
        if s == "0/0":
            s = stream.get("r_frame_rate", "0/0")
        n, d = map(int, s.split("/"))
        return 0 if d == 0 else int(round(float(n) / float(d)))

    @staticmethod
    def get_ffmpeg_cmd():
        if not __class__.has_ffmpeg_binary():
            return None
        return __class__.FFMPEG_BINARY_PATH + "ffmpeg"

    @staticmethod
    def get_ffprobe_cmd():
        if not __class__.has_ffmpeg_binary():
            return None
        return __class__.FFMPEG_BINARY_PATH + "ffprobe"

    @staticmethod
    def load_metadata(video_path, fps=0):
        try:
            if video_path is None or not len(video_path):
                raise Exception("Invalid input_video path value.")
            if not __class__.has_ffmpeg_binary():
                raise Exception("FFmpeg binaries not exist.")
            ffmpeg
            probe = ffmpeg.probe(
                video_path, cmd=__class__.get_ffprobe_cmd(), v="error", hide_banner=None
            )  # suppress unnecessary stderr output
            file_duration = float(probe["format"]["duration"]) if "duration" in probe["format"] else None
            if file_duration is None:
                return None, Exception(f"Input file is not a video file:\n{video_path}")
            video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            file_fps = __class__.calculate_fps_int(video_stream)
            file_start_time = float(probe["format"]["start_time"]) if "start_time" in probe["format"] else float(0)
            total_duration = file_duration - file_start_time
            has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))

            actual_fps = fps
            if actual_fps < 1:
                actual_fps = file_fps
            frame_count = math.ceil(total_duration * actual_fps)
        except ffmpeg.Error as e:
            return None, Exception(f"FFMPEG error while probing video file: {e.stderr.decode()}")
        except Exception as e:
            return None, e
        return {
            "width": width,
            "height": height,
            "file_fps": file_fps,
            "actual_fps": actual_fps,
            "total_duration": total_duration,
            "frame_count": frame_count,
            "has_audio": has_audio,
        }, None

    @staticmethod
    def parse_output_str(ffmpeg_out_line, regex):
        parsed_dict = {}
        for match in re.finditer(regex, ffmpeg_out_line):
            parsed_dict[match.group(1)] = match.group(2)
        return parsed_dict

    @staticmethod
    def read_parse_stderr(ffmpeg_process, wait: bool = False):
        parsed_output = {}
        stderr_lines = []
        has_error = False
        continue_reading = True
        while (rdline_str := ffmpeg_process.stderr.readline()) or wait:
            if not continue_reading:
                break
            rdline_str = rdline_str.decode("utf-8").strip()
            if not rdline_str and wait:
                continue
            for rdline in rdline_str.split("\r"):
                stderr_lines.append(rdline)
                # Sample: "frame=    1 fps=0.0 q=-0.0 size=    6075kB time=00:00:00.16 bitrate=298597.8kbits/s speed=2.82x"
                if rdline.startswith(__class__.FFMPEG_OUT_TEXT_FRAME):
                    parsed_dict = __class__.parse_output_str(rdline, __class__.FFMPEG_OUT_TEXT_FRAME_REGEX)
                    if "frame" in parsed_dict:
                        parsed_output["frame"] = int(parsed_dict["frame"])
                    if "time" in parsed_dict:
                        parsed_output["time"] = parsed_dict["time"]
                    if "bitrate" in parsed_dict:
                        parsed_output["bitrate"] = parsed_dict["bitrate"]
                    if "speed" in parsed_dict:
                        parsed_output["speed"] = parsed_dict["speed"]
                # Sample: [Parsed_showinfo_1 @ 0x5b35c7b15d80] n:   1 pts:      1 pts_time:0.166667 pos:   160720 fmt:nv12 sar:1/1 s:1920x1080 i:P iskey:0 type:P checksum:1A4DDAB3 plane_checksum:[8D527CC1 8F755DF2] mean:[109 127] stdev:[47.2 32.3]
                elif rdline.startswith(__class__.FFMPEG_OUT_TEXT_SHOWINFO):
                    rdline = rdline[
                        rdline.find(__class__.FFMPEG_OUT_TEXT_SHOWINFO)
                        + len(__class__.FFMPEG_OUT_TEXT_SHOWINFO):
                    ]
                    rdline = rdline[rdline.find("]") + len("]") :]
                    parsed_dict = __class__.parse_output_str(rdline, __class__.FFMPEG_OUT_TEXT_SHOWINFO_REGEX)
                    if "n" in parsed_dict:
                        parsed_output["n"] = int(parsed_dict["n"])
                    if "pts" in parsed_dict:
                        parsed_output["pts"] = int(parsed_dict["pts"])
                    if "pts_time" in parsed_dict:
                        parsed_output["pts_time"] = float(parsed_dict["pts_time"])
                    # if "mean" in parsed_dict:
                    #     parsed_output["mean"] = parsed_dict["mean"]
                    # if "checksum" in parsed_dict:
                    #     parsed_output["checksum"] = parsed_dict["checksum"]
                # Sample: "[av1 @ 0xa2ac25500] Failed setup for format videotoolbox_vld: hwaccel initialisation returned error."
                elif any((failed_keyword in rdline) for failed_keyword in __class__.FFMPEG_OUT_TEXT_ERROR_FAILED_KEYWORDS):
                    has_error = True
                elif __class__.FFMPEG_OUT_TEXT_ERROR_SUBMITTING_PACKET in rdline:
                    continue_reading = False
                    break

            if "pts" in parsed_output and not has_error:
                return parsed_output
        if has_error:
            stderr_lines_str = "\n".join(stderr_lines)
            parsed_output["err"] = stderr_lines_str
            if any(
                (hwaccel_keyword in stderr_lines_str)
                for hwaccel_keyword in __class__.FFMPEG_OUT_TEXT_ERROR_HW_ACCELERATED_KEYWORDS
            ):
                parsed_output["hwaccel_err"] = True
        return parsed_output

    @staticmethod
    def test_binary_executable(args: list, timeout_secs: float = 5.0):
        try:
            cp = subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_secs,
                check=False,
            )
            return cp.returncode == 0
        except Exception:
            return False

    @staticmethod
    def check_ffmpeg_binary(path: Path, check_exists: bool = True, timeout_secs: float = 5.0) -> str:
        if path is not None and not isinstance(path, Path):
            path = Path(path)
        if check_exists:
            if not path.is_file():
                return None
            if not sys.platform.startswith("win") and not os.access(str(path), os.X_OK):
                return None
        try:
            if __class__.test_binary_executable([path, "-version"], timeout_secs=timeout_secs):
                return str(path.parent)
        except Exception:
            return None

    @staticmethod
    def locate_ffmpeg_binary_internal(extra_dirs: list[str | Path], check_direct_call=True) -> str:
        from lumachords.profile_store import ProfileStore
        
        binary_filename = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
        if check_direct_call:
            binary_file_path = Path(binary_filename).expanduser()
            if __class__.check_ffmpeg_binary(binary_file_path, check_exists=False) is not None:
                return ""

        if profile_binary_path := ProfileStore.get_ffmpeg_binary_path():
            binary_file_path = Path(profile_binary_path).expanduser() / binary_filename
            if binary_file_dir := __class__.check_ffmpeg_binary(binary_file_path):
                __class__.set_ffmpeg_binary_path(binary_file_dir, write_profile=False)
                return binary_file_dir
            
        candidates: list[Path] = [Path(d).expanduser() for d in extra_dirs] if extra_dirs else []

        if sys.platform.startswith("win"):
            candidates += [
                Path(os.environ.get("ProgramFiles", r"C:\\Program Files")) / "ffmpeg" / "bin",
                Path(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")) / "ffmpeg" / "bin",
                Path(os.environ.get("ChocolateyInstall", r"C:\\ProgramData\\chocolatey")) / "bin",
                Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin",
            ]
        elif sys.platform == "darwin":
            if homebrew_prefix := os.getenv("HOMEBREW_PREFIX"):
                candidates += [Path(os.path.join(homebrew_prefix, "bin"))]
            candidates += [
                Path("/usr/local/bin"),
                Path("/opt/local/bin"),
                Path("/usr/bin"),
                Path("/bin"),
            ]
        elif sys.platform.startswith("linux"):
            candidates += [
                Path("/usr/bin"),
                Path("/usr/local/bin"),
                Path("/snap/bin"),
                Path("/var/lib/flatpak/exports/bin"),
                Path.home() / ".local" / "bin",
            ]

        if not sys.platform.startswith("win"):
            # Note: This is NOT redundant with os.environ.get("PATH", "").
            # On Windows, GUI and console apps usually share the same system/user PATH (same source of truth).
            # On macOS, however, PATH differs depending on how the app is launched:
            # - From Terminal: it inherits shell-initialized environment (e.g., ~/.zshrc, ~/.bashrc, etc.).
            # - From Finder (double-click): it does not read those shell init files, so PATH is often not containing required details.
            # Therefore, we call a shell to obtain the shell PATH.
            try:
                cp = subprocess.run(
                    ["sh", "-lc", "echo $PATH"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                shell_path = cp.stdout.strip() if cp.returncode == 0 else ""
                if shell_path:
                    candidates += [Path(p).expanduser() for p in shell_path.split(os.pathsep) if p]
            except Exception:
                pass

        env_path = os.environ.get("PATH", "")
        if env_path:
            candidates += [Path(p) for p in env_path.split(os.pathsep) if p]

        for candidate in candidates:
            binary_file_path = candidate / binary_filename
            binary_file_dir = __class__.check_ffmpeg_binary(binary_file_path)
            if binary_file_dir is not None:
                return binary_file_dir
        return None

    @staticmethod
    def locate_ffmpeg_binary(extra_dirs: list[str | Path], check_direct_call=True) -> bool:
        path = __class__.locate_ffmpeg_binary_internal(extra_dirs, check_direct_call=check_direct_call)
        __class__.set_ffmpeg_binary_path(path)
        return path is not None

    @staticmethod
    def set_ffmpeg_binary_path(path: str, write_profile: bool=True):
        from lumachords.profile_store import ProfileStore
        
        if path is not None and len(path) and not path.endswith(os.sep):
            path += os.sep
        __class__.FFMPEG_BINARY_PATH = path
        if write_profile:
            ProfileStore.set_ffmpeg_binary_path(path)

    @staticmethod
    def has_ffmpeg_binary() -> bool:
        return __class__.FFMPEG_BINARY_PATH is not None


class FfmpegVideoReaderBackend(BaseVideoReaderBackend):
    backend_name = "ffmpeg"

    def load_metadata(self, video_path, fps=0):
        metadata, err = FfmpegVideoUtils.load_metadata(video_path, fps=fps)
        if metadata is not None:
            self.actual_fps = metadata["actual_fps"]
        return metadata, err

    def initiate_process(
        self,
        video_path: str,
        stop_event: asyncio.Event,
        seek=None,
        frames=None,
        hwaccel_try: bool = True,
        preread_metadata: dict[str, any] = None,
    ):
        ffmpeg_cmd = FfmpegVideoUtils.get_ffmpeg_cmd()
        if ffmpeg_cmd is None:
            return None, None, Exception("FFmpeg binaries not exist.")

        metadata, err = (preread_metadata, None) if preread_metadata is not None else self.load_metadata(video_path, self.fps)
        if metadata is None:
            return None, None, err

        total_duration = metadata["total_duration"]
        scale_filter_str = ""
        if self.height_limit and metadata["width"] > self.height_limit:
            metadata["width"] = int(self.height_limit * metadata["width"] / metadata["height"])
            metadata["height"] = self.height_limit
            scale_filter_str = f"scale={metadata['width']}:{metadata['height']},"

        input_args_dict = {"hwaccel": ("auto" if hwaccel_try else "none")}
        output_args_dict = {
            "vf": f"fps={self.actual_fps},{scale_filter_str}showinfo",
            "format": "rawvideo",
            "pix_fmt": self.pix_fmt,
        }
        start_pts = 0
        end_pts = math.ceil(total_duration * self.actual_fps)

        if seek:
            input_args_dict["ss"] = seek
            start_pts = math.ceil(CommonVideoUtils.time_to_frame_number(seek, self.actual_fps))

        if frames:
            output_args_dict["frames"] = frames
            end_pts = start_pts + frames

        metadata["start_pts"] = start_pts
        metadata["end_pts"] = end_pts
        metadata["total_duration"] = math.ceil(total_duration)

        ffmpeg_process = (
            ffmpeg.input(video_path, **input_args_dict)
            .output("pipe:", **output_args_dict)
            .run_async(cmd=ffmpeg_cmd, pipe_stdout=True, pipe_stderr=True)
        )
        os.set_blocking(ffmpeg_process.stderr.fileno(), False)
        ffmpeg_process.poll()

        loop_wait_for_press_key_text = True
        stderr_lines = []

        while not stop_event.is_set() and loop_wait_for_press_key_text and ffmpeg_process.poll() is None:
            while not stop_event.is_set() and (rdline := ffmpeg_process.stderr.readline()):
                if stop_event.is_set():
                    break
                rdline = rdline.decode("utf-8").strip()
                stderr_lines.append(rdline)
                if FfmpegVideoUtils.FFMPEG_OUT_TEXT_PRESS_KEY in rdline:
                    loop_wait_for_press_key_text = False
                    break
                if FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_DISABLED in rdline:
                    metadata["hwaccel"] = "disabled"
                if FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_TYPE in rdline:
                    metadata["hwaccel"] = rdline[
                        rdline.find(FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_TYPE)
                        + len(FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_TYPE):
                    ].strip()

        if "hwaccel" not in metadata:
            metadata["hwaccel"] = "disabled"
        if stop_event.is_set():
            ffmpeg_process.terminate()
            return None, None, Exception("FFmpeg reader was cancelled.")
        if ffmpeg_process.returncode:
            last_err = ("FFMPEG Error: " + stderr_lines[-1]) if stderr_lines else None
            ffmpeg_process.terminate()
            return None, None, last_err
        return ffmpeg_process, metadata, None

    async def initiate_process_with_retries(
        self,
        video_path: str,
        stop_event: asyncio.Event,
        seek: any,
        frames: any,
        preread_metadata: dict[str, any] = None,
    ):
        loop = asyncio.get_running_loop()
        for hwaccel_try in (True, False):
            ffmpeg_process, metadata, err = await loop.run_in_executor(
                None, self.initiate_process, video_path, stop_event, seek, frames, hwaccel_try, preread_metadata
            )
            await asyncio.sleep(0)
            if ffmpeg_process is None:
                if err:
                    raise Exception(err)
                raise Exception("Unknown error while initiating FFMPEG process")

            await asyncio.sleep(0)
            try:
                parsed_stderr = await asyncio.wait_for(
                    loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process, True), timeout=1
                )
            except Exception:
                parsed_stderr = await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)

            if parsed_stderr and "err" in parsed_stderr:
                if "hwaccel_err" in parsed_stderr and hwaccel_try:
                    self.terminate_ffmpeg_process(ffmpeg_process)
                    await asyncio.sleep(0)
                    continue
                raise Exception(
                    f"FFMPEG Error:{'\n' if '\n' in parsed_stderr['err'] else ' '}{parsed_stderr['err']}"
                )
            await asyncio.sleep(0)
            return ffmpeg_process, metadata, err

    def terminate_ffmpeg_process(self, ffmpeg_process):
        for p in (ffmpeg_process.stdin, ffmpeg_process.stderr, ffmpeg_process.stdout):
            try:
                if p:
                    p.close()
            except Exception:
                pass
        ffmpeg_process.terminate()

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

        loop = asyncio.get_running_loop()
        ffmpeg_process, metadata, err = await self.initiate_process_with_retries(
            video_path, stop_event, seek, frames, preread_metadata=preread_metadata
        )
        if ffmpeg_process is None:
            if err:
                raise Exception(err)
            raise Exception("Unknown error while initiating FFMPEG process")
        if stop_event.is_set():
            return

        progress_bar = None
        try:
            width = metadata["width"]
            height = metadata["height"]
            total_duration = metadata["total_duration"]
            start_pts = metadata["start_pts"]
            end_pts = metadata["end_pts"]
            hwaccel = metadata["hwaccel"]
            print(
                f"width: {width}, height: {height}, total_duration: {total_duration}, hwaccel: {hwaccel}, "
                f"start_pts: {start_pts}, end_pts: {end_pts}"
            )

            total_frame_count = end_pts - start_pts
            progress_bar = tqdm(total=total_frame_count) if use_tqdm else None
            current_time = None
            current_bitrate = None
            current_speed = None
            first_frame_read = False
            last_pts: int | None = None

            while not stop_event.is_set() and ffmpeg_process.poll() is None:
                if first_frame_read:
                    in_bytes = await loop.run_in_executor(None, ffmpeg_process.stdout.read, width * height * 3)
                else:
                    try:
                        in_bytes = await asyncio.wait_for(
                            loop.run_in_executor(None, ffmpeg_process.stdout.read, width * height * 3), 0.5
                        )
                    except asyncio.TimeoutError:
                        raise Exception("First frame could not be read.")
                if not in_bytes:
                    break
                first_frame_read = True

                parsed_stderr = await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)
                current_pts = parsed_stderr.get("pts")
                if current_pts is not None:
                    last_pts = current_pts
                elif last_pts is None:
                    current_pts = 0
                    last_pts = current_pts
                else:
                    current_pts = last_pts + 1
                    last_pts = current_pts

                in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

                if progress_bar:
                    if "time" in parsed_stderr:
                        current_time = parsed_stderr["time"]
                    if "bitrate" in parsed_stderr:
                        current_bitrate = parsed_stderr["bitrate"]
                    if "speed" in parsed_stderr:
                        current_speed = parsed_stderr["speed"]

                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        speed=current_speed,
                        frame=(f"{start_pts + current_pts + 1}/{end_pts}"),
                        pts=current_pts,
                        time=current_time,
                        bitrate=current_bitrate,
                        refresh=True,
                    )

                meta_dict = {
                    "time": current_time,
                    "bitrate": current_bitrate,
                    "speed": current_speed,
                }
                yield current_pts, meta_dict, in_frame
        finally:
            if progress_bar:
                progress_bar.close()
            try:
                if ffmpeg_process:
                    print(f"Terminating video reader ffmpeg_process {ffmpeg_process.pid}...")
                    await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)
                    self.terminate_ffmpeg_process(ffmpeg_process)
                    print(f"Terminated video reader ffmpeg_process {ffmpeg_process.pid}...")
            except Exception as e:
                print("Termination error at video reader: ", e)


class FfmpegVideoWriterBackend(BaseVideoWriterBackend):
    backend_name = "ffmpeg"

    def __init__(self, reader_fps: int, writer_fps: int = 0, pix_fmt="bgra"):
        super().__init__(reader_fps, writer_fps=writer_fps, pix_fmt=pix_fmt)
        self.audio_supported = True

    def initiate_process(
        self,
        video_path: str,
        out_path: str,
        stop_event: asyncio.Event,
        seek=None,
        frames=None,
    ) -> tuple[subprocess.Popen | None, dict[str, any] | None, str | None]:
        ffmpeg_cmd = FfmpegVideoUtils.get_ffmpeg_cmd()
        if ffmpeg_cmd is None:
            return None, None, Exception("FFmpeg binaries not exist.")

        metadata, err = FfmpegVideoUtils.load_metadata(video_path, self.writer_fps)
        if metadata is None:
            return None, None, err

        out_dir_name = os.path.dirname(out_path)
        if out_dir_name:
            try:
                os.makedirs(out_dir_name, exist_ok=True)
            except Exception:
                return None, None, Exception(f"Failed to create output directory: {out_dir_name}")

        self.writer_actual_fps = metadata["actual_fps"]
        total_duration = metadata["total_duration"]
        width = metadata["width"]
        height = metadata["height"]

        input_args_dict = {}
        start_pts = 0
        if frames:
            end_pts = frames
        else:
            end_pts = math.ceil(total_duration * self.writer_actual_fps)
        if seek:
            input_args_dict["ss"] = seek
            start_pts = math.ceil(CommonVideoUtils.time_to_frame_number(seek, self.writer_actual_fps))

        metadata["start_pts"] = start_pts
        metadata["end_pts"] = end_pts

        main = ffmpeg.input(video_path, **input_args_dict)
        overlay_in = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt=self.pix_fmt,
            s=f"{width}x{height}",
            framerate=self.writer_actual_fps,
        )

        main_v = main.video.filter("setpts", "PTS-STARTPTS")
        ov_v = overlay_in.video.filter("setpts", "PTS-STARTPTS")
        comp = ffmpeg.overlay(main_v, ov_v, x=0, y=0, eof_action="endall", shortest=1)
        out_streams = [comp]

        out_kwargs = dict(
            vcodec="libx264",
            crf=18,
            preset="veryfast",
            acodec="copy",
            movflags="+faststart",
            shortest=None
        )

        has_audio = metadata.get("has_audio", True)
        if has_audio:
            out_streams.append(main.audio)
            out_kwargs["acodec"] = "copy"
        else:
            # No audio: explicitly disable audio, so no map 0:a happens.
            out_kwargs["an"] = None  # ffmpeg flag: -an


        out = (
            ffmpeg.output(
                *out_streams,
                out_path,
                **out_kwargs,
            )
            .global_args("-fflags", "+genpts")
            .overwrite_output()
        )

        ffmpeg_process: subprocess.Popen = out.run_async(cmd=ffmpeg_cmd, pipe_stdin=True, pipe_stderr=True)
        os.set_blocking(ffmpeg_process.stderr.fileno(), False)
        ffmpeg_process.poll()

        loop_wait_for_ffmpeg_version = True
        loop_wait_for_empty_text = True
        stderr_lines = []

        while (
            not stop_event.is_set()
            and (loop_wait_for_ffmpeg_version or loop_wait_for_empty_text)
            and ffmpeg_process.poll() is None
        ):
            saw_line = False
            while not stop_event.is_set() and (rdline := ffmpeg_process.stderr.readline()):
                saw_line = True
                if stop_event.is_set():
                    break
                rdline = rdline.decode("utf-8").strip()
                if "ffmpeg version" in rdline:
                    loop_wait_for_ffmpeg_version = False
                    break
                stderr_lines.append(rdline)
                if FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_DISABLED in rdline:
                    metadata["hwaccel"] = "disabled"
                if FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_TYPE in rdline:
                    metadata["hwaccel"] = rdline[
                        rdline.find(FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_TYPE)
                        + len(FfmpegVideoUtils.FFMPEG_OUT_TEXT_HWACCEL_TYPE):
                    ].strip()
            if not loop_wait_for_ffmpeg_version and not saw_line:
                loop_wait_for_empty_text = False
                break

        if "hwaccel" not in metadata:
            metadata["hwaccel"] = "disabled"
        if ffmpeg_process.returncode:
            last_err = ("FFMPEG Error: " + stderr_lines[-1]) if stderr_lines else None
            ffmpeg_process.terminate()
            return None, None, last_err
        return ffmpeg_process, metadata, None

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
        
        writer_stop_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        ffmpeg_process, metadata, err = await loop.run_in_executor(
            None, self.initiate_process, video_path, out_path, writer_stop_event, None, None
        )
        if ffmpeg_process is None:
            if err:
                raise Exception(err)
            raise Exception("Unknown error while initiating FFMPEG process")
        if stop_event.is_set():
            writer_stop_event.set()
            return

        width = metadata["width"]
        height = metadata["height"]
        start_pts = metadata["start_pts"]
        end_pts = metadata["end_pts"]
        write_pts = start_pts
        any_frame_written = False

        await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)

        if use_tqdm:
            print("")
            progress_bar = tqdm(total=until_reader_pts, desc="Writing video frames", unit="frame", ncols=100, leave=False)
        else:
            progress_bar = None

        reader_last_pts = 0
        try:
            while (not stop_event.is_set() or not overlay_queue.empty()) and write_pts < end_pts:
                reader_target_pts, im_overlay = await overlay_queue.get()
                if reader_target_pts is None:
                    break

                writer_target_pts = self.convert_reader_pts_to_writer_pts(reader_target_pts)
                writer_target_pts = min(int(writer_target_pts), end_pts - 1)
                if stop_event.is_set():
                    not_written_duration_secs = (end_pts - writer_target_pts) / self.writer_actual_fps
                    if not_written_duration_secs < 5:
                        writer_target_pts = end_pts - 1
                    else:
                        writer_target_pts = min(writer_target_pts + self.writer_actual_fps // 4, end_pts - 1)

                im_overlay = self._prepare_overlay_bgra(im_overlay, width, height)
                im_overlay_bytes = im_overlay.tobytes()
                del im_overlay

                try:
                    while write_pts <= writer_target_pts and write_pts < end_pts:
                        await loop.run_in_executor(None, ffmpeg_process.stdin.write, im_overlay_bytes)
                        await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)
                        any_frame_written = True
                        write_pts += 1

                    if progress_bar:
                        progress_bar.update(max(0, reader_target_pts - reader_last_pts))
                    if progress_overlay:
                        progress_overlay.set_progress(
                            100 * reader_target_pts / max(until_reader_pts, 1),
                            message=f"Writing video... {reader_target_pts}/{until_reader_pts} frames",
                        )
                    reader_last_pts = reader_target_pts
                except BrokenPipeError:
                    break
        finally:
            await asyncio.sleep(0)
            if progress_bar:
                progress_bar.close()
                progress_bar.write(f"\n\nVideo file has been created: {out_path}\n\n")
            try:
                if ffmpeg_process:
                    print(f"Terminating video writer ffmpeg_process {ffmpeg_process.pid}...")
                    if not any_frame_written:
                        im_overlay = self._prepare_overlay_bgra(np.zeros((1, width, 4), dtype=np.uint8), width, height)
                        ffmpeg_process.stdin.write(im_overlay.tobytes())
                        await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)
                    if ffmpeg_process.stdin:
                        ffmpeg_process.stdin.close()
                    if ffmpeg_process.poll() is None:
                        await loop.run_in_executor(None, FfmpegVideoUtils.read_parse_stderr, ffmpeg_process)
                        try:
                            await asyncio.wait_for(asyncio.to_thread(ffmpeg_process.wait), timeout=100)
                        except Exception:
                            pass
                    ffmpeg_process.send_signal(signal.SIGINT)
                    if ffmpeg_process.stderr:
                        FfmpegVideoUtils.read_parse_stderr(ffmpeg_process)
                    try:
                        await asyncio.wait_for(asyncio.to_thread(ffmpeg_process.wait), timeout=1_000)
                    except Exception:
                        pass
                    if ffmpeg_process.stderr:
                        FfmpegVideoUtils.read_parse_stderr(ffmpeg_process)
                        ffmpeg_process.stderr.close()
                    await asyncio.sleep(0)
                    print(f"Terminated video writer ffmpeg_process {ffmpeg_process.pid}...")
                if stop_event.is_set():
                    writer_stop_event.set()
            except Exception as e:
                print("Termination error at video writer: ", e)
