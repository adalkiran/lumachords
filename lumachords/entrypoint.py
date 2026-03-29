# ruff: noqa: E402
import multiprocessing
import os
import sys

def ensure_sys_streams() -> bool:
    any_null_sys_stream = any(sys_stream is None for sys_stream in [sys.stdout, sys.stderr, sys.stdin])
    if not any_null_sys_stream:
        return True
    devnull = open(os.devnull, "w", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = devnull
    if sys.stderr is None:
        sys.stderr = devnull
    if sys.stdin is None:
        sys.stdin = devnull
    return False

terminal_is_available = ensure_sys_streams()

# Set environment variables before any lumachords imports that may trigger pygame/OpenGL loading
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
if sys.platform == "linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# This is because of OpenCV (cv2) and other .so and .dylib dependency files are
# validated and scanned by MacOS if the application is on MacOS.
print("Initializing... (This may take some time at the first run)")

from pathlib import Path

from lumachords.utils import Utils

import argparse
import asyncio
from lumachords.app import App
from lumachords.video import YoutubeDownloadService
from lumachords.runtime_config import AppMode, AppSettings
from lumachords.profile_store import ProfileStore

def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_pause_pts(value: str):
    if "-" in value:
        start_str, end_str = value.split("-", 1)
        return (int(start_str), int(end_str))
    return int(value)

def parse_split_note(value: str) -> str:
    note_name = Utils.parse_split_note_to_name(value)
    if note_name is None:
        raise argparse.ArgumentTypeError(f"Invalid split midi num: {value}")
    return value

def parse_transpose_octaves(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid transpose octaves value: {value}") from exc
    if parsed < -2 or parsed > 2:
        raise argparse.ArgumentTypeError("Transpose octaves must be between -2 and 2.")
    return parsed

def is_valid_source_type(input: str, input_source: str) -> bool:
    if input_source == "file" and Path(input).is_file():
        return True
    if input_source == "youtube" and YoutubeDownloadService.is_valid_youtube_input(input):
        return True
    return False

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument(
        "--mode",
        choices=("headless", "basic", "advanced"),
        default=None,
        help="Run mode: headless (no GUI window), basic GUI window, or advanced GUI window",
    )
    parser.add_argument("-i", "--input", required=False, help='Could be input video file path, a valid Youtube video ID (with "-is youtube" argument), a valid Youtube video URL (with "-is youtube" argument).')
    parser.add_argument(
        "-is", "--input-source",
        choices=("file", "youtube"),
        default="file",
        help="Specifies if the input value is a file or a Youtube URL/ID.",
    )
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-to", "--transpose-octaves", type=parse_transpose_octaves, default=None)
    parser.add_argument("--split-note", type=parse_split_note, default=None)
    parser.add_argument("--skip-until-pts", type=parse_int_list, default=None)
    parser.add_argument(
        "--pause-at-pts",
        action="append",
        type=parse_pause_pts,
        default=None,
        help="Repeat for multiple values. Use N or START-END.",
    )
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument(
        "--auto-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable automatic timing estimation (time signature + BPM).",
    )
    parser.add_argument("--time-sync", action="store_true", default=None)
    parser.add_argument(
        "--video-backend",
        choices=("ffmpeg", "opencv"),
        default=None,
        help="Video processing backend. FFmpeg is recommended but it needs to be installed first.",
    )
    parser.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug_mode", action="store_true", default=False, help=argparse.SUPPRESS)
    return parser


def demo_args(args):
    # The following input sample videos are third-party YouTube videos for
    # limited research/testing use purposes only. See experiment_samples/EXPERIMENT_SOURCES.txt
    # and tests/data/TEST_SOURCES.txt for ownership and limited-use notes.
    
    # UNCOMMENT ONLY ONE OF THE FOLLOWING LINES:

    input_video_path, input_source, split_note, skip_until_pts, pause_at_pts = (

        "VuRKmmpV35w", "youtube", None, None, None # one man's dream / It's Piano - One Man's Dream - Yanni | Piano | Synthesia | Relaxing music
        #"NGbKJ0mS3bQ", "youtube", None, None, None # gulpembe / Piano by VN - Konser piyanisti GÜLPEMBE çalarsa :)
        #"79MFcQJizto", "youtube", None, None, None # alman dansı / KolayNota - Haydn - Alman Dansı - Piyano
        #"QJyJaisdCpg", "youtube", None, None, None # bring me to life / HDpiano - How to Play "Bring Me to Life" by Evanescence | HDpiano (Part 1) Piano Tutorial
        #"SjZxAaD1I10", "youtube", None, None, None # santa lucia / It's Piano - Santa Lucia - Teodoro Cottrau | Traditional Neapolitan Song | Piano Solo Synthesia Tutorial
        #"QU7BZhDZ8zY", "youtube", None, None, None # senorita / Piano Go Life - Shawn Mendes, Camila Cabello - Señorita Piano Tutorial

    )    

    try_input_video_path = f"data/videos/{input_video_path}"
    if Path(try_input_video_path).is_file():
        input_video_path = try_input_video_path
    demo_args = argparse.Namespace(
        time_sync=True,
        input=input_video_path,
        input_source=input_source,
        output=".",
        split_note=split_note,
        skip_until_pts=skip_until_pts,
        pause_at_pts=pause_at_pts,
    )
    merged = vars(args).copy()
    merged.update({k: v for k, v in vars(demo_args).items() if v is not None})
    return argparse.Namespace(**merged)

def parse_args():
    mode_dict = {
        "headless": AppMode.HEADLESS,
        "basic": AppMode.GUI_BASIC,
        "advanced": AppMode.GUI_ADVANCED,
    }
    parser = build_arg_parser()
    try:
        args = parser.parse_args()
        if args.demo:
            args = demo_args(args)

        if sys.platform == "win32" and bool(getattr(sys, "frozen", False)):
            if terminal_is_available:
                if not args.mode:
                    args.mode = "headless"
                if args.mode != "headless":
                    exe_name = Path(sys.executable).stem
                    exe_name_without_cli = exe_name[:4]
                    raise Exception(f"Only headless mode is available for this executable file. Use {exe_name_without_cli}.exe for windowed usage.") 
            elif args.mode == "headless":
                exe_name = Path(sys.executable).stem
                raise Exception(f"Headless mode is not available for this executable file. Use {exe_name}-cli.exe for console usage.") 
        elif sys.platform == "linux":
            if not os.environ.get("DISPLAY"):
                if not args.mode:
                    args.mode = "headless"
                if args.mode != "headless":
                    raise Exception("Only headless mode is available for this machine. Use --mode=headless with your application.") 
        saved_mode = ProfileStore.get_global_mode()
        if saved_mode and saved_mode not in ["basic", "advanced"]:
            saved_mode = None
        mode_name = args.mode or saved_mode or "basic"
        if mode_name not in mode_dict:
            mode_name = "basic"
        app_mode = mode_dict[mode_name]

        if args.input_source:
            args.input_source = args.input_source.strip()

        if args.input:
            args.input = args.input.strip()
            input_profile = ProfileStore.get_input_profile(args.input_source, args.input)
            if input_profile:
                if args.transpose_octaves is None:
                    try:
                        saved_transpose = int(input_profile.get("transpose_octaves"))
                        args.transpose_octaves = saved_transpose if -2 <= saved_transpose <= 2 else None
                    except Exception:
                        args.transpose_octaves = None
                if args.split_note is None:
                    saved_split_note = input_profile.get("split_note")
                    if saved_split_note and Utils.parse_split_note_to_name(str(saved_split_note)) is not None:
                        args.split_note = str(saved_split_note)

        if args.split_note is None:
            args.split_note = "C4"
        if args.video_backend is None:
            args.video_backend = "ffmpeg"

        from lumachords.midi_rt.common import MidiRtOption

        midirt_backend, midirt_title = ProfileStore.get_midirt_device()
        args.midirt_option = MidiRtOption(midirt_title, midirt_backend, None) if midirt_backend else None
        args.midirt_velocity = ProfileStore.get_midirt_velocity()
        args.midirt_use_pedal = ProfileStore.get_midirt_use_pedal()


        if app_mode not in [AppMode.GUI_BASIC, AppMode.GUI_ADVANCED]:
            if not args.input:
                raise argparse.ArgumentTypeError("argument -i/--input is required.")
            else:
                if not is_valid_source_type(args.input, args.input_source):
                    raise argparse.ArgumentTypeError("argument -i/--input and -is/--input-source argument pair has incompatible values.")
            if not args.output:
                raise argparse.ArgumentTypeError("argument -o/--output is required.")
        return args, app_mode, mode_name
    except Exception as e:
        if sys.platform == "win32" and not terminal_is_available:
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, str(e), "LumaChords", 0x10)  # MB_ICONERROR
                sys.exit(2)
            except Exception:
                parser.error(str(e))
        else:
            parser.error(str(e))


async def _async_main():
    args, app_mode, mode_name = parse_args()
    input_video_path = args.input
    output_video_path = args.output if args.output else None
    settings = AppSettings(
        app_mode=app_mode,
        debug_mode=args.debug_mode,
        start_paused=args.start_paused,
        input_video_path=input_video_path if args.input_source == "file" else None,
        youtube_input=input_video_path if args.input_source == "youtube" else None,
        input_source=args.input_source,
        output_video_path=output_video_path,
        transpose_octaves=args.transpose_octaves,
        split_note=args.split_note,
        skip_until_pts=args.skip_until_pts,
        pause_at_pts=args.pause_at_pts,
        time_sync=args.time_sync,
        video_backend=args.video_backend,
        midirt_option=args.midirt_option,
        midirt_velocity=args.midirt_velocity,
        midirt_use_pedal=args.midirt_use_pedal,
        auto_timing=args.auto_timing,
    )

    if getattr(sys, "frozen", False) and app_mode != AppMode.HEADLESS:
        # PyInstaller (onefile/onedir), to avoid adding an extra process icon onto MacOS Dock.
        import threading
        from tqdm import tqdm
        tqdm.set_lock(threading.RLock())

    ProfileStore.set_global_mode(mode_name)
    if args.input:
        ProfileStore.set_input_profile(args.input_source, args.input, settings.transpose_octaves, settings.split_note)

    app = App(settings)
    await app.run()


def main():
    # Needed for PyInstaller/frozen apps so multiprocessing child/helper processes (spawn/resource_tracker)
    # bootstrap correctly instead of re-running our main() with weird args.
    multiprocessing.freeze_support()
    Utils.add_extra_library_search_paths()
    asyncio.run(_async_main())
    print("Finished.")


if __name__ == "__main__":
    main()
