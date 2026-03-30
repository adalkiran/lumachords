import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys
from typing import TYPE_CHECKING

import pygame
import numpy as np


from lumachords.data_types import BackgroundType
from lumachords.hands_detector import HandsType
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.overlays import MenuType, OverlayManager
from lumachords.processing_state import ProcessingState
from lumachords.notation_placer import NotationPlacer
from lumachords.midi_tracker import MidiTracker
from lumachords.midi_rt import MidiRt
from lumachords.runtime_config import AppMode, AppSettings, ProdMode, RuntimeConfig
from lumachords.image_utils import ImageUtils
from lumachords.image_input import ImagePreprocessor
from lumachords.profile_store import ProfileStore
from lumachords.preferences import Preferences
from lumachords.processor import Processor
from lumachords.video import VideoReader, YoutubeDownloadService
from lumachords.utils import TimeSync, Utils
from lumachords.video.backends import FfmpegVideoUtils

from lumachords.gui.file_dialog_utils import FileDialogUtils
from lumachords.ui_types import PlaybackMode, UICommand, UIUtils

if TYPE_CHECKING:
    from lumachords.gui.window import Window


@dataclass
class AppComponents:
    midi_tracker: MidiTracker = None
    midi_rt: MidiRt = None
    note_placer: NotationPlacer = None
    active_note_placer: NotationPlacer = None
    video_reader: VideoReader = None
    processor: Processor = None


class App:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.cmp: AppComponents = None
        self.window: Window = None
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.stdin_reader_registered = False
        self.quit_confirm_pending = False
        self.ffmpeg_prompt_active = False
        self.ffmpeg_prompt_event: asyncio.Event | None = None
        self.input_metadata: dict[str, any] = None
        self.frame: np.ndarray = None
        self.pts: int = -1
        self.video_processing_finished = False
        self.state: ProcessingState = None
        self.play_y_lag_time_delta: float = 0.0
        self.apply_lag_to_edge: bool = False
        self.app_level_stop_event = asyncio.Event()
        self.info_panel_data: dict[str, any] = {
            "velocity_consensus": None,
            "phase": "Initializing",
            "transpose_octaves": None,
            "key_count": None,
            "note_rain_height": None, 
            "keybed_height": None,
            "hands_type": None,
            "bg_type": None,
            "note_detection_method": None,
        }
    
    def bind_state(self, state: ProcessingState):
        self.state = state
        if self.window:
            self.window.bind_state(self.state)

    async def finalize_video_processing(self):
        self.video_processing_finished = True
        self.mute_rt()
        if self.window:
            self.app_level_stop_event.clear()
            OverlayManager.instance().rebuild_and_show_menu(MenuType.MENU_FINISH)
        if not self.settings.requires_gui() and self.settings.output_video_path and not self.app_level_stop_event.is_set():
            await self.do_save_video(self.settings.output_video_path)

    async def on_change_settings(self):
        self.apply_lag_to_edge = (self.settings.app_mode != AppMode.GUI_ADVANCED)
        if not (self.cmp and self.cmp.processor):
            return
        processor = self.cmp.processor
        old_runtime_config = processor.get_current_runtime_config()
        runtime_config = RuntimeConfig(self.settings.app_mode, old_runtime_config.prod_mode, old_runtime_config.log_level)
        self.bind_state(processor.set_current_runtime_config(runtime_config))
        if self.window:
            await self.window.present_async(preview_fps=30)

    async def do_save_video(self, filename: str) -> bool|str:
        if not (self.cmp and self.cmp.midi_tracker and self.cmp.processor and self.cmp.video_reader):
            raise Exception("Components not initialized for saving video.")
        user_level_stop_event = asyncio.Event()
        save_video_text = "Saving video..." if self.settings.video_backend == "ffmpeg" else "Saving video (no audio)..."
        progress_overlay = OverlayManager.show_progress(save_video_text, user_level_stop_event)
        try:
            saved_valid_video = await self.cmp.midi_tracker.save_video(
                progress_overlay,
                self.settings.input_video_path,
                filename,
                self.frame.shape[:2],
                self.pts,
                self.play_y_lag_time_delta,
                self.app_level_stop_event,
                user_level_stop_event,
                self.cmp.note_placer.copy(),
                self.cmp.video_reader.actual_fps,
                writer_fps=0,
                pix_fmt="bgra"
            )
        finally:
            if progress_overlay:
                progress_overlay.close_dialog()
        if not saved_valid_video:
            return "Video was not created because of no note events detected."
        return not (user_level_stop_event.is_set() or self.app_level_stop_event.is_set())

    async def handle_commands_async(self):
        cmd = None
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if cmd == UICommand.QUIT:
                    print("Quitting...")
                    return False
                elif cmd == UICommand.START:
                    # do nothing
                    pass
                elif cmd == UICommand.DOWNLOAD_YOUTUBE:
                    if not self.settings.youtube_input or not self.settings.youtube_input.strip():
                        OverlayManager.show_toast("YouTube URL or video ID must be specified.")
                        continue
                    user_level_stop_event = asyncio.Event()
                    progress_overlay = OverlayManager.show_progress(
                        f"Checking for YouTube video: {self.settings.youtube_input}...",
                        user_level_stop_event,
                    )
                    try:
                        video_path, cache_used = await asyncio.to_thread(
                            YoutubeDownloadService.download_youtube_video,
                            self.settings.youtube_input,
                            FfmpegVideoUtils.has_ffmpeg_binary(),
                            user_level_stop_event,
                            progress_callback=lambda download_percentage, _: progress_overlay.set_progress(download_percentage, message=f"Downloading YouTube video: {self.settings.youtube_input}...")
                        )
                        self.settings.input_video_path = video_path
                        OverlayManager.show_toast(
                            f"YouTube video downloaded: {os.path.basename(video_path)}" if not cache_used else f"Using YouTube video from cache: {os.path.basename(video_path)}"
                        )
                        menu_overlay = OverlayManager.instance().get(OverlayManager.OVERLAY_MENU)
                        if menu_overlay and menu_overlay.is_enabled():
                            menu_overlay.toggle()
                        self.command_queue.put_nowait(UICommand.START)
                    finally:
                        if progress_overlay:
                            progress_overlay.close_dialog()
                elif cmd == UICommand.TAKE_SHOT:
                    file_path = FileDialogUtils.show_save_dialog("Select Output Screenshot File", "png")
                    if file_path:
                        await ImageUtils.imsave(self.frame, file_path)
                        OverlayManager.show_toast(f"Shot saved to file {file_path}")
                elif cmd == UICommand.SAVE_MIDI:
                    file_path = FileDialogUtils.show_save_dialog("Select Output MIDI File", "mid")
                    if file_path:
                        self.cmp.midi_tracker.save_midi(file_path)
                        OverlayManager.show_toast(f"File saved: {file_path}")
                elif cmd == UICommand.SAVE_MEI:
                    file_path = FileDialogUtils.show_save_dialog("Select Output MEI File", "mei")
                    if file_path:
                        await asyncio.to_thread(Utils.save_mei, self.cmp.note_placer.midi_to_mei(None, for_file=True), file_path)
                        OverlayManager.show_toast(f"File saved: {file_path}")
                elif cmd == UICommand.SAVE_VIDEO:
                    file_path = FileDialogUtils.show_save_dialog("Select Output Video", "mp4")
                    if file_path:
                        is_success = await self.do_save_video(file_path)
                        if isinstance(is_success, str):
                            message = f"No video saved: {is_success}"
                        else:
                            message = f"File saved: {file_path}" if is_success else f"Operation was cancelled, but the completed part has been saved in file: {file_path}"
                        OverlayManager.show_toast(message)
                elif cmd == UICommand.CHANGE_SETTINGS:
                    await self.on_change_settings()
            except Exception as e:
                OverlayManager.show_error(e)
        return True
    
    def handle_commands(self):
        asyncio.create_task(self.handle_commands_async())

    def mute_rt(self):
        self.cmp.midi_rt.mute_all()

    def start_headless_terminal_listener(self):
        if self.stdin_reader_registered:
            return
        if not sys.stdin or not sys.stdin.isatty():
            return
        # Headless flow: try locating ffmpeg once and surface a clear hint if missing.
        try:
            ffmpeg_binary_located = FfmpegVideoUtils.locate_ffmpeg_binary(None, check_direct_call=True)
            if not ffmpeg_binary_located and self.settings.video_backend == "ffmpeg":
                if ProfileStore.get_continue_without_ffmpeg():
                    self.settings.video_backend = "opencv"
                    sys.stdout.write('FFmpeg was not found. Using remembered preference: continue without FFmpeg ("OpenCV" backend).\n')
                    sys.stdout.flush()
                else:
                    try:
                        from lumachords.overlays.menu_builders.ffmpeg_builder import print_ffmpeg_terminal_instructions
                        self.ffmpeg_prompt_event = asyncio.get_running_loop().create_future()
                        print_ffmpeg_terminal_instructions()
                        self.ffmpeg_prompt_active = True
                    except Exception:
                        sys.stdout.write("FFmpeg was not found. Install it or add it to PATH. See docs/installing-ffmpeg.html\n")
                        sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"FFmpeg detection failed: {e}\n")
            sys.stdout.flush()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            def _on_stdin():
                try:
                    line = sys.stdin.readline().strip()
                except Exception:
                    return
                if not line:
                    return
                from lumachords.overlays.menu_builders.ffmpeg_builder import handle_ffmpeg_prompt
                if len(line) > 1:
                    if self.ffmpeg_prompt_active:
                        handle_ffmpeg_prompt(self, "\0")
                        return
                ch = line[0]
                if self.ffmpeg_prompt_active:
                    handle_ffmpeg_prompt(self, ch)
                    return
                if self.quit_confirm_pending:
                    if ch.lower() == "y":
                        self.quit_confirm_pending = False
                        self.command_queue.put_nowait(UICommand.QUIT)
                    elif ch.lower() == "n":
                        self.quit_confirm_pending = False
                        sys.stdout.write("Resuming...\n")
                        sys.stdout.flush()
                    return
                if ch.lower() == "q":
                    self.quit_confirm_pending = True
                    sys.stdout.write("Are you sure to quit? (y/n)\n")
                    sys.stdout.flush()
            loop.add_reader(sys.stdin, _on_stdin)
            self.stdin_reader_registered = True
        except Exception:
            return
        
    async def wait_for_menu(self):
        if not self.window or self.app_level_stop_event.is_set():
            return
        overlay_manager = OverlayManager.instance()
        while overlay_manager.get_any_overlay_require_pause() and not self.app_level_stop_event.is_set():
            try:
                if not await self.window.present_async(preview_fps=30):
                    return False
                if not await self.handle_commands_async():
                    return True
            except Exception as e:
                OverlayManager.show_error(e)
            await asyncio.sleep(0)
        return True

    async def setup_window_or_terminal(self) -> dict[str, any]:
        window = None
        has_video_system = False
        if self.settings.requires_gui():
            has_video_system = UIUtils.ensure_video_system_init()
            if not has_video_system:
                self.settings.app_mode = AppMode.HEADLESS
        if self.settings.requires_gui() and has_video_system:
            window = UIUtils.create_window_object(self.settings, self.handle_commands)
            self.command_queue = window.command_queue
            window.playback_mode = PlaybackMode.NOT_STARTED
            self.window = window
            if not window.create_window():
                self.window = None
            if self.window:
                OverlayManager.show_toast("Initializing...", duration_ms=0)
                if not await self.window.present_async(preview_fps=0):
                    return None
                if not await self.wait_for_menu():
                    return None
                await asyncio.sleep(0)
                if not await self.window.present_async(preview_fps=0):
                    return None
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            ffmpeg_binary_located = FfmpegVideoUtils.locate_ffmpeg_binary(None, check_direct_call=True)
            if not ffmpeg_binary_located:
                if ProfileStore.get_continue_without_ffmpeg():
                    if self.settings.video_backend == "ffmpeg":
                        OverlayManager.show_toast('FFmpeg was not found. Using remembered preference: continue without FFmpeg ("OpenCV" backend).')
                    self.settings.video_backend = "opencv"
                else:
                    await asyncio.to_thread(OverlayManager.instance().rebuild_and_show_menu, MenuType.MENU_FFMPEG)
                    OverlayManager.show_toast(None, 0.000001)
                    if not await self.wait_for_menu():
                        return None
                    if self.settings.video_backend == "ffmpeg" and FfmpegVideoUtils.FFMPEG_BINARY_PATH is None:
                        return None
            await NotationPlacer.check_is_ready()
            await asyncio.to_thread(OverlayManager.instance().rebuild_and_show_menu, MenuType.MENU_START)
            if self.window:
                OverlayManager.show_toast(None, 0.000001)
            if not await self.wait_for_menu():
                return None
        else:
            self.window = None
            self.start_headless_terminal_listener()
            if self.ffmpeg_prompt_event:
                try:
                    await self.ffmpeg_prompt_event
                except Exception:
                    pass
            print(f"\n\nVideo Backend: {self.settings.get_video_backend_title()}\n\n")
            await NotationPlacer.check_is_ready()
            if self.settings.input_source == "youtube" and not self.settings.input_video_path:
                youtube_input = (self.settings.youtube_input or "").strip()
                if not youtube_input:
                    raise Exception("YouTube URL or video ID is required in headless mode when input_source is 'youtube'.")
                print("Detected YouTube input in -i/--input. Downloading...")
                video_path, cache_used = await asyncio.to_thread(
                    YoutubeDownloadService.download_youtube_video,
                    youtube_input,
                    FfmpegVideoUtils.has_ffmpeg_binary(),
                    self.app_level_stop_event,
                    progress_callback=None,
                )
                self.settings.input_video_path = video_path
                print(f"YouTube download completed: {video_path}" if not cache_used else f"Using YouTube video from cache: {video_path}")
        video_path = self.settings.input_video_path
        await self.setup_components(compact=True)
        await self.on_change_settings()
        metadata, err = await asyncio.to_thread(self.cmp.video_reader.load_metadata, video_path, self.cmp.video_reader.fps)
        if err:
            raise Exception(err)
        return metadata
    
    def play_y_lag_time_delta_callback(self, play_y_lag_time_delta: float, velocity_consensus: float):
        self.play_y_lag_time_delta = float(play_y_lag_time_delta)
        self.info_panel_data["velocity_consensus"] = float(velocity_consensus)
        self.update_info_panel()

    def get_actual_play_y_lag_time(self, force=True) -> float:
        return self.play_y_lag_time_delta if self.apply_lag_to_edge or force else 0.0
    
    def get_actual_happening_time(self, pts_time: float, force=True) -> float:
        return Utils.calculate_actual_happening_time(pts_time, self.apply_lag_to_edge or force, self.play_y_lag_time_delta, self.cmp.video_reader.actual_fps)

    def hands_type_callback(self, hands_type: HandsType):
        self.info_panel_data["hands_type"] = hands_type
        self.update_info_panel()


    def update_info_panel(self):
        analyzing_str = "..."
        metadata = self.input_metadata
        info_panel_data = self.info_panel_data
        video_path = self.settings.input_video_path
        phase = info_panel_data["phase"]

        key_count = info_panel_data["key_count"]
        key_count_str = str(key_count) if key_count is not None else None
        transpose_octaves = info_panel_data["transpose_octaves"]
        transpose_octaves_str = str(transpose_octaves or "-") if transpose_octaves is not None else None
        note_rain_height = info_panel_data["note_rain_height"]
        note_rain_height_str = str(note_rain_height) if note_rain_height is not None else None
        keybed_height = info_panel_data["keybed_height"]
        keybed_height_str = str(keybed_height) if keybed_height is not None else None
        

        bg_type = info_panel_data["bg_type"]
        bg_type_str = f"{bg_type}" if bg_type else None
        note_detection_method = info_panel_data["note_detection_method"]
        note_detection_method_str = f"{note_detection_method}" if note_detection_method else None
        hands_type = info_panel_data["hands_type"]
        hands_type_str = str(hands_type) if hands_type is not None else None
        velocity_consensus = info_panel_data["velocity_consensus"]
        note_rain_velocity_per_sec_str = f"{(velocity_consensus * metadata["actual_fps"]):.2f}" if velocity_consensus is not None else None
        note_rain_velocity_str = f"{velocity_consensus:.2f}" if velocity_consensus is not None else None
        play_y_lag_time_delta_str = f"{self.play_y_lag_time_delta:.2f}" if velocity_consensus is not None else None
        

        info_panel_list = [
            "Press F1 for main menu",
            f'File: {os.path.basename(video_path)}',
            f'Frame Size: {metadata["width"]} x {metadata["height"]}',
            f'File FPS: {metadata["file_fps"]}',
            f'Actual FPS: {metadata["actual_fps"]}',
            "",
            f"Phase: {phase}",
            f"Key Count: {key_count_str or analyzing_str}",
            f"Transpose Octaves: {transpose_octaves_str or analyzing_str}",
            f"Note Rain Section Height (pixels): {note_rain_height_str or analyzing_str}",
            f"Keybed Section Height (pixels): {keybed_height_str or analyzing_str}",
            "",
            f'Background Type: {bg_type_str or analyzing_str}',
            f'Detection Method: {note_detection_method_str or analyzing_str}',
            f'Hands Type: {hands_type_str or analyzing_str}',
            f'Note Rain Velocity (pixels per frame): {note_rain_velocity_str or analyzing_str}',
            f'Note Rain Velocity (pixels per seconds): {note_rain_velocity_per_sec_str or analyzing_str}',
            f'Lag time delta (seconds): {play_y_lag_time_delta_str or analyzing_str}',            
        ]
        info_panel_str = "\n".join(info_panel_list)
        self.state.info_panel_str = info_panel_str

    async def post_setup_window_or_terminal(self):
        window = self.window
        if window:
            metadata = self.input_metadata
            frame_w = metadata["width"]
            frame_h = metadata["height"]
            frame_count = metadata["frame_count"]

            OverlayManager.instance().rebuild_menu(MenuType.MENU_PROCESSING)
            window.wdef.frame_size = (int(frame_w), int(frame_h))
            window.apply_window_size((int(frame_w * 2.1), int(frame_h * 2.1)), force=True, apply_center=True)
            self.bind_state(self.cmp.processor.init_keybed_detector_phase())
            window.progresss_init(frame_count)
            if self.settings.start_paused:
                window.playback_mode = PlaybackMode.PAUSED
            else:
                window.playback_mode = PlaybackMode.NORMAL
            self.info_panel_data["phase"] = "Keybed detection"
            self.update_info_panel()
            OverlayManager.show_toast("Detecting keybed...")
        else:
            self.bind_state(self.cmp.processor.init_keybed_detector_phase())

    async def setup_components(self, compact=False):
        if not self.cmp:
            self.cmp = AppComponents()
        pref = Preferences(video_fps=10, video_height_limit=None)
        if not self.cmp.video_reader:
            self.cmp.video_reader = VideoReader(fps=pref.engine.video_fps, height_limit=pref.engine.video_height_limit, backend=self.settings.video_backend)
        if compact:
            return
        keybed_runtime_config = RuntimeConfig(self.settings.app_mode, ProdMode.PROD, self.settings.keybed_detection_log_level)
        note_rain_runtime_config = RuntimeConfig(self.settings.app_mode, ProdMode.PROD, self.settings.note_rain_detection_log_level)
        self.cmp.processor = Processor(
            pref,
            keybed_runtime_config,
            note_rain_runtime_config,
            self.cmp.video_reader.actual_fps,
            play_y_lag_time_delta_callback_fn=self.play_y_lag_time_delta_callback,
            hands_type_callback_fn=self.hands_type_callback,
        )
        self.cmp.midi_tracker = MidiTracker(self.cmp.video_reader.actual_fps, default_split_midi_num=self.settings.split_midi_num, video_backend=self.settings.video_backend, midi_velocity=self.settings.midirt_velocity)
        self.cmp.midi_rt = MidiRt(self.settings.midirt_option, velocity=self.settings.midirt_velocity, use_pedal=self.settings.midirt_use_pedal)

        self.cmp.note_placer = NotationPlacer(crop_silence_at_start=True, take_latest_measures=4, default_split_midi_num=self.settings.split_midi_num, print_timing=True, auto_timing=self.settings.auto_timing)
        self.cmp.active_note_placer = NotationPlacer(crop_silence_at_start=False, min_measures=1, default_split_midi_num=self.settings.split_midi_num, always_use_latest_hands_range=True, include_image_alpha_channel=False, print_timing=False, auto_timing=False)

    def show_detecting_notes_toast(self, keybed_output: KeybedDetectorOutput, transpose_octaves: int, background_type: BackgroundType):
        info_panel_data = self.info_panel_data
        info_panel_data["phase"] = "Note detection"
        info_panel_data["transpose_octaves"] = transpose_octaves
        info_panel_data["key_count"] = len(keybed_output.all_keys_data)
        _, keybed_top_y, _, keybed_height = keybed_output.keybed_bounds
        info_panel_data["note_rain_height"] = keybed_top_y
        info_panel_data["keybed_height"] = keybed_height

        message = f"Keybed with {info_panel_data["key_count"]} keys has been detected, detecting notes{f' with transposing {transpose_octaves} octaves' if transpose_octaves != 0 else ''}..."
        if background_type is None:
            background_message = "Analyzing..."
        else:
            info_panel_data["bg_type"] = f"{background_type}"
            background_message = f"{background_type}. "
            if background_type == BackgroundType.SPARSE:
                info_panel_data["note_detection_method"] = "edges on black/white form"
            elif background_type == BackgroundType.TEXTURED:
                info_panel_data["note_detection_method"] = "start/end lines on luma form"
            background_message += f"Note rain boxes will be detected from {info_panel_data["note_detection_method"]}..."
        message += f"\nDetected background type: {background_message}"
        self.update_info_panel()
        OverlayManager.show_toast(message, duration_ms=10_000)

    async def process_video(self, metadata: dict[str, any]):
        if self.app_level_stop_event.is_set():
            return
        self.input_metadata = metadata
        window = self.window
        input_video_path = self.settings.input_video_path
        transpose_octaves = self.settings.transpose_octaves
        await self.setup_components()
        await self.post_setup_window_or_terminal()
        keybed_output: KeybedDetectorOutput = None
        is_keybed_output_success = False
        hands_output_ranges = None
        hands_midi_num_ranges = None

        cmp = self.cmp

        seek = None
        time_sync = TimeSync(True) if window and self.settings.time_sync else None
        pause_after_keybed_detection = False
        skip_frame = False
        skip_until_pts = np.array(self.settings.skip_until_pts) if self.settings.skip_until_pts is not None else None
        pause_at_pts = np.array([x for v in self.settings.pause_at_pts for x in (range(v[0], v[1] + 1) if isinstance(v, tuple) else (v,))], dtype=int) if self.settings.pause_at_pts else None
        current_meta_dict = {}
        last_app_mode = self.settings.app_mode
        await self.check_matplotlib_config_cache_exists_if_required()

        known_background_info: tuple[BackgroundType, int] = None
        known_background_info_hist = []
        try:
            async for pts, meta_dict, frame in cmp.video_reader.read_send(input_video_path, self.app_level_stop_event, seek=seek, preread_metadata=metadata):
                self.pts = pts
                self.frame = frame
                if window:
                    window.progress_current = pts
                    if meta_dict:
                        current_meta_dict = {**current_meta_dict, **meta_dict}
                        current_meta_str = ", ".join([f"{k}: {v}" for k, v in current_meta_dict.items() if v is not None])
                        window.progress_post_text = f"({current_meta_str})" if len(current_meta_str) else None
                pts_time = Utils.pts_to_pts_time(pts, cmp.video_reader.actual_fps)
                cmp.midi_rt.set_pts_time(pts_time)
                happening_pts, happening_time = self.get_actual_happening_time(pts_time)
                if time_sync and is_keybed_output_success:
                    time_sync.step(pts_time)

                skip_frame = False
                if skip_until_pts is not None and is_keybed_output_success and np.any(pts < skip_until_pts):
                    skip_frame = True
                if window and pause_at_pts is not None and np.any(pts == pause_at_pts):
                    window.playback_mode = PlaybackMode.PAUSED
                frame_processed = None
                if skip_frame:
                    self.state.set_state_image(0, frame)
                else:
                    if not is_keybed_output_success:
                        if window:
                            window.progress_text = "Detecting Keybed..."
                            if self.window and last_app_mode != self.settings.app_mode:
                                last_app_mode = self.settings.app_mode
                                await self.check_matplotlib_config_cache_exists_if_required()
                        kb_image_input = await ImagePreprocessor.preprocess_for_keybed(frame)
                        keybed_output = await cmp.processor.detect_keys(kb_image_input, pts)
                        is_keybed_output_success = keybed_output.evaluation_result is None
                        if is_keybed_output_success and pause_after_keybed_detection and window:
                            window.playback_mode = PlaybackMode.PAUSED
                        elif isinstance(keybed_output.evaluation_result, Exception):
                            raise keybed_output.evaluation_result
                        if not is_keybed_output_success and (pts_time > 0.5 and pts_time % 10 <= 0.1):
                            OverlayManager.show_toast("Keyboard hasn’t been detected yet.\n\n• If your video has an intro before the keyboard appears, that’s normal, please wait.\n• If the keyboard is already visible, the app may need at least one clear frame showing the keyboard without hands.")
                    else:
                        if cmp.processor.note_rain_pipeline is None:
                            if transpose_octaves is None:
                                transpose_octaves = keybed_output.get_transpose_suggestion()
                            self.bind_state(cmp.processor.init_note_rain_pipeline_phase(keybed_output))
                            self.show_detecting_notes_toast(keybed_output, transpose_octaves, None)
                        nr_image_input = await ImagePreprocessor.preprocess_for_note_rain(frame, cmp.processor.note_rain_pipeline.keybed_output.keybed_bounds, known_background_info)
                        if known_background_info is None:
                            known_background_info_hist.append(nr_image_input.background_info)
                            if len(known_background_info_hist) > 1 and known_background_info_hist[-2] != known_background_info_hist[-1]:
                                known_background_info_hist = []
                            if len(known_background_info_hist) == 10:
                                known_background_info = known_background_info_hist[-1]
                                known_background_info_hist = []
                                known_background_type, _ = known_background_info
                                self.show_detecting_notes_toast(keybed_output, transpose_octaves, known_background_type)
                        if window:
                            window.progress_text = "Detecting Notes..."

                        raw_events, hands_output_ranges = await cmp.processor.detect_note_rain(
                            pts,
                            nr_image_input,
                            transpose_octaves,
                        )
                        hands_midi_num_ranges = hands_output_ranges.to_midi_num_ranges()
                        note_events = await asyncio.to_thread(cmp.midi_tracker.step_frame, pts, raw_events, hands_midi_num_ranges, self.get_actual_play_y_lag_time())
                        cmp.midi_rt.play(note_events)                
                    if self.settings.requires_active_notes():
                        self.state.midi_events_str = cmp.midi_tracker.report_last_items(pts, pts_time, happening_pts=happening_pts)
                        hands_report = ["-", "-"]
                        if hands_output_ranges is not None and len(hands_output_ranges.items) < 3:
                            for i, range_item in enumerate(hands_output_ranges.items):
                                if range_item is None:
                                    continue
                                range_item_start, range_item_end = range_item
                                text_start, text_end = range_item_start.note_name, range_item_end.note_name
                                hands_report[i] = f"{text_start} - {text_end}"
                        self.state.midi_events_str += f"\n\nHANDS:\nLEFT: {hands_report[0]}\nRIGHT: {hands_report[1]}"
                        vp_active_notes = window.panels[self.state.IDX_ACTIVE_NOTES].viewport
                        # Drawing Active Notes
                        state_item = self.state.get_state(self.state.IDX_ACTIVE_NOTES)
                        prev_measure_events = state_item.data
                        cur_measure_events = cmp.midi_tracker.get_active_groups_as_measure_events(hands_midi_num_ranges, happening_pts=happening_pts)
                        if prev_measure_events != cur_measure_events:
                            cmp.active_note_placer.set_state(
                                    cur_measure_events,
                                    cmp.midi_tracker.hands_midi_num_ranges_per_time,
                            )
                            self.state.set_state(
                                self.state.IDX_ACTIVE_NOTES,
                                cur_measure_events,
                                await cmp.active_note_placer.midi_to_image(
                                    None,
                                    foreground_color="white",
                                    fixed_size=True,
                                    margin_vertical_extra_units=None,
                                    output_width=vp_active_notes.w,
                                    output_height=vp_active_notes.h
                                )
                            ) # Artifact (STATE): Active notes visualization
                    # Setting note placer state
                    state_item = self.state.get_state(0)
                    prev_event_pairs = state_item.data
                    cur_event_pairs = cmp.midi_tracker.get_event_pairs(True, happening_time=happening_time)
                    if prev_event_pairs != cur_event_pairs:
                        cmp.note_placer.set_state(
                                cur_event_pairs,
                                cmp.midi_tracker.hands_midi_num_ranges_per_time,
                        )
                    if self.settings.requires_notes():
                        # Drawing Staff
                        if prev_event_pairs != cur_event_pairs:
                            state_item.cache = await cmp.note_placer.midi_to_image(
                                None,
                                background_color="white",
                                alpha_rate=0.5,
                                fixed_size=True,
                                print_measure_nums_interval=1,
                                margin_horizontal_extra_units=2,
                                output_width=frame.shape[1],
                                output_height=frame.shape[0]*1//3,
                            )
                        frame_processed = await ImageUtils.blend_images(frame, state_item.cache)
                        self.state.set_state(0, cur_event_pairs, frame_processed, cache=state_item.cache) # Artifact (STATE): Main output with notation
                    else:
                        self.state.set_state(0, cur_event_pairs, None, cache=None)
                # Don't call the next one if previous returned False
                if window:
                    if not await window.present_async(preview_fps=30):
                        return None, None
                    if not await window.sleep_if_paused_async(callback_fn=self.mute_rt):
                        return None, None
                    if not await window.controlled_sleep_async(0):
                        return None, None
                if not await self.handle_commands_async():
                    return None, None
                if window:
                    if not await window.sleep_if_paused_async(callback_fn=self.mute_rt):
                        return None, None
                del frame
                del frame_processed
                # Yield back to the event loop so other tasks (e.g. MIDI worker) can run
                await asyncio.sleep(0)
            # Video processing is finished
            if not self.video_processing_finished:
                await self.finalize_video_processing()
            if window:
                if not await self.wait_for_menu():
                    return None, None
        except Exception as e:
            if window:
                await asyncio.sleep(0)
                OverlayManager.show_error(e)
                await asyncio.sleep(0)
                await self.wait_for_menu()
            else:
                raise e
        finally:
            if not self.video_processing_finished:
                await self.finalize_video_processing()
            if cmp.midi_rt is not None:
                cmp.midi_rt.close()
                del cmp.midi_rt
            if window is not None:
                await asyncio.sleep(0)
                await window.cleanup_async()
            await asyncio.sleep(0)
    
    def do_app_level_stop(self):
        if self.app_level_stop_event and self.app_level_stop_event.is_set():
            try:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            except:
                exit(0)
            return
        self.app_level_stop_event.set()

    async def check_matplotlib_config_cache_exists_if_required(self):
        if self.settings.app_mode == AppMode.GUI_ADVANCED and not Utils.check_matplotlib_config_cache_exists():
            OverlayManager.show_toast("Initializing the Fourier Analysis chart...\nThis is a one-time setup and may take up to 15 seconds.", duration_ms=0)
            if not await self.window.present_async(preview_fps=0):
                return None
            if not await self.wait_for_menu():
                return None
            await asyncio.sleep(0)
            try:
                ensure_matplotlib_task = asyncio.create_task(asyncio.to_thread(Utils.ensure_matplotlib))
                while not ensure_matplotlib_task.done():
                    if not await self.window.present_async(preview_fps=30):
                        return None
                    if not await self.handle_commands_async():
                        return None
                    await asyncio.sleep(0)
                await ensure_matplotlib_task
            except Exception as e:
                OverlayManager.show_error(e)
                if not await self.window.present_async(preview_fps=30):
                    return None
            OverlayManager.show_toast(None, 0.000001)

    async def run(self):
        if self.settings.debug_mode:
            asyncio.get_event_loop().set_debug(True)
        if self.settings.app_mode != AppMode.NOTEBOOK:
            ImageUtils.imshow = lambda *args, **kwargs: None
        asyncio.get_event_loop().slow_callback_duration = 10.0 # 10 seconds
        if not self.settings.requires_gui() and not self.settings.output_video_path:
            raise Exception("In headless mode, output video path must be specified.")
        loop = asyncio.get_running_loop()
        is_win = (sys.platform == "win32")
        if is_win:
            # Windows: poll for Ctrl+C
            def signal_handler(sig, frame):
                loop.call_soon_threadsafe(lambda: asyncio.create_task(self.do_app_level_stop()))
        
            signal.signal(signal.SIGINT, signal_handler)
        else:
            loop.add_signal_handler(signal.SIGINT, lambda: (self.do_app_level_stop()))
            loop.add_signal_handler(signal.SIGTERM, lambda: (self.do_app_level_stop()))
        try:
            print("Initialization completed.")
            metadata = None
            try:
                metadata = await self.setup_window_or_terminal()
            except Exception as e:
                OverlayManager.show_error(e)
                if self.window:
                    await self.wait_for_menu()
                return

            if not await self.handle_commands_async():
                return
            
            if self.app_level_stop_event.is_set():
                return

            if metadata is None:
                return

            if self.settings.output_video_path:
                p = Path(self.settings.output_video_path)
                if p.is_dir():
                    base_name = os.path.splitext(os.path.basename(self.settings.input_video_path))[0]
                    base_name = f"out-overlay-{base_name}.mp4"
                    self.settings.output_video_path = str(p / base_name)
            await self.process_video(metadata)
        finally:
            if not is_win:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                    loop.remove_signal_handler(signal.SIGTERM)
                except Exception:
                    pass
            if self.window is not None:
                await asyncio.sleep(0)
                await self.window.cleanup_async()
            
