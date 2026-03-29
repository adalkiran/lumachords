from dataclasses import dataclass
from enum import IntEnum

from lumachords.midi_rt import MidiRtOption
from lumachords.utils import Utils


class LogLevel(IntEnum):
    LOGLEVEL_NONE = 0
    LOGLEVEL_INFO = 1
    LOGLEVEL_DEBUG = 2
    LOGLEVEL_VERBOSE = 3

class AppMode(IntEnum):
    HEADLESS = 1
    GUI_BASIC = 2
    GUI_ADVANCED = 3
    NOTEBOOK = 4
class ProdMode(IntEnum):
    PROD = 1
    DEBUG = 2

@dataclass(frozen=True)
class RuntimeConfig:
    app_mode: AppMode
    prod_mode: ProdMode
    log_level: LogLevel

@dataclass
class AppSettings:
    app_mode: AppMode
    debug_mode: bool = False
    keybed_detection_log_level: LogLevel = LogLevel.LOGLEVEL_NONE
    note_rain_detection_log_level: LogLevel = LogLevel.LOGLEVEL_NONE
    start_paused: bool = False
    input_video_path: str = None
    input_source: str = "file"
    youtube_input: str = ""
    output_video_path: str = None
    transpose_octaves: int = None
    split_note: int|str = "C4"
    split_midi_num: int = None
    skip_until_pts: list[int] = None
    pause_at_pts: list[int]|list[tuple[int, int]] = None
    time_sync: bool = True
    video_backend: str = "ffmpeg"
    midirt_option: MidiRtOption = None
    midirt_velocity: int = 20
    midirt_use_pedal: bool = True
    auto_timing: bool = False

    def __post_init__(self):
        if self.split_midi_num is None:
            self.split_midi_num = Utils.parse_split_note_to_midi_num(self.split_note) or 60
        else:
            self.split_note = Utils.midi_num_to_name(self.split_midi_num)

    def requires_gui(self) -> bool:
        return self.app_mode in (AppMode.GUI_BASIC, AppMode.GUI_ADVANCED)
    
    def requires_notes(self) -> bool:
        return self.app_mode in (AppMode.GUI_BASIC, AppMode.GUI_ADVANCED)
    
    def requires_active_notes(self) -> bool:
        return self.app_mode in (AppMode.GUI_BASIC, AppMode.GUI_ADVANCED)
    
    def get_video_backend_title(self) -> str:
        return "OpenCV" if self.video_backend == "opencv" else "FFmpeg"
