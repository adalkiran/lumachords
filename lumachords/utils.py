from ast import Module
#import logging
import importlib.util
import os
from pathlib import Path
import sys
import sysconfig
import time
import xml.etree.ElementTree as ET
import copy
import numpy as np
from platformdirs import user_config_dir

from lumachords.data_types import BoxIsValid

class DetectionException(Exception):
    pass

class Utils:
    NOTE_ALL_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    NOTE_ALL_OCTAVES = [str(i) for i in range(1, 9)]
    MIDI_NUM_C0 = 12

    MATPLOTLIB_MODULE_CACHE: Module = None
    HAS_MATPLOTLIB_GUI = False

    APP_ROOT_PATH: Path = None
    CONFIG_PATH: Path = None
    RESOURCE_PATH: Path = None

    @staticmethod
    def check_matplotlib_config_cache_exists():
        # Mathplotlib generates a file like "fontlist-v390.json" containing information about system fonts at the first run.
        # If the file does not exist, matplotlib scans installed fonts and gather information, it takes time.
        # Here, we check if the file already exists or not.
        cfg = __class__.config_path() / "matplotlib"
        return cfg.is_dir() and any(cfg.glob("font*.json"))

    @staticmethod
    def ensure_matplotlib():
        if __class__.MATPLOTLIB_MODULE_CACHE is not None:
            return __class__.MATPLOTLIB_MODULE_CACHE

        cfg = __class__.config_path() / "matplotlib"
        cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cfg)

        import matplotlib
        from matplotlib import pyplot as plt
        __class__.MATPLOTLIB_MODULE_CACHE = matplotlib
        __class__.HAS_MATPLOTLIB_GUI = plt.backend_registry.resolve_backend(plt.get_backend())[1] is not None
        return __class__.MATPLOTLIB_MODULE_CACHE

    @staticmethod
    def pts_to_pts_time(pts, actual_fps):
        return pts / actual_fps
    
    @staticmethod
    def pts_time_to_pts(pts_time, actual_fps):
        return int(np.floor(pts_time * actual_fps))

    @staticmethod
    def calculate_actual_happening_time(pts_time: float, apply_lag_to_edge: bool, play_y_lag_time_delta: float, actual_fps: int) -> float:
        happening_time = (pts_time - play_y_lag_time_delta) if apply_lag_to_edge and play_y_lag_time_delta > 0.0 else None
        if happening_time and happening_time < 0.0:
            happening_time = 0.0
        happening_pts = __class__.pts_time_to_pts(happening_time, actual_fps) if happening_time is not None else None
        return happening_pts, happening_time

    @staticmethod
    def k_sigma_threshold(arr: np.ndarray, k: float = 1.0, eps: float = 1e-12) -> float:
        mu = float(arr.mean())
        sigma = float(arr.std()) + eps
        thr = mu + k * sigma
        return thr

    @staticmethod
    def font_scale(im, scale):
        h, w = im.shape[:2]
        return (min(w,h)/(25/scale))*0.02

    @staticmethod
    def mad_filter(arr):
        arr = np.array(arr)
        # MAD (Mean Abdsolute Deviation) filter
        med  = np.median(arr)
        mad  = np.median(np.abs(arr - med))
        keep = np.abs(arr - med) <= 3*mad
        return np.median(arr[keep])

    @staticmethod
    def invalidate_covered_rectangles(rects: np.ndarray, tol) -> np.ndarray:
        valid_rects = rects[rects["is_valid"] > 0]
        x0, y0, x1, y1 = valid_rects["x0"], valid_rects["y0"], valid_rects["x1"], valid_rects["y1"]
        snap_score = np.sum(((valid_rects["snap_diff_top"] > -999), (valid_rects["snap_diff_bottom"] > -999)), axis=0)
        w, h = (x1-x0).astype("i4"), (y1-y0).astype("i4")

        area = w * h

        # i is inside j (allowing equal borders)
        inside_matrix = (
            (x0[:, None] + tol >= x0[None, :]) &
            (y0[:, None] + tol >= y0[None, :]) &
            (x1[:, None] - tol <= x1[None, :]) &
            (y1[:, None] - tol <= y1[None, :])
        )

        pairs = {}
        for inside_idx, cover_idx in zip(*np.where(inside_matrix)):
            if inside_idx != cover_idx:  # skip self
                if cover_idx not in pairs:
                    pairs[cover_idx] = []
                pairs[cover_idx].append(inside_idx)

        mark_invalid = np.zeros(valid_rects.shape, dtype=bool)
        for cover_idx, inside_list in pairs.items():
            inside_list = np.array(inside_list)
            if len(inside_list) == 1:
                inside_idx = inside_list[0]
                if snap_score[inside_idx] == snap_score[cover_idx]:
                    if area[inside_idx] == area[cover_idx]:
                        mark_invalid[min(inside_idx, cover_idx)] = True
                    elif area[inside_idx] > area[cover_idx]:
                        mark_invalid[cover_idx] = True
                    else:
                        mark_invalid[inside_idx] = True
                elif snap_score[inside_idx] > snap_score[cover_idx]:
                    mark_invalid[cover_idx] = True
                else:
                    mark_invalid[inside_idx] = True
            else:
                inside_larger_w_mask = w[inside_list] >= w[cover_idx]
                inside_larger_w_list = inside_list[inside_larger_w_mask]
                inside_smaller_w_list = inside_list[~inside_larger_w_mask]

                if len(inside_smaller_w_list):
                    if not mark_invalid[cover_idx]:
                        mark_invalid[inside_smaller_w_list] = True

                if np.any(snap_score[inside_larger_w_list] > 0):
                    mark_invalid[cover_idx] = True
                else:
                    mark_invalid[inside_larger_w_list] = True

        valid_idx = np.flatnonzero(rects["is_valid"] > 0)
        rects["is_valid"][valid_idx[mark_invalid]] = BoxIsValid.Invalid
        return rects


    @staticmethod
    def normalize(data):
        return (data - data.mean()) / (data.std(ddof=0) + 1e-9) # z-score

    @staticmethod
    def color_to_bgr(rgb_color):
        return tuple(reversed(rgb_color[:3])) + tuple(rgb_color[3:])
    
    @staticmethod
    def midi_num_to_name(midi_num):
        all_notes_cnt = len(__class__.NOTE_ALL_NAMES)
        tmp = midi_num - __class__.MIDI_NUM_C0
        octave = int(np.floor(tmp / all_notes_cnt))
        note_idx = tmp % all_notes_cnt
        note_name = __class__.NOTE_ALL_NAMES[note_idx] + str(octave)
        return note_name

    @staticmethod
    def name_to_midi_num(note_name: str) -> int:
        if note_name is None:
            raise ValueError("note_name is required")
        name = note_name.strip().upper()
        if len(name) < 2:
            raise ValueError(f"Invalid note name: {note_name}")
        note = name[0]
        idx = 1
        if len(name) > 1 and name[1] == "#":
            note += "#"
            idx = 2
        octave_str = name[idx:]
        if not octave_str or not octave_str.lstrip("-").isdigit():
            raise ValueError(f"Invalid octave in note name: {note_name}")
        if note not in __class__.NOTE_ALL_NAMES:
            raise ValueError(f"Invalid note name: {note_name}")
        octave = int(octave_str)
        all_notes_cnt = len(__class__.NOTE_ALL_NAMES)
        note_idx = __class__.NOTE_ALL_NAMES.index(note)
        return __class__.MIDI_NUM_C0 + (octave * all_notes_cnt) + note_idx

    @staticmethod
    def parse_split_note_to_name(value: str) -> str:
        midi_num = None
        try:
            midi_num = int(value)
            note_name = Utils.midi_num_to_name(midi_num)
            if note_name is not None:
                return note_name
        except ValueError:
            note_name = value.strip().upper()
            try:
                midi_num = Utils.name_to_midi_num(note_name)
                return note_name
            except ValueError:
                pass
        return None

    @staticmethod
    def parse_split_note_to_midi_num(value: str) -> int:
        midi_num = None
        try:
            midi_num = int(value)
            if Utils.midi_num_to_name(midi_num):
                return midi_num
        except ValueError:
            note_name = value.strip().upper()
            try:
                midi_num = Utils.name_to_midi_num(note_name)
                return midi_num
            except ValueError:
                pass
        return None
    
    staticmethod
    def split_note_name_to_parts(note: str):
        note = (note or "C4").strip().upper()
        # match C, C#, etc + octave
        if len(note) >= 2 and note[1] in ("#", "B"):  # if you ever want flats, handle 'B' carefully
            name, octv = note[:2], note[2:]
        else:
            name, octv = note[:1], note[1:]
        if name not in __class__.NOTE_ALL_NAMES:
            name = "C"
        if octv not in __class__.NOTE_ALL_OCTAVES:
            octv = "4"
        return name, octv


    @staticmethod
    def midi_num_to_staff_idx(midi_num, split_midi_num):
        return 0 if midi_num >= split_midi_num else 1

    @staticmethod
    def save_mei(mei: ET.Element, out_path: str):
        mei_str = __class__.xml_element_to_str(mei)
        with open(out_path, 'w') as file:
            file.write(mei_str)

    @staticmethod
    def xml_element_to_str(xml_element: ET.Element, indent: int = None):
        if indent:
            xml_element = copy.deepcopy(xml_element)
            ET.indent(xml_element, space=" " * 4)
        xml_str = ET.tostring(xml_element, encoding="utf-8", xml_declaration=True).decode("utf-8")
        return xml_str
    
    @staticmethod
    def app_root() -> Path:
        if __class__.APP_ROOT_PATH:
            return __class__.APP_ROOT_PATH
        if getattr(sys, "frozen", False):
            # PyInstaller (onefile/onedir)
            __class__.APP_ROOT_PATH = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
            return __class__.APP_ROOT_PATH

        # Prefer the installed module location over the console-script wrapper.
        # Under `uv run lumachords`, `__main__.__file__` points at `.venv/bin/lumachords`,
        # while this module still resolves to the project source tree in editable mode.
        candidates = [Path(__file__).resolve().parent.parent]

        main_mod = sys.modules.get("__main__")
        main_file = getattr(main_mod, "__file__", None)
        if main_file:
            candidates.append(Path(main_file).resolve().parent.parent)

        for candidate in candidates:
            if (candidate / "lumachords").is_dir() and (candidate / "resources").is_dir():
                __class__.APP_ROOT_PATH = candidate
                return __class__.APP_ROOT_PATH

        __class__.APP_ROOT_PATH = candidates[0]
        return __class__.APP_ROOT_PATH

    @staticmethod
    def config_path() -> Path:
        if __class__.CONFIG_PATH:
            return __class__.CONFIG_PATH
        # User config directory, the parent path may change per operating system.
        __class__.CONFIG_PATH = Path(user_config_dir("LumaChords", "Adil Alper DALKIRAN"))
        __class__.CONFIG_PATH.mkdir(parents=True, exist_ok=True)
        return __class__.CONFIG_PATH

    @staticmethod
    def resource_path(file_name: str, *path_parts: str, return_as_str:bool = True) -> str | Path:
        if __class__.RESOURCE_PATH is None:
            candidates = [
                __class__.app_root().joinpath(*path_parts, file_name),
                Path(sysconfig.get_path("data")).joinpath(*path_parts, file_name),
                Path(sys.prefix).joinpath(*path_parts, file_name),
            ]

            path = candidates[0]
            for candidate in candidates:
                if candidate.is_file():
                    __class__.RESOURCE_PATH = candidate.parent
                    path = candidate
                    break
        else:
            path = __class__.RESOURCE_PATH.joinpath(file_name)
        return str(path) if return_as_str else path

    @staticmethod
    def add_extra_library_search_paths():
        extras = []
        platform = sys.platform
        if getattr(sys, "frozen", False):
            # PyInstaller (onefile/onedir)
            meipass = str(Path(getattr(sys, "_MEIPASS")).resolve())
            extras.append(meipass)
        else:
            spec = importlib.util.find_spec("pygame")
            pygame_mod_file = spec.origin if spec and spec.origin else None
            if pygame_mod_file:
                pygame_mod_dir = Path(Path(pygame_mod_file).resolve().parent).resolve().parent
                if platform == "darwin":
                    pygame_mod_path = pygame_mod_dir / "pygame" / ".dylibs"
                    extras.append(str(pygame_mod_path))
                    found_libs = list(pygame_mod_path.glob("libfluidsynth*.dylib"))
                    if len(found_libs):
                        found_lib = found_libs[0]
                        target = found_lib.parent / (found_lib.stem.split(".")[0] + found_lib.suffix)
                        if not target.is_file():
                            target.symlink_to(found_lib)
                elif platform.startswith("linux"):
                    pygame_mod_path = pygame_mod_dir / "pygame.libs"
                    extras.append(str(pygame_mod_path))
                    found_libs = list(pygame_mod_path.glob("libfluidsynth*.so*"))
                    if len(found_libs):
                        found_lib = found_libs[0]
                        target = found_lib.parent / (found_lib.stem.split(".")[0] + ".so")
                        if not target.is_file():
                            target.symlink_to(found_lib)
        if extras:
            env_key = None
            if platform == "darwin":
                env_key = "DYLD_FALLBACK_LIBRARY_PATH"
            elif platform.startswith("linux"):
                env_key = "LD_LIBRARY_PATH"
            if env_key:
                env_val = os.environ.get(env_key, None)
                os.environ[env_key] =  ":".join(extras) + ((":" + env_val) if env_val else "")

class TimeSync:
    def __init__(self, enabled):
        self.enabled = enabled
        self.anchor = None
        self.last_pts_time = None
    

    def step(self, pts_time):
        if not self.enabled:
            return
        now = time.perf_counter()
        if self.anchor is None or (self.last_pts_time is not None and pts_time < self.last_pts_time - 0.5):
            # first frame or a big backward jump: re-sync
            self.anchor = now - pts_time
        self.last_pts_time = pts_time
        target = self.anchor + pts_time
        wait = target - now
        if wait > 0:
            # coarse sleep (optional micro spin avoids oversleep)
            if wait > 0.005:
                time.sleep(wait - 0.002)
            while (self.anchor + pts_time) - time.perf_counter() > 0:
                pass  # tiny spin (~sub-ms)
