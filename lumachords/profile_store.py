from __future__ import annotations

import json
from pathlib import Path

from lumachords.midi_rt.midi_rt import MidiRt
from lumachords.utils import Utils
from lumachords.video import YoutubeDownloadService


class ProfileStore:
    FILE_NAME = "lumachords.profile.json"
    MAX_INPUTS = 20
    PROFILE_PATH = None
    _cache: dict | None = None

    @staticmethod
    def _default_data() -> dict:
        return {
            "version": 1,
            "global": {
                "mode": None,
                "continue_without_ffmpeg": False,
                "ffmpeg_binary_path": None,
                "last_open_dir": None,
                "last_save_dir": None,
                "midirt_device": None,
                "midirt_velocity": None,
                "midirt_use_pedal": True,
            },
            "inputs": [],
        }

    @staticmethod
    def _profile_path() -> Path:
        if __class__.PROFILE_PATH is not None:
            return __class__.PROFILE_PATH
        __class__.PROFILE_PATH = Utils.config_path() / __class__.FILE_NAME
        return __class__.PROFILE_PATH

    @staticmethod
    def load() -> dict:
        if __class__._cache is not None:
            return __class__._cache
        path = __class__._profile_path()
        if not path.is_file():
            __class__._cache = __class__._default_data()
            return __class__._cache
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Invalid profile root object")
            if not isinstance(data.get("global"), dict):
                data["global"] = {}
            if not isinstance(data.get("inputs"), list):
                data["inputs"] = []
            __class__._cache = data
        except Exception:
            __class__._cache = __class__._default_data()
        return __class__._cache

    @staticmethod
    def save() -> None:
        data = __class__.load()
        path = __class__._profile_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)

    @staticmethod
    def get_global_mode() -> str | None:
        data = __class__.load()
        mode = data.get("global", {}).get("mode")
        if mode and mode not in ["basic", "advanced"]:
            mode = None
        return mode

    @staticmethod
    def set_global_mode(mode: str) -> None:
        data = __class__.load()
        if mode and mode not in ["basic", "advanced"]:
            mode = None
        data.setdefault("global", {})["mode"] = mode
        __class__.save()

    @staticmethod
    def get_continue_without_ffmpeg() -> bool:
        data = __class__.load()
        return bool(data.get("global", {}).get("continue_without_ffmpeg"))

    @staticmethod
    def set_continue_without_ffmpeg(value: bool) -> None:
        data = __class__.load()
        data.setdefault("global", {})["continue_without_ffmpeg"] = bool(value)
        __class__.save()

    @staticmethod
    def get_ffmpeg_binary_path() -> str | None:
        data = __class__.load()
        value = data.get("global", {}).get("ffmpeg_binary_path")
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value else None

    @staticmethod
    def set_ffmpeg_binary_path(path: str | None) -> None:
        data = __class__.load()
        value = None
        if isinstance(path, str):
            path = path.strip()
            value = path if path else None
        data.setdefault("global", {})["ffmpeg_binary_path"] = value
        __class__.save()

    @staticmethod
    def get_last_open_dir() -> str | None:
        data = __class__.load()
        value = data.get("global", {}).get("last_open_dir")
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value else None

    @staticmethod
    def set_last_open_dir(path: str | None) -> None:
        data = __class__.load()
        value = None
        if isinstance(path, str):
            path = path.strip()
            value = path if path else None
        data.setdefault("global", {})["last_open_dir"] = value
        __class__.save()

    @staticmethod
    def get_last_save_dir() -> str | None:
        data = __class__.load()
        value = data.get("global", {}).get("last_save_dir")
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value else None

    @staticmethod
    def set_last_save_dir(path: str | None) -> None:
        data = __class__.load()
        value = None
        if isinstance(path, str):
            path = path.strip()
            value = path if path else None
        data.setdefault("global", {})["last_save_dir"] = value
        __class__.save()

    @staticmethod
    def get_midirt_device() -> tuple[str, str]:
        data = __class__.load()
        val = data.get("global", {}).get("midirt_device")
        if not val or "|" not in val:
            return None, None
        return val.split("|")[:2]

    @staticmethod
    def set_midirt_device(backend: str, title: str) -> None:
        data = __class__.load()
        data.setdefault("global", {})["midirt_device"] = f"{backend}|{title}"
        __class__.save()

    @staticmethod
    def get_midirt_velocity() -> int:
        data = __class__.load()
        try:
            return int(data.get("global", {}).get("midirt_velocity", MidiRt.DEFAULT_MIDIRT_VELOCITY))
        except:
            return MidiRt.DEFAULT_MIDIRT_VELOCITY

    @staticmethod
    def set_midirt_velocity(value: int) -> None:
        data = __class__.load()
        data.setdefault("global", {})["midirt_velocity"] = int(value)
        __class__.save()

    @staticmethod
    def get_midirt_use_pedal() -> bool:
        data = __class__.load()
        return bool(data.get("global", {}).get("midirt_use_pedal", True))

    @staticmethod
    def set_midirt_use_pedal(value: bool) -> None:
        data = __class__.load()
        data.setdefault("global", {})["midirt_use_pedal"] = bool(value)
        __class__.save()

    @staticmethod
    def get_input_profile(input_source: str, input_value: str) -> dict | None:
        input_key = __class__._build_input_key(input_source, input_value)
        if input_key is None:
            return None
        data = __class__.load()
        for item in data.get("inputs", []):
            if isinstance(item, dict) and item.get("key") == input_key:
                return item
        return None

    @staticmethod
    def set_input_profile(input_source: str, input_value: str, transpose_octaves: int, split_note: str) -> None:
        input_key = __class__._build_input_key(input_source, input_value)
        if input_key is None:
            return
        data = __class__.load()
        entries = data.get("inputs", [])
        if not isinstance(entries, list):
            entries = []

        entries = [item for item in entries if isinstance(item, dict) and item.get("key") != input_key]
        entries.append(
            {
                "key": input_key,
                "transpose_octaves": int(transpose_octaves) if transpose_octaves is not None else None,
                "split_note": split_note or "C4",
            }
        )
        if len(entries) > __class__.MAX_INPUTS:
            entries = entries[-__class__.MAX_INPUTS:]

        data["inputs"] = entries
        __class__.save()

    @staticmethod
    def _build_input_key(input_source: str, input_value: str) -> str | None:
        source = (input_source or "").strip().lower()
        value = (input_value or "").strip()
        if not value:
            return None
        if source == "file":
            return f"file:{Path(value).stem}"
        if source == "youtube":
            video_id = YoutubeDownloadService.extract_youtube_id(value)
            return f"youtube:{video_id}" if video_id else None
        return None
