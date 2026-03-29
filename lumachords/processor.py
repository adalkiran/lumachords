from typing import Callable

import numpy as np
from lumachords.hands_detector import HandsDetectorOutputRanges, HandsType
from lumachords.processing_state import ProcessingState
from lumachords.runtime_config import RuntimeConfig
from lumachords.image_input import KeybedImageInput, NoteRainImageInput
from lumachords.keybed_detector import KeybedDetector, KeybedDetectorOutput
from lumachords.note_rain import NoteRainPipeline
from lumachords.preferences import Preferences

class Processor:
    def __init__(self, pref: Preferences, keybed_runtime_config: RuntimeConfig, note_rain_runtime_config: RuntimeConfig, actual_fps: int, play_y_lag_time_delta_callback_fn: Callable[[float, float], None]=None, hands_type_callback_fn: Callable[[HandsType], None]=None):
        self.pref = pref
        self.keybed_runtime_config = keybed_runtime_config
        self.note_rain_runtime_config = note_rain_runtime_config
        self.actual_fps = actual_fps
        self.play_y_lag_time_delta_callback_fn = play_y_lag_time_delta_callback_fn
        self.hands_type_callback_fn = hands_type_callback_fn
        self.keybed_detector: KeybedDetector = None
        self.note_rain_pipeline: NoteRainPipeline = None
    
    def init_keybed_detector_phase(self) -> ProcessingState:
        self.keybed_detector = KeybedDetector(self.pref, self.keybed_runtime_config)
        return self.keybed_detector.init_state()

    def init_note_rain_pipeline_phase(self, keybed_output) -> ProcessingState:
        self.note_rain_pipeline = NoteRainPipeline(self.pref, self.note_rain_runtime_config, self.actual_fps, keybed_output, play_y_lag_time_delta_callback_fn=self.play_y_lag_time_delta_callback_fn, hands_type_callback_fn=self.hands_type_callback_fn)
        return self.note_rain_pipeline.init_state()

    def emit_keybed_detection_result(self, keybed_output: KeybedDetectorOutput, pts=None):
        if not self.keybed_detector.artifact_sink.wants("Keybed Detection Result"):
            return
        messages = [
            f"Total key count: {len(keybed_output.all_keys_data)}",
            f"White key count: {len([key_data for key_data in keybed_output.all_keys_data if key_data["color"] == "w"])}",
            f"Black key count: {len([key_data for key_data in keybed_output.all_keys_data if key_data["color"] == "b"])}",
            f"Starts with note {keybed_output.all_keys_data[0]["note_name"]}",
            f"Ends with note {keybed_output.all_keys_data[-1]["note_name"]}",
            f"White key default width: {keybed_output.white_key_default_width}",
            f"Black key default width: {keybed_output.black_key_default_width}",
        ]
        pts_str = f" for PTS={pts}" if pts is not None else ""
        result = f"Keybed detection has been succeeded{pts_str}!\n{"\n".join(messages)}"
        self.keybed_detector.artifact_sink.emit("Keybed Detection Result", result)

    async def detect_keys(self, kb_image_input: KeybedImageInput, pts=None) -> KeybedDetectorOutput:
        keybed_output = await self.keybed_detector.detect(kb_image_input)
        if keybed_output.evaluation_result:
            return keybed_output
        self.emit_keybed_detection_result(keybed_output, pts=pts)
        return keybed_output
    
    async def detect_note_rain(self, pts: int, nr_image_input: NoteRainImageInput, transpose_octaves = 0, filter_for_tracking: bool=True) -> tuple[np.ndarray, HandsDetectorOutputRanges]:
        if not self.note_rain_pipeline:
            raise Exception("note_rain_pipeline is not defined, call detect_keys then init_note_rain_pipeline_phase first.")
        raw_events, hands_output_ranges = await self.note_rain_pipeline.detect(pts, nr_image_input, transpose_octaves=transpose_octaves, filter_for_tracking=filter_for_tracking)
        return raw_events, hands_output_ranges

    def get_current_runtime_config(self):
        if self.note_rain_pipeline:
            return self.note_rain_pipeline.runtime_config
        if self.keybed_detector:
            return self.keybed_detector.runtime_config
        raise Exception("No active detector got found.")

    def set_current_runtime_config(self, new_val: RuntimeConfig) -> ProcessingState:
        self.note_rain_runtime_config = new_val
        self.keybed_runtime_config = new_val
        if self.note_rain_pipeline:
            return self.note_rain_pipeline.set_runtime_config(new_val)
        if self.keybed_detector:
            return self.keybed_detector.set_runtime_config(new_val)
        raise Exception("No active detector got found.")
