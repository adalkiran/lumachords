from lumachords.data_types import NoteRainBoundaryLimits
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.preferences import Preferences


class NoteRainUtils:
    @staticmethod
    def calculate_boundary_limits(pref: Preferences, keybed_output: KeybedDetectorOutput):
        min_width = int(min(keybed_output.white_key_default_width, keybed_output.black_key_default_width) * pref.engine.note_rain_min_width_rate)
        max_width = int(max(keybed_output.white_key_default_width, keybed_output.black_key_default_width) * pref.engine.note_rain_max_width_rate)
        min_height = int(min_width * pref.engine.note_rain_min_height_rate)
        final_min_width = int(min(keybed_output.white_key_default_width, keybed_output.black_key_default_width) * pref.engine.note_rain_final_min_width_rate)
        final_max_width = int(max(keybed_output.white_key_default_width, keybed_output.black_key_default_width) * pref.engine.note_rain_final_max_width_rate)
        final_min_height = int(final_min_width * pref.engine.note_rain_final_min_height_rate)
        return NoteRainBoundaryLimits(
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            final_min_width=final_min_width,
            final_max_width=final_max_width,
            final_min_height=final_min_height,
        )
