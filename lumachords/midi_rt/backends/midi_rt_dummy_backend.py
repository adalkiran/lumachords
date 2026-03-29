import mido

from ..common import MidiRtOption
from .midi_rt_base_backend import MidiRtBaseBackend


class MidiRtDummyBackend(MidiRtBaseBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create() -> tuple[MidiRtBaseBackend, str]:
        return MidiRtDummyBackend()

    @staticmethod
    def create_backend_options() -> list:
        return [MidiRtOption("Don't play", "dummy", [])]

    def close(self) -> bool:
        pass
    
    def send(self, event: mido.Message):
        pass

    def get_output_name(self) -> str:
        return "Dummy"

    def get_successful_init_message(self) -> str:
        return "Not playing MIDI is selected. Sound will not be played!"
    
    def get_successful_close_message(self) -> str:
        return None

    def is_playable(self) -> bool:
        return False

