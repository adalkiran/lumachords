
import sys
import mido

from ..common import MidiRtOption
from .midi_rt_base_backend import MidiRtBaseBackend


class MidiRtMidoBackend(MidiRtBaseBackend):
    def __init__(self, port_num: int, port_name: str):
        super().__init__()
        self.port_num = port_num
        self.port_name = port_name
        self.out = mido.open_output(self.port_name)
        self.out.send(mido.Message("program_change", program=0, channel=0))

    @staticmethod
    def create(port_num: int = 0) -> tuple[MidiRtBaseBackend, str]:
        port_names = None
        try:
            mido.set_backend('mido.backends.rtmidi')
            port_names = mido.get_output_names() # We call here to test out existence of the related backend module
        except Exception:
            return None, 'No mido backend found: "mido.backends.rtmidi".'
        if not port_names:
            return None, "No MIDI output ports available for mido."
        try:
            port_num = 0
            backend = MidiRtMidoBackend(port_num, port_names[port_num])
            return backend, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def create_backend_options() -> list:
        backend, err = __class__.create(0)
        if err or not backend:
            return []
        backend.close()
        del backend
        port_names = mido.get_output_names()
        if sys.platform == "darwin":
            # On MacOS, the mido library decodes port names with mac_roman encoding, but it needs to be decoded with UTF-8
            try:
                port_names = [port_name.encode("mac_roman").decode("utf-8") for port_name in port_names]
            except:
                pass
        port_names = [port_name for port_name in port_names if not port_name.startswith("Fluid")]
        return [MidiRtOption(port_name, "mido", [port_num]) for port_num, port_name in enumerate(port_names)]

    def close(self) -> bool:
        try:
            self.out.close()
            return True
        except Exception:
            pass
        return False
    
    def send(self, event: mido.Message):
        self.out.send(event)

    def get_output_name(self) -> str:
        return self.port_name

    def get_successful_init_message(self) -> str:
        return f"MIDI output port {self.port_num} ({self.port_name}) has been opened."
    
    def get_successful_close_message(self) -> str:
        return f"MIDI output port {self.port_num} ({self.port_name}) has been closed."
