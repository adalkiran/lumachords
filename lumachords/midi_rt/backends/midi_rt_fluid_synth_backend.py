import os
import sys
import mido

from ..common import MidiRtOption
from .midi_rt_base_backend import MidiRtBaseBackend
from lumachords.utils import Utils


class MidiRtFluidsynthBackend(MidiRtBaseBackend):
    IS_BACKEND_AVAILABLE: bool = None
    AVAILABLE_BACKEND_NAME: str = None

    def __init__(self, sound_font_path: str, test_only: bool=False):
        import fluidsynth

        super().__init__()
        synth_kwargs = {}
        if test_only:
            synth_kwargs["synth.dynamic-sample-loading"] = 1
        synth = self.synth = fluidsynth.Synth(gain=1.0, **synth_kwargs)
        if not test_only:
            synth.start()
        self.sound_font_id = synth.sfload(sound_font_path)
        if self.sound_font_id < 0:
            raise Exception(f'FluidSynth could not load given SoundFont file: "{sound_font_path}"')
        self.sound_font_path = sound_font_path
        if not test_only:
            synth.program_select(0, self.sound_font_id, 0, 0)
            synth.set_reverb(1.0, 1.0, 100.0, 1.0)

    @staticmethod
    def is_required_on_platform() -> bool:
        # On Windows machines, no need to FluidSynth, Windows has "Microsoft GS Wavetable Synth" MIDI synthesizer.
        # And we don't need a multi-instrument and best quality sound.
        return not sys.platform.startswith("win")

    @staticmethod
    def create(test_only: bool=False) -> tuple[MidiRtBaseBackend, str]:
        if not __class__.is_required_on_platform():
            return None, "FluidSynth Python module not available: FluidSynth is not required on your platform"
        try:
            import fluidsynth # noqa
        except Exception as e:
            return None, f"FluidSynth Python module not available: {e}"

        sound_font_path = __class__.find_soundfont_path()
        if sound_font_path is None:
            return None, "FluidSynth library loaded, but it needs an .sf2 soundfont. No soundfont file found in searched directories. Set LUMACHORDS_SF2 to enable it."

        try:
            backend = MidiRtFluidsynthBackend(sound_font_path, test_only=test_only)
            return backend, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def create_backend_options() -> list[MidiRtOption]:
        if not __class__.is_required_on_platform():
            return []
        if __class__.IS_BACKEND_AVAILABLE is not None:
            return [MidiRtOption(__class__.AVAILABLE_BACKEND_NAME, "fluidsynth", [])] if __class__.IS_BACKEND_AVAILABLE else []
        backend, err = __class__.create(test_only=True)
        if err or not backend:
            if err:
                print(err)
            __class__.IS_BACKEND_AVAILABLE, __class__.AVAILABLE_BACKEND_NAME = False, None
            return __class__.create_backend_options()
        output_name = backend.get_output_name()
        backend.close()
        del backend
        __class__.IS_BACKEND_AVAILABLE, __class__.AVAILABLE_BACKEND_NAME = True, output_name
        return __class__.create_backend_options()

    @staticmethod
    def find_soundfont_path() -> str | None:
        env_paths = [
            os.getenv("LUMACHORDS_SF2"),
            os.getenv("FLUIDSYNTH_SF2"),
            os.getenv("SOUNDFONT"),
        ]
        for path in env_paths:
            if path and os.path.isfile(path):
                return path

        candidates = [
            Utils.resource_path("UprightPianoKW-small-bright-20190703.sf2", "resources"),
            "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def close(self) -> bool:
        try:
            self.synth.delete()
            return True
        except Exception:
            pass
        return False

    def send(self, event: mido.Message):
        if event.type == "note_on" or event.type == "note_off":
            velocity = event.velocity if event.type == "note_on" else 0
            if event.type == "note_on":
                self.synth.noteon(event.channel, int(event.note), int(velocity))
            else:
                self.synth.noteoff(event.channel, int(event.note))
        elif event.type == "control_change":
            self.synth.cc(event.channel, event.control, event.value)
        else:
            raise Exception(f"Not supported MIDI message type by {__class__.__name__}: {event.type}")

    def get_output_name(self) -> str:
        return "FluidSynth (Integrated)"
    
    def get_successful_init_message(self) -> str:
        return f'FluidSynth output initialized with soundfont: "{self.sound_font_path}".'
    
    def get_successful_close_message(self) -> str:
        return "FluidSynth backend has been closed."
