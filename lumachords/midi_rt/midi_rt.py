import asyncio
import time
import threading
import heapq
import mido

from lumachords.midi_rt.backends import (
    MidiRtBaseBackend,
    MidiRtDummyBackend,
    MidiRtFluidsynthBackend,
    MidiRtMidoBackend,
)

from .common import MidiRtOption
from lumachords.data_types import NoteEvent


class MidiRt:
    DEFAULT_MIDIRT_VELOCITY = 50
    
    def __init__(self, midirt_option: MidiRtOption, velocity=50, use_pedal=True):
        self.midirt_option = midirt_option
        self.velocity = max(min(velocity, 127), 20)
        # Pedal effect is normally depends on the note's duration and velocity, but here we just use a constant lag in seconds.
        self.pedal_lag = 0.5 if use_pedal else 0.0
        self.backend: MidiRtBaseBackend = None
        self.on_notes = {}
        self._on_notes_lock = threading.Lock()

        # current video time (seconds) provided externally
        self.pts_time: float = float("-inf")

        # queued events (time-ordered)
        self._heap = []  # (time, seq, NoteEvent)
        self._seq = 0
        self._wake = None            # asyncio.Event, created lazily in running loop
        self._task = None            # asyncio.Task
        self._closed = False

        if self.initialize_backend():
            print(self.backend.get_successful_init_message())
        else:
            print("No MIDI output ports or FluidSynth backend available. Sound will not be played!")

    @staticmethod
    def get_output_port_names():
        return mido.get_output_names()

    @staticmethod
    def create_backend_options() -> list[MidiRtOption]:
        return (
            MidiRtFluidsynthBackend.create_backend_options() + 
            MidiRtMidoBackend.create_backend_options() + 
            MidiRtDummyBackend.create_backend_options()
        )

    def initialize_backend(self) -> bool:
        self.backend = None
        if self.midirt_option:
            if self.midirt_option.backend == "fluidsynth" and MidiRtFluidsynthBackend.is_required_on_platform():
                try:
                    backend, err = MidiRtFluidsynthBackend.create(*(self.midirt_option.args or []))
                    if not err:
                        self.backend = backend
                        return True
                except:
                    pass
            elif self.midirt_option.backend == "mido":
                try:
                    backend, err = MidiRtMidoBackend.create(*self.midirt_option.args)
                    if not err:
                        self.backend = backend
                        return True
                except:
                    pass
            elif self.midirt_option.backend == "dummy":
                self.backend = MidiRtDummyBackend.create()
                return True
        if self.backend is None:
            if self.midirt_option:
                print("Selected MIDI output option is not functionable now. Falling back to other alternatives...")
            if MidiRtFluidsynthBackend.is_required_on_platform():
                try:
                    backend, err = MidiRtFluidsynthBackend.create()
                    if not err:
                        self.backend = backend
                        return True
                except:
                    pass
            try:
                port_names = __class__.get_output_port_names()
                if len(port_names):
                    port_num = 0
                    backend, err = MidiRtMidoBackend.create(port_num)
                    if not err:
                        self.backend = backend
                        return True
            except:
                pass
        self.backend = MidiRtDummyBackend.create()
        return False

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def close(self):
        """Stop worker and close the MIDI port (best-effort, non-async)."""
        self.mute_all()
        self._closed = True
        if self._wake is not None:
            try:
                self._wake.set()
            except Exception:
                pass
        if self._task is not None:
            try:
                self._task.cancel()
            except Exception:
                pass
        try:
            if self.backend.close():
                close_message = self.backend.get_successful_close_message()
                if close_message:
                    print(close_message)
        except Exception:
            pass
        if self.backend:
            del self.backend
            self.backend = None

    def set_pts_time(self, pts_time: float):
        """Set current video time in seconds; worker will send any due queued events."""
        self.pts_time = float(pts_time)
        if self._wake is not None:
            self._wake.set()

    def _ensure_worker(self):
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        if self._wake is None:
            self._wake = asyncio.Event()
        self._task = loop.create_task(self._worker())

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while not self._closed:
            # send everything due as of current pts_time
            while self._heap and self._heap[0][0] <= self.pts_time and not self._closed:
                _, _, e = heapq.heappop(self._heap)
                vel = self.velocity if e.is_on else 0
                try:
                    with self._on_notes_lock:
                        if e.is_on:
                            self.on_notes[int(e.midi_num)] = True
                        elif int(e.midi_num) in self.on_notes:
                            del self.on_notes[int(e.midi_num)]
                    await loop.run_in_executor(None, self.send_note_event, int(e.midi_num), vel)
                except Exception:
                    # If the port breaks, stop trying.
                    self._closed = True
                    break

            if self._closed:
                break

            # wait until either pts_time advances or new events arrive
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=0.1)
            except:
                pass

    def play(self, new_events: list[NoteEvent]):
        if not self.backend.is_playable() or not new_events:
            return

        self._ensure_worker()

        for e in new_events:
            self._seq += 1
            # Pedal effect is normally depends on the note's duration and velocity, but here we just use a constant lag in seconds.
            heapq.heappush(self._heap, (float(e.time + (0 if e.is_on else self.pedal_lag)), self._seq, e))

        self._wake.set()

    def mute_all(self):
        if not self.backend.is_playable():
            return
        with self._on_notes_lock:
            notes_to_off = [int(midi_num) for midi_num in self.on_notes.keys()]
            self.on_notes = {}
        for midi_num in notes_to_off:
            self.send_note_event(midi_num, 0, force_off=True)
        self.backend.send(mido.Message('control_change', channel=0, control=120, value=0)) # All Sound Off
        self.backend.send(mido.Message('control_change', channel=0, control=121, value=0)) # Reset All Controllers
        self.backend.send(mido.Message('control_change', channel=0, control=123, value=0)) # All Notes Off
        self.backend.send(mido.Message('control_change', channel=0, control=64, value=0)) # Sustain off
        time.sleep(0)

    def send_note_event(self, midi_num: int, velocity: int, force_off: bool=False):
        if self.backend.is_playable():
            message_type = "note_off" if (velocity == 0 and force_off) else "note_on"
            self.backend.send(mido.Message(message_type, note=int(midi_num), velocity=int(velocity), channel=0))
