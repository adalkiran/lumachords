from dataclasses import dataclass


@dataclass
class MidiRtOption:
    title: str
    backend: str
    args: list
