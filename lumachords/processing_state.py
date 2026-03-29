from dataclasses import dataclass

import numpy as np


@dataclass
class ProcessingStateItem:
    data: any = None
    cache: any = None
    image: np.ndarray = None

class ProcessingState:
    IDX_SPECIAL_START = 10_000
    IDX_MIDI_EVENTS = IDX_SPECIAL_START + 1
    IDX_ACTIVE_NOTES = IDX_SPECIAL_START + 2
    IDX_INFO = IDX_SPECIAL_START + 3

    def __init__(self, panel_titles: list[str]):
        self.panel_titles = panel_titles
        self.states = {i: ProcessingStateItem() for i in range(len(self.panel_titles))}
        self.states[__class__.IDX_ACTIVE_NOTES] = ProcessingStateItem()
        self.states[__class__.IDX_INFO] = ProcessingStateItem()
        self.midi_events_str = None
        self.info_panel_str = None

    def get_state(self, index: int):
        assert index in self.states
        return self.states[index]

    def set_state(self, index: int, data: any, image: np.ndarray, cache: any = None):
        assert index in self.states
        item = self.states[index]
        item.data = data
        item.image = image
        item.cache = cache if cache is not None else image

    def set_state_image(self, index: int, image: np.ndarray):
        assert index in self.states
        item = self.states[index]
        item.image = image

    def inherit_from(self, existing_state: 'ProcessingState'):
        if existing_state:
            self.info_panel_str = existing_state.info_panel_str

    @staticmethod
    def from_existing_state(existing_state: 'ProcessingState', panel_titles: list[str]):
        new_state = ProcessingState(panel_titles)
        if existing_state:
            for key, val in existing_state.states.items():
                if key in new_state.states:
                    new_state.states[key] = val
        return new_state
