from abc import abstractmethod

import mido


class MidiRtBaseBackend:
    @staticmethod
    def is_required_on_platform() -> bool:
        return True

    @abstractmethod
    def close(self) -> bool:
        pass

    @abstractmethod
    def send(self, event: mido.Message):
        pass

    @abstractmethod
    def get_output_name(self) -> str:
        pass

    @abstractmethod
    def get_successful_init_message(self) -> str:
        pass

    @abstractmethod
    def get_successful_close_message(self) -> str:
        pass

    def is_playable(self) -> bool:
        return True
