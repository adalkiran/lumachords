from abc import abstractmethod

import numpy as np

from lumachords.data_types import NoteRainBoundaryLimits
from lumachords.hands_detector import HandsDetectorOutput
from lumachords.image_input import NoteRainImageInput
from lumachords.keybed_detector import KeybedDetectorOutput


class NoteRainDetectionStrategy:
    """Common contract for note-rain detection strategies."""
    
    @abstractmethod
    def get_note_rain_boundary_limits(self) -> NoteRainBoundaryLimits:
        raise Exception("Not implemented.")

    @abstractmethod
    def get_keybed_output(self) -> KeybedDetectorOutput:
        raise Exception("Not implemented.")

    @abstractmethod
    async def detect(
        self,
        nr_image_input: NoteRainImageInput,
        hands_output: HandsDetectorOutput,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Detect note-rain candidates for a frame.

        Returns:
            tuple:
                box_candidates: Structured rectangle array.
                obstacle_lines: Structured line array or empty array.
                coverage_tol: Coverage tolerance used by downstream tracking.
        """
        raise Exception("Not implemented.")
