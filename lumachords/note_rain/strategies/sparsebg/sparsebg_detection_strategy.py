import numpy as np

from lumachords.artifact_sink import ArtifactSink
from lumachords.data_types import NoteRainBoundaryLimits
from lumachords.hands_detector import HandsDetectorOutput
from lumachords.image_input import NoteRainImageInput
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.preferences import Preferences

from ..note_rain_detection_strategy import NoteRainDetectionStrategy
from .box_detector import BoxDetector


class SparseBgDetectionStrategy(NoteRainDetectionStrategy):
    def __init__(
        self,
        pref: Preferences,
        artifact_sink: ArtifactSink,
        keybed_output: KeybedDetectorOutput,
    ):
        self.box_detector = BoxDetector(pref, artifact_sink, keybed_output) 

    def get_note_rain_boundary_limits(self) -> NoteRainBoundaryLimits:
        return self.box_detector.note_rain_boundary_limits

    def get_keybed_output(self) -> KeybedDetectorOutput:
        return self.box_detector.keybed_output

    async def detect(
        self,
        nr_image_input: NoteRainImageInput,
        hands_output: HandsDetectorOutput,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        box_candidates = await self.box_detector.detect_boxes(nr_image_input, hands_output.hands_bgr if hands_output else None)
        obstacle_lines, coverage_tol = np.array([]), 0
        return box_candidates, obstacle_lines, coverage_tol
