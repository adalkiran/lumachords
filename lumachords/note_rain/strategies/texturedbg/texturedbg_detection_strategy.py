import numpy as np

from lumachords.artifact_sink import ArtifactSink
from lumachords.data_types import NoteRainBoundaryLimits
from lumachords.hands_detector import HandsDetectorOutput
from lumachords.image_input import NoteRainImageInput
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.preferences import Preferences
from lumachords.processing_state import ProcessingState

from ..note_rain_detection_strategy import NoteRainDetectionStrategy
from .boundary_detector import BoundaryDetector
from .edge_detector import EdgeDetector


class TexturedBgDetectionStrategy(NoteRainDetectionStrategy):
    def __init__(
        self,
        pref: Preferences,
        artifact_sink: ArtifactSink,
        keybed_output: KeybedDetectorOutput,
        state: ProcessingState,
    ):        
        self.edge_detector = EdgeDetector(pref, artifact_sink, keybed_output, state)
        self.boundary_detector = BoundaryDetector(pref, keybed_output)

    def get_note_rain_boundary_limits(self) -> NoteRainBoundaryLimits:
        return self.boundary_detector.note_rain_boundary_limits

    def get_keybed_output(self) -> KeybedDetectorOutput:
        return self.boundary_detector.keybed_output

    async def detect(
        self,
        nr_image_input: NoteRainImageInput,
        hands_output: HandsDetectorOutput,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        lines, lim_edge_tickness, im_crop_shape = await self.edge_detector.detect_lines(nr_image_input, hands_output.hands_bgr)
        # ======= PAIRING LINES INTO RECTANGLES =======
        box_candidates, obstacle_lines, coverage_tol = self.boundary_detector.pair_lines(lines, lim_edge_tickness, im_crop_shape)
        
        await self.edge_detector.artifact_sink.emit_lazy_async(
            "Line Detection Details", 
            lambda: f"""    lim_edge_tickness, im_crop_shape = {lim_edge_tickness}, {im_crop_shape}
    detector = NoteRainBoundaryDetector(Preferences(), KeybedDetectorOutput(None, None, None, {self.edge_detector.keybed_output.white_key_default_width}, {self.edge_detector.keybed_output.black_key_default_width}, None))
    lines = np.array([{', '.join([str(line) for line in lines])}], dtype=DT_LINE)
    expected = np.array([\n        {',\n        '.join([str(box) for box in box_candidates[box_candidates['is_valid'] > 0]])}\n    ], dtype=DT_RECT)
"""
                )
        box_candidates = box_candidates[box_candidates["is_valid"] > 0]
        return box_candidates, obstacle_lines, coverage_tol
