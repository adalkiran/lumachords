from typing import Callable
import numpy as np

from lumachords.data_types import BackgroundType
from lumachords.hands_detector import HandsDetector, HandsDetectorOutputRanges, HandsType
from lumachords.processing_state import ProcessingState
from lumachords.runtime_config import AppMode, LogLevel, ProdMode, RuntimeConfig
from lumachords.image_input import NoteRainImageInput
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.preferences import Preferences
from lumachords.rendering import NoteRainRenderer
from lumachords.utils import Utils
from lumachords.artifact_sink import ArtifactConfigEntry, ArtifactSink

from .note_rain_tracker import NoteRainTracker
from .strategies.note_rain_detection_strategy import NoteRainDetectionStrategy
from .strategies.texturedbg.texturedbg_detection_strategy import TexturedBgDetectionStrategy
from .strategies.sparsebg.sparsebg_detection_strategy import SparseBgDetectionStrategy


class NoteRainPipeline:
    """Orchestrates the note rain detection pipeline using edge detection and boundary processing."""
    
    STATE_IMAGE_OUTPUT = 0
    STATE_IMAGE_LINES = 1
    STATE_IMAGE_BOXES = 2

    def __init__(self, pref: Preferences, runtime_config: RuntimeConfig, actual_fps: int, keybed_output: KeybedDetectorOutput, play_y_lag_time_delta_callback_fn: Callable[[float, float], None]=None, hands_type_callback_fn: Callable[[HandsType], None]=None):
        self.pref = pref
        self.runtime_config = runtime_config
        self.actual_fps = actual_fps
        self.keybed_output = keybed_output
        self.play_y_lag_time_delta_callback_fn = play_y_lag_time_delta_callback_fn
        self.hands_type_callback_fn = hands_type_callback_fn

        self.keybed_top_y = keybed_output.keybed_bounds[1]
        self.draw_box_tickness = int(np.ceil(self.keybed_output.white_key_default_width * pref.appearance.box_tickness_rate))

        # State management
        self.state = None
        
        # Playline configuration
        self.tracking_y0 = None
        self.tracking_y1 = None
        self.tracking_y_band_tuple = None
        self.play_y = None
        self.play_y_distance = None
        self.init_y_bands()
        
        self.hands_detector: HandsDetector = None
        self.detection_strategies: dict[BackgroundType, NoteRainDetectionStrategy] = None
        self.note_rain_tracker: NoteRainTracker = None
        
        self.artifact_sink: ArtifactSink = None

        self.init_artifact_sink()

    def init_artifact_sink(self):
        enable_extra_panels = (self.runtime_config.app_mode in [AppMode.GUI_ADVANCED, AppMode.NOTEBOOK])
        artifact_config = {
            "Output": ArtifactConfigEntry(
                ProdMode.PROD, 
                None, 
                emit_fn=lambda data:self.state.set_state_image(__class__.STATE_IMAGE_OUTPUT, data),
            ),
            "Lines": ArtifactConfigEntry(
                ProdMode.PROD,
                None,
                emit_fn=lambda data:self.state.set_state_image(__class__.STATE_IMAGE_LINES, data),
                filename="data/debug/final_lines.png",
                enabled=enable_extra_panels,
            ),
            "Boxes": ArtifactConfigEntry(
                ProdMode.PROD,
                None,
                emit_fn=lambda data:self.state.set_state_image(__class__.STATE_IMAGE_BOXES, data),
                filename="data/debug/final_boxes.png",
                enabled=enable_extra_panels,
            ),
            "Hands Skin Mask": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_DEBUG,
                filename="data/debug/hands_skin_mask.png"
            ),
            "Hands Mask": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_DEBUG,
                filename="data/debug/hands_mask.png"
            ),
            "Crop Luma": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_DEBUG,
                filename="data/debug/crop_luma.png"
            ),
            "Component": ArtifactConfigEntry(ProdMode.DEBUG, LogLevel.LOGLEVEL_VERBOSE),
            "Grads on Black Combined": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_DEBUG,
                filename="data/debug/grads_on_black_combined.png"
            ),
            "Labels Combined": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_DEBUG,
                filename="data/debug/labels_combined.png"
            ),
            "Line Detection Details": ArtifactConfigEntry(
                ProdMode.DEBUG,
                LogLevel.LOGLEVEL_INFO,
            ),
        }
        for axis in ["x", "y"]:
            artifact_config_axis = {
                f"Grads Normalized {axis}": ArtifactConfigEntry(
                    ProdMode.DEBUG, 
                    LogLevel.LOGLEVEL_DEBUG,
                    filename=f"data/debug/grads_normalized_{axis}.png"
                ),
                f"Grads on Black {axis}": ArtifactConfigEntry(
                    ProdMode.DEBUG, 
                    LogLevel.LOGLEVEL_DEBUG,
                    filename=f"data/debug/grads_on_black_{axis}.png"
                ),
                f"Grads on Image {axis}": ArtifactConfigEntry(
                    ProdMode.DEBUG, 
                    LogLevel.LOGLEVEL_DEBUG,
                    filename=f"data/debug/grads_on_image_{axis}.png"
                ),
                f"Positive Mask {axis}": ArtifactConfigEntry(
                    ProdMode.DEBUG, 
                    LogLevel.LOGLEVEL_DEBUG,
                    filename=f"data/debug/pos_mask_{axis}.png"
                ),
                f"Negative Mask {axis}": ArtifactConfigEntry(
                    ProdMode.DEBUG, 
                    LogLevel.LOGLEVEL_DEBUG,
                    filename=f"data/debug/neg_mask_{axis}.png"
                ),
                f"Labels {axis}": ArtifactConfigEntry(
                    ProdMode.DEBUG, 
                    LogLevel.LOGLEVEL_DEBUG,
                    filename=f"data/debug/labels_{axis}.png"
                ),
            }
            artifact_config = {**artifact_config, **artifact_config_axis}
        if self.artifact_sink:
            self.artifact_sink.config = artifact_config
            self.runtime_config = self.runtime_config
        else:
            self.artifact_sink = ArtifactSink(artifact_config, self.runtime_config)                

    def set_runtime_config(self, new_val: RuntimeConfig) -> ProcessingState:
        self.runtime_config = new_val
        self.init_artifact_sink()
        return self.init_state()

    def init_state(self):
        """Initializes the processing state for visualization."""
        panel_titles = ["OUTPUT", "DETECTED LINES", "DETECTED BOXES"]
        if self.runtime_config.app_mode == AppMode.GUI_BASIC:
            panel_titles = panel_titles[:1]
        self.state = ProcessingState.from_existing_state(self.state, panel_titles)
        
        # Initialize components
        if self.detection_strategies is None:
            self.hands_detector = HandsDetector(self.pref, self.keybed_output, self.artifact_sink, hands_type_callback_fn=self.hands_type_callback_fn)
            self.detection_strategies = {
                BackgroundType.TEXTURED: TexturedBgDetectionStrategy(self.pref, self.artifact_sink, self.keybed_output, self.state),
                BackgroundType.SPARSE: SparseBgDetectionStrategy(self.pref, self.artifact_sink, self.keybed_output),
            }
            self.note_rain_tracker = NoteRainTracker(self.pref, self.actual_fps, self.keybed_output, self.tracking_y_band_tuple, self.play_y, velocity_consensus_callback_fn=self.velocity_consensus_callback)
        return self.state

    def init_y_bands(self):
        """Initializes the playline band parameters."""
        self.tracking_y0 = int(self.keybed_top_y * self.pref.engine.tracking_band_top_offset_rate)
        self.tracking_y1 = int(self.keybed_top_y * (1 - self.pref.engine.tracking_band_bottom_offset_rate))
        self.tracking_y_band_tuple = (self.tracking_y0, self.tracking_y1)

        self.play_y = int(self.keybed_top_y * (1 - self.pref.engine.play_edge_vertical_offset_rate))
        self.play_y_distance = self.keybed_top_y - self.play_y
    
    def velocity_consensus_callback(self, velocity_consensus: float):
        #Calculate Lag through the keybed edge
        play_y_lag_pts = (self.play_y_distance / velocity_consensus) if velocity_consensus > 0 else 0
        play_y_lag_time_delta = Utils.pts_to_pts_time(play_y_lag_pts, self.actual_fps)
        if self.play_y_lag_time_delta_callback_fn:
            self.play_y_lag_time_delta_callback_fn(play_y_lag_time_delta, velocity_consensus)

    async def detect(self, pts: int, nr_image_input: NoteRainImageInput, transpose_octaves: int = 0, filter_for_tracking: bool=True) -> tuple[np.ndarray, HandsDetectorOutputRanges]:
        """Main detection pipeline: detects edges, pairs them into rectangles, and converts to NoteRect objects."""
        self.artifact_sink.emit("Output", nr_image_input.im_bgr)
        want_debug_data = self.artifact_sink.wants("Hands Skin Mask") or self.artifact_sink.wants("Hands Mask")
        hands_output = await self.hands_detector.detect(nr_image_input.im_crop_keybed_bgr, nr_image_input.crop_keybed_extra_height, transpose_octaves, return_debug=want_debug_data)
        self.artifact_sink.emit("Hands Skin Mask", hands_output.skin_mask)
        self.artifact_sink.emit("Hands Mask", hands_output.hands_mask)

        background_type, _ = nr_image_input.background_info
        detection_strategy = self.detection_strategies[background_type]
        box_candidates, obstacle_lines, coverage_tol = await detection_strategy.detect(nr_image_input, hands_output)
        box_candidates_cpy = box_candidates.copy()

        for i, box in enumerate(box_candidates):
            key_idx = self.keybed_output.find_best_fitting_slot(box)
            box_candidates["key_idx"][i] = key_idx

        boxes, raw_events = self.note_rain_tracker.step_frame(pts, box_candidates, coverage_tol, transpose_octaves, filter_for_tracking=filter_for_tracking)
        await self.artifact_sink.emit_lazy_async(
            "Line Detection Details", 
            lambda: f"""self.note_rain_tracker.step_frame
    cur_boxes = np.array([\n        {',\n        '.join([str(box) for box in box_candidates_cpy])}\n    ], dtype=DT_RECT)
    expected = np.array([\n        {',\n        '.join([str(box) for box in boxes])}\n    ], dtype=DT_RECT)
""" # noqa
                )
        del box_candidates_cpy
        del box_candidates
        boxes = boxes[boxes["is_valid"] > 0]


        
        # ======= VISUALIZATION =======
        await self.artifact_sink.emit_lazy_async(
            "Boxes",
            lambda: NoteRainRenderer.render_pipeline_overlay(
                nr_image_input.im_bgr.copy(),
                boxes,
                obstacle_lines,
                self.tracking_y_band_tuple,
                self.play_y,
                self.keybed_output.keybed_bounds,
                self.pref,
                self.draw_box_tickness,
            )
        )
        return raw_events, hands_output.ranges
