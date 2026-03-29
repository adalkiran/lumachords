from lumachords.utils import Utils

class AppearancePrefs:
    COLOR_WHITE = (255, 255, 255)
    COLOR_BEIGE = (237, 232, 208)
    COLOR_YELLOW = (255, 255, 32)
    COLOR_PURPLE = (255, 32, 255)
    COLOR_RED = (255, 32, 32)
    COLOR_GREEN = (32, 255, 32)
    COLOR_SALMON = (250,128,114)

    DEFAULT_WHITE_KEY_LINE_COLOR_RGB = COLOR_YELLOW
    DEFAULT_BLACK_KEY_LINE_COLOR_RGB = COLOR_PURPLE
    DEFAULT_FLOATING_BOX_BORDER_COLOR_RGB = COLOR_RED
    DEFAULT_FLOATING_ESTIMATED_BOX_BORDER_COLOR_RGB = COLOR_PURPLE
    DEFAULT_FLOATING_INVALID_BOX_BORDER_COLOR_RGB = COLOR_SALMON
    DEFAULT_KEYBED_BORDER_COLOR_RGB = COLOR_RED
    DEFAULT_KEYBED_BORDER_UNVALIDATED_COLOR_RGB = COLOR_YELLOW
    DEFAULT_NOTE_TEXT_COLOR_RGB = COLOR_RED
    DEFAULT_TRACKING_BAND_COLOR_RGB = COLOR_BEIGE
    DEFAULT_PLAY_EDGE_COLOR_RGB = COLOR_WHITE
    DEFAULT_START_LINE_COLOR_RGB = COLOR_GREEN
    DEFAULT_END_LINE_COLOR_RGB = COLOR_RED
    DEFAULT_BOX_TICKNESS_RATE = 0.05

    def __init__(self, 
        white_key_line_color_rgb, 
        black_key_line_color_rgb,
        floating_box_border_color_rgb,
        floating_estimated_box_border_color_rgb,
        floating_invalid_box_border_color_rgb,
        keybed_border_color_rgb,
        keybed_border_unvalidated_color_rgb,
        note_text_color_rgb,
        tracking_band_color_rgb,
        play_edge_color_rgb,
        start_line_color_rgb,
        end_line_color_rgb,
        box_tickness_rate, # multiplier for white_key_default_width
    ):
        cls = type(self)
        self.white_key_line_color_bgr = Utils.color_to_bgr(white_key_line_color_rgb or cls.DEFAULT_WHITE_KEY_LINE_COLOR_RGB)
        self.black_key_line_color_bgr = Utils.color_to_bgr(black_key_line_color_rgb or cls.DEFAULT_BLACK_KEY_LINE_COLOR_RGB)
        self.floating_box_border_color_bgr = Utils.color_to_bgr(floating_box_border_color_rgb or cls.DEFAULT_FLOATING_BOX_BORDER_COLOR_RGB)
        self.floating_estimated_box_border_color_bgr = Utils.color_to_bgr(floating_estimated_box_border_color_rgb or cls.DEFAULT_FLOATING_ESTIMATED_BOX_BORDER_COLOR_RGB)
        self.floating_invalid_box_border_color_bgr = Utils.color_to_bgr(floating_invalid_box_border_color_rgb or cls.DEFAULT_FLOATING_INVALID_BOX_BORDER_COLOR_RGB)
        self.keybed_border_color_bgr = Utils.color_to_bgr(keybed_border_color_rgb or cls.DEFAULT_KEYBED_BORDER_COLOR_RGB)
        self.keybed_border_unvalidated_color_bgr = Utils.color_to_bgr(keybed_border_unvalidated_color_rgb or cls.DEFAULT_KEYBED_BORDER_UNVALIDATED_COLOR_RGB)
        self.note_text_color_bgr = Utils.color_to_bgr(note_text_color_rgb or cls.DEFAULT_NOTE_TEXT_COLOR_RGB)
        self.tracking_band_color_bgr = Utils.color_to_bgr(tracking_band_color_rgb or cls.DEFAULT_TRACKING_BAND_COLOR_RGB)
        self.play_edge_color_bgr = Utils.color_to_bgr(play_edge_color_rgb or cls.DEFAULT_PLAY_EDGE_COLOR_RGB)
        self.start_line_color_bgr = Utils.color_to_bgr(start_line_color_rgb or cls.DEFAULT_START_LINE_COLOR_RGB)
        self.end_line_color_bgr = Utils.color_to_bgr(end_line_color_rgb or cls.DEFAULT_END_LINE_COLOR_RGB)
        self.box_tickness_rate = box_tickness_rate or cls.DEFAULT_BOX_TICKNESS_RATE

class EnginePrefs:
    DEFAULT_KEYBED_MAX_FFT_BINS = 256
    DEFAULT_KEYBED_MIN_SEGMENT_RATE = 0.5
    DEFAULT_KEYBED_MIN_SELECTED_FREQS_MEAN = 0.001
    DEFAULT_KEYBED_MIN_BOTTOM_RATE = 0.7
    DEFAULT_GRADIENT_THRESHOLD_STANDARD_DEVIATION_FACTOR = 3.0
    DEFAULT_EDGE_MORPHOLOGY_KERNEL_THICKNESS = 2
    DEFAULT_MAX_EDGE_TICKNESS_RATE = 0.6
    DEFAULT_MIN_EDGE_LENGTH_RATE_X = 0.01
    DEFAULT_MAX_EDGE_LENGTH_RATE_X = 0.05
    DEFAULT_MIN_EDGE_LENGTH_RATE_Y = 0.005
    DEFAULT_NOTE_RAIN_MIN_WIDTH_RATE = 0.65
    DEFAULT_NOTE_RAIN_MAX_WIDTH_RATE = 1.22
    DEFAULT_NOTE_RAIN_MIN_HEIGHT_RATE = 0.5
    DEFAULT_NOTE_RAIN_FINAL_MIN_WIDTH_RATE = 0.9
    DEFAULT_NOTE_RAIN_FINAL_MAX_WIDTH_RATE = 1.22
    DEFAULT_NOTE_RAIN_FINAL_MIN_HEIGHT_RATE = 0.7
    DEFAULT_NOTE_RAIN_GROUPING_PROXIMITY_LIMIT_RATE = 0.0025
    DEFAULT_NOTE_RAIN_VERTICAL_CONTINUITY_GAP_TOLERANCE_RATE = 0.02
    DEFAULT_NOTE_RAIN_HORIZONTAL_LINE_VERTICAL_SNAP_TOLERANCE_RATE = 0.01
    DEFAULT_NOTE_RAIN_HORIZONTAL_LINE_HORIZONTAL_SNAP_TOLERANCE_RATE = 0.3
    DEFAULT_NOTE_RAIN_OBSTACLE_BLOCKING_OVERLAP_RATE = 0.65
    DEFAULT_NOTE_RAIN_COVERAGE_TOLERANCE_RATE = 0.005
    DEFAULT_TRACKING_BAND_TOP_OFFSET_RATE = 0.25
    DEFAULT_TRACKING_BAND_BOTTOM_OFFSET_RATE = 0.15
    DEFAULT_PLAY_EDGE_VERTICAL_OFFSET_RATE = 0.35

    DEFAULT_VIDEO_FPS = 30
    DEFAULT_VIDEO_HEIGHT_LIMIT = None

    def __init__(self, 
        keybed_max_fft_bins,
        keybed_min_segment_rate,
        keybed_min_selected_freqs_mean,
        keybed_min_bottom_rate, #  multiplier for image height
        gradient_threshold_standard_deviation_factor, # multiplies of standard deviation
        edge_morphology_kernel_thickness, # value in pixels
        max_edge_tickness_rate, # multiplier for white_key_default_width
        min_edge_length_rate_x, # multiplier for image width
        max_edge_length_rate_x, # multiplier for image width
        min_edge_length_rate_y, # multiplier for image width
        note_rain_min_width_rate, # multiplier for min(white_key_default_width,black_key_default_width)
        note_rain_max_width_rate, # multiplier for max(white_key_default_width,black_key_default_width)
        note_rain_min_height_rate, # multiplier for note_rain_min_width
        note_rain_final_min_width_rate, # multiplier for min(white_key_default_width,black_key_default_width)
        note_rain_final_max_width_rate, # multiplier for max(white_key_default_width,black_key_default_width)
        note_rain_final_min_height_rate, # multiplier for note_rain_min_width
        note_rain_grouping_proximity_limit_rate, # multiplier for white_key_default_width
        note_rain_vertical_continuity_gap_tolerance_rate, # multiplier for image width
        note_rain_horizontal_line_vertical_snap_tolerance_rate, # multiplier for image width
        note_rain_horizontal_line_horizontal_snap_tolerance_rate, # multiplier for each found local_box
        note_rain_obstacle_blocking_overlap_rate, # multiplier for each found local_box
        note_rain_coverage_tolerance_rate, # multiplier for image width
        tracking_band_top_offset_rate, # multiplier for keybed_top_y
        tracking_band_bottom_offset_rate, # multiplier for keybed_top_y
        play_edge_vertical_offset_rate, # multiplier for keybed_top_y

        video_fps,
        video_height_limit,
    ):
        cls = type(self)
        self.keybed_max_fft_bins = keybed_max_fft_bins or cls.DEFAULT_KEYBED_MAX_FFT_BINS
        self.keybed_min_segment_rate = keybed_min_segment_rate or cls.DEFAULT_KEYBED_MIN_SEGMENT_RATE
        self.keybed_min_selected_freqs_mean = keybed_min_selected_freqs_mean or cls.DEFAULT_KEYBED_MIN_SELECTED_FREQS_MEAN
        self.keybed_min_bottom_rate = keybed_min_bottom_rate or cls.DEFAULT_KEYBED_MIN_BOTTOM_RATE
        self.gradient_threshold_standard_deviation_factor = gradient_threshold_standard_deviation_factor or cls.DEFAULT_GRADIENT_THRESHOLD_STANDARD_DEVIATION_FACTOR
        self.edge_morphology_kernel_thickness = edge_morphology_kernel_thickness or cls.DEFAULT_EDGE_MORPHOLOGY_KERNEL_THICKNESS
        self.max_edge_tickness_rate = max_edge_tickness_rate or cls.DEFAULT_MAX_EDGE_TICKNESS_RATE
        self.min_edge_length_rate_x = min_edge_length_rate_x or cls.DEFAULT_MIN_EDGE_LENGTH_RATE_X
        self.max_edge_length_rate_x = max_edge_length_rate_x or cls.DEFAULT_MAX_EDGE_LENGTH_RATE_X
        self.min_edge_length_rate_y = min_edge_length_rate_y or cls.DEFAULT_MIN_EDGE_LENGTH_RATE_Y
        self.note_rain_min_width_rate = note_rain_min_width_rate or cls.DEFAULT_NOTE_RAIN_MIN_WIDTH_RATE
        self.note_rain_max_width_rate = note_rain_max_width_rate or cls.DEFAULT_NOTE_RAIN_MAX_WIDTH_RATE
        self.note_rain_min_height_rate = note_rain_min_height_rate or cls.DEFAULT_NOTE_RAIN_MIN_HEIGHT_RATE
        self.note_rain_final_min_width_rate = note_rain_final_min_width_rate or cls.DEFAULT_NOTE_RAIN_FINAL_MIN_WIDTH_RATE
        self.note_rain_final_max_width_rate = note_rain_final_max_width_rate or cls.DEFAULT_NOTE_RAIN_FINAL_MAX_WIDTH_RATE
        self.note_rain_final_min_height_rate = note_rain_final_min_height_rate or cls.DEFAULT_NOTE_RAIN_FINAL_MIN_HEIGHT_RATE
        self.note_rain_grouping_proximity_limit_rate = note_rain_grouping_proximity_limit_rate or cls.DEFAULT_NOTE_RAIN_GROUPING_PROXIMITY_LIMIT_RATE
        self.note_rain_vertical_continuity_gap_tolerance_rate = note_rain_vertical_continuity_gap_tolerance_rate or cls.DEFAULT_NOTE_RAIN_VERTICAL_CONTINUITY_GAP_TOLERANCE_RATE
        self.note_rain_horizontal_line_vertical_snap_tolerance_rate = note_rain_horizontal_line_vertical_snap_tolerance_rate or cls.DEFAULT_NOTE_RAIN_HORIZONTAL_LINE_VERTICAL_SNAP_TOLERANCE_RATE
        self.note_rain_horizontal_line_horizontal_snap_tolerance_rate = note_rain_horizontal_line_horizontal_snap_tolerance_rate or cls.DEFAULT_NOTE_RAIN_HORIZONTAL_LINE_HORIZONTAL_SNAP_TOLERANCE_RATE
        self.note_rain_obstacle_blocking_overlap_rate = note_rain_obstacle_blocking_overlap_rate or cls.DEFAULT_NOTE_RAIN_OBSTACLE_BLOCKING_OVERLAP_RATE
        self.note_rain_coverage_tolerance_rate = note_rain_coverage_tolerance_rate or cls.DEFAULT_NOTE_RAIN_COVERAGE_TOLERANCE_RATE
        self.tracking_band_top_offset_rate = tracking_band_top_offset_rate or cls.DEFAULT_TRACKING_BAND_TOP_OFFSET_RATE
        self.tracking_band_bottom_offset_rate = tracking_band_bottom_offset_rate or cls.DEFAULT_TRACKING_BAND_BOTTOM_OFFSET_RATE
        self.play_edge_vertical_offset_rate = play_edge_vertical_offset_rate or cls.DEFAULT_PLAY_EDGE_VERTICAL_OFFSET_RATE

        self.video_fps = video_fps if video_fps is not None else cls.DEFAULT_VIDEO_FPS
        self.video_height_limit = video_height_limit if video_height_limit is not None else cls.DEFAULT_VIDEO_HEIGHT_LIMIT


class Preferences:
    

    def __init__(self, 
        # Args for appearance
        white_key_line_color_rgb = None, 
        black_key_line_color_rgb = None,
        floating_box_border_color_rgb = None,
        floating_estimated_box_border_color_rgb = None,
        floating_invalid_box_border_color_rgb = None,
        keybed_border_color_rgb = None,
        keybed_border_unvalidated_color_rgb = None,
        note_text_color_rgb = None,
        tracking_band_color_rgb = None,
        play_edge_color_rgb = None,
        start_line_color_rgb = None,
        end_line_color_rgb = None,
        box_tickness_rate = None, # multiplier for white_key_default_width

        # Args for engine
        keybed_max_fft_bins = None,
        keybed_min_segment_rate = None,
        keybed_min_selected_freqs_mean = None,
        keybed_min_bottom_rate = None, #  multiplier for image height
        gradient_threshold_standard_deviation_factor = None, # multiplies of standard deviation
        edge_morphology_kernel_thickness = None, # value in pixels
        max_edge_tickness_rate = None, # multiplier for white_key_default_width
        min_edge_length_rate_x = None, # multiplier for image width
        max_edge_length_rate_x = None, # multiplier for image width
        min_edge_length_rate_y = None, # multiplier for image width
        note_rain_min_width_rate = None, # multiplier for min(white_key_default_width,black_key_default_width)
        note_rain_max_width_rate = None, # multiplier for max(white_key_default_width,black_key_default_width)
        note_rain_min_height_rate = None, # multiplier for note_rain_min_width
        note_rain_final_min_width_rate = None, # multiplier for min(white_key_default_width,black_key_default_width)
        note_rain_final_max_width_rate = None, # multiplier for max(white_key_default_width,black_key_default_width)
        note_rain_final_min_height_rate = None, # multiplier for note_rain_min_width
        note_rain_grouping_proximity_limit_rate = None, # multiplier for white_key_default_width
        note_rain_vertical_continuity_gap_tolerance_rate = None, # multiplier for image width
        note_rain_horizontal_line_vertical_snap_tolerance_rate = None, # multiplier for image width
        note_rain_horizontal_line_horizontal_snap_tolerance_rate = None, # multiplier for each found local_box
        note_rain_obstacle_blocking_overlap_rate = None, # multiplier for each found local_box
        note_rain_coverage_tolerance_rate = None, # multiplier for image width
        tracking_band_top_offset_rate = None, # multiplier for keybed_top_y
        tracking_band_bottom_offset_rate = None, # multiplier for keybed_top_y
        play_edge_vertical_offset_rate = None, # multiplier for keybed_top_y

        video_fps = None,
        video_height_limit = None,
    ):
        self.appearance = AppearancePrefs(
            white_key_line_color_rgb=white_key_line_color_rgb,
            black_key_line_color_rgb=black_key_line_color_rgb,
            floating_box_border_color_rgb=floating_box_border_color_rgb,
            floating_estimated_box_border_color_rgb=floating_estimated_box_border_color_rgb,
            floating_invalid_box_border_color_rgb=floating_invalid_box_border_color_rgb,
            keybed_border_color_rgb=keybed_border_color_rgb,
            keybed_border_unvalidated_color_rgb=keybed_border_unvalidated_color_rgb,
            note_text_color_rgb=note_text_color_rgb,
            tracking_band_color_rgb=tracking_band_color_rgb,
            play_edge_color_rgb=play_edge_color_rgb,
            start_line_color_rgb=start_line_color_rgb,
            end_line_color_rgb=end_line_color_rgb,
            box_tickness_rate=box_tickness_rate,
        )
        self.engine = EnginePrefs(
            keybed_max_fft_bins=keybed_max_fft_bins,
            keybed_min_segment_rate=keybed_min_segment_rate,
            keybed_min_selected_freqs_mean=keybed_min_selected_freqs_mean,
            keybed_min_bottom_rate=keybed_min_bottom_rate,
            gradient_threshold_standard_deviation_factor=gradient_threshold_standard_deviation_factor,
            edge_morphology_kernel_thickness=edge_morphology_kernel_thickness,
            max_edge_tickness_rate=max_edge_tickness_rate,
            min_edge_length_rate_x=min_edge_length_rate_x,
            max_edge_length_rate_x=max_edge_length_rate_x,
            min_edge_length_rate_y=min_edge_length_rate_y,
            note_rain_min_width_rate=note_rain_min_width_rate,
            note_rain_max_width_rate=note_rain_max_width_rate,
            note_rain_min_height_rate=note_rain_min_height_rate,
            note_rain_final_min_width_rate=note_rain_final_min_width_rate,
            note_rain_final_max_width_rate=note_rain_final_max_width_rate,
            note_rain_final_min_height_rate=note_rain_final_min_height_rate,
            note_rain_grouping_proximity_limit_rate=note_rain_grouping_proximity_limit_rate,
            note_rain_vertical_continuity_gap_tolerance_rate=note_rain_vertical_continuity_gap_tolerance_rate,
            note_rain_horizontal_line_vertical_snap_tolerance_rate=note_rain_horizontal_line_vertical_snap_tolerance_rate,
            note_rain_horizontal_line_horizontal_snap_tolerance_rate=note_rain_horizontal_line_horizontal_snap_tolerance_rate,
            note_rain_obstacle_blocking_overlap_rate=note_rain_obstacle_blocking_overlap_rate,
            note_rain_coverage_tolerance_rate=note_rain_coverage_tolerance_rate,
            tracking_band_top_offset_rate=tracking_band_top_offset_rate,
            tracking_band_bottom_offset_rate=tracking_band_bottom_offset_rate,
            play_edge_vertical_offset_rate=play_edge_vertical_offset_rate,

            video_fps=video_fps,
            video_height_limit=video_height_limit,
        )
