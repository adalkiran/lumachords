import asyncio

import numpy as np

from lumachords.data_types import DT_RECT, AxisType, BackgroundType
from lumachords.image_utils import ImageUtils
from lumachords.image_input import ImagePreprocessor
from lumachords.runtime_config import AppMode, LogLevel, ProdMode, RuntimeConfig
from lumachords.note_rain.strategies.texturedbg.texturedbg_detection_strategy import TexturedBgDetectionStrategy
from lumachords.preferences import Preferences
from lumachords.processor import Processor
from tests.utils import diff_items


# The following input sample images are captured/extracted from third-party YouTube videos for
# limited research/testing use purposes only. See experiment_samples/EXPERIMENT_SOURCES.txt
# and tests/data/TEST_SOURCES.txt for ownership and limited-use notes.


async def process_image_and_return_thr_async(im_file: str, prev_im_file: str, keybed_detection_log_level=LogLevel.LOGLEVEL_NONE, note_rain_detection_log_level=LogLevel.LOGLEVEL_NONE):
    pref = Preferences()
    keybed_runtime_config = RuntimeConfig(AppMode.HEADLESS, ProdMode.DEBUG, keybed_detection_log_level)
    note_rain_runtime_config = RuntimeConfig(AppMode.HEADLESS, ProdMode.DEBUG, note_rain_detection_log_level)
    processor = Processor(pref, keybed_runtime_config, note_rain_runtime_config, actual_fps=10, play_y_lag_time_delta_callback_fn=None)
    processor.init_keybed_detector_phase()
    
    im_bgr = ImageUtils.read_image(im_file)
    prev_im_bgr = ImageUtils.read_image(prev_im_file) if prev_im_file else None
    if prev_im_file and prev_im_bgr is None:
        raise Exception(f"prev_im_file is specified and could not be read: {prev_im_file}")
    kb_im_bgr = prev_im_bgr if prev_im_bgr is not None else im_bgr
    if kb_im_bgr is None:
        kb_im_file = im_file if im_bgr is None else prev_im_file
        raise Exception(f"Could not read image file: {kb_im_file}")
    
    kb_image_input = await ImagePreprocessor.preprocess_for_keybed(kb_im_bgr)

    keybed_output = await processor.detect_keys(kb_image_input)
    if not keybed_output.evaluation_result:
        if keybed_detection_log_level > LogLevel.LOGLEVEL_INFO:
            ImageUtils.imshow(kb_image_input.im_gray, "im_gray")

    if isinstance(keybed_output.evaluation_result, Exception):
        raise keybed_output.evaluation_result

    if keybed_output.evaluation_result is not None:
        raise Exception(keybed_output.evaluation_result)
    elif keybed_output.keybed_bounds is None:
        raise Exception("keybed_output.keybed_bounds is not valid.")

    processor.init_note_rain_pipeline_phase(keybed_output)
    nr_image_input = await ImagePreprocessor.preprocess_for_note_rain(im_bgr, processor.note_rain_pipeline.keybed_output.keybed_bounds, None)
    
    im_crop_luma, background_info = nr_image_input.im_crop_luma, nr_image_input.background_info
    (background_type, _) = background_info
    detection_strategy = processor.note_rain_pipeline.detection_strategies[background_type]
    if background_type == BackgroundType.TEXTURED:
        detection_strategy: TexturedBgDetectionStrategy = detection_strategy
        edge_detector = detection_strategy.edge_detector
        if im_crop_luma.ndim == 3:
            im_crop_luma = im_crop_luma[:,:,0]
        lim_edge_tickness, lim_edge_length_x, lim_edge_length_y = edge_detector.calculate_edge_limits(im_crop_luma.shape[1])
        pos_mask, neg_mask, grads, thr = edge_detector.get_gradient_mask_thresholded(im_crop_luma, background_type, AxisType.Y, lim_edge_length_y)
    else:
        thr = None
    return thr, background_info, nr_image_input, detection_strategy

def process_image_and_return_thr(im_file: str, prev_im_file: str):
    return asyncio.run(process_image_and_return_thr_async(im_file, prev_im_file, keybed_detection_log_level=LogLevel.LOGLEVEL_NONE, note_rain_detection_log_level=LogLevel.LOGLEVEL_NONE))

def test_shot_01_alman_dansi_sparsebg_strong_longline_letters():
    # YouTube link: https://www.youtube.com/watch?v=79MFcQJizto
    # Channel: KolayNota
    # Title: Haydn - Alman Dansı - Piyano

    # black background (background is RGB value <= 48)
    im_file = "tests/data/threshold/shot_01_alman_dansi_sparsebg_strong_longline_letters.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 48)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr

def test_shot_02_sparsebg_weak_longline():
    # black background (background is RGB value = 0)
    im_file = "tests/data/threshold/shot_02_sparsebg_weak_longline.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 0)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr


def test_shot_03_one_mans_dream_texturebg_smoke():
    # YouTube link: https://www.youtube.com/watch?v=VuRKmmpV35w
    # Channel: It's Piano
    # Title: One Man's Dream - Yanni | Piano | Synthesia | Relaxing music

    # non-stable background, have a colorful landscape bacground image
    # lower thresholds cause noise, so the threshold must be high (~392 is fine)
    im_file = "tests/data/threshold/shot_03_one_mans_dream_texturebg_smoke.png"
    expected_thr = 392.47927093506155
    expected_thr_min = expected_thr * 0.9
    expected_thr_max = expected_thr * 1.1
    expected_background_info = (BackgroundType.TEXTURED, None)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr_min <= actual_thr <= expected_thr_max

def test_shot_04_alman_dansi_sparsebg_strong_longline():
    # YouTube link: https://www.youtube.com/watch?v=79MFcQJizto
    # Channel: KolayNota
    # Title: Haydn - Alman Dansı - Piyano

    # black background (background is RGB value <= 48)
    im_file = "tests/data/threshold/shot_04_alman_dansi_sparsebg_strong_longline.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 48)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr

def test_shot_05_sparsebg_strong_longline():
    # black background (background is RGB value <= 31)
    im_file = "tests/data/threshold/shot_05_sparsebg_strong_longline.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 31)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr

def test_shot_06_senorita_sparsebg_smoke_letters():
    # YouTube link: https://www.youtube.com/watch?v=QU7BZhDZ8zY
    # Channel: Piano Go Life
    # Title: Shawn Mendes, Camila Cabello - Señorita Piano Tutorial

    # black background (background is RGB value <= 31)
    im_file = "tests/data/threshold/shot_06_senorita_sparsebg_smoke_letters.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 0)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr

def test_shot_07_gulpembe_sparsebg_strong_longline():
    # YouTube link: https://www.youtube.com/watch?v=NGbKJ0mS3bQ
    # Channel: Piano by VN
    # Title: Konser piyanisti GÜLPEMBE çalarsa :)

    # black background (background is RGB value <= 27) (background have a textured image, but near black)
    im_file = "tests/data/threshold/shot_07_gulpembe_sparsebg_strong_longline.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 28)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr

def test_shot_09_santa_lucia_texturebg_smoke():
    # YouTube link: https://www.youtube.com/watch?v=SjZxAaD1I10
    # Channel: It's Piano
    # Title: Santa Lucia - Teodoro Cottrau | Traditional Neapolitan Song | Piano Solo Synthesia Tutorial

    # non-stable background, have a colorful landscape bacground image
    # lower thresholds cause noise
    im_file = "tests/data/threshold/shot_09_santa_lucia_texturebg_smoke.png"
    expected_thr = 315.7442016601592
    expected_thr_min = expected_thr * 0.9
    expected_thr_max = expected_thr * 1.1
    expected_background_info = (BackgroundType.TEXTURED, None)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr_min <= actual_thr <= expected_thr_max

def test_shot_10_bittersuite_sparseng_no_longline():
    # YouTube link: https://www.youtube.com/watch?v=0yljm2qtUb8
    # Channel: Piano Pop Tv
    # Title: Billie Eilish - BITTERSUITE - Full Version (Piano Tutorial)

    # black background (background is RGB value <= 14)
    im_file = "tests/data/threshold/shot_10_bittersuite_sparseng_no_longline.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 14)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr

def test_shot_11_alman_dansi_sparsebg_strong_longline_letters():
    # YouTube link: https://www.youtube.com/watch?v=79MFcQJizto
    # Channel: KolayNota
    # Title: Haydn - Alman Dansı - Piyano

    # black background (background is RGB value <= 48)
    im_file = "tests/data/threshold/shot_11_alman_dansi_sparsebg_strong_longline_letters.png"
    expected_thr = None
    expected_background_info = (BackgroundType.SPARSE, 48)
    actual_thr, actual_background_info, _, _ = process_image_and_return_thr(im_file, None)
    assert expected_background_info == actual_background_info
    assert expected_thr == actual_thr



def test_edge_detection_shot_04_alman_dansi_sparsebg_strong_longline():
    # YouTube link: https://www.youtube.com/watch?v=79MFcQJizto
    # Channel: KolayNota
    # Title: Haydn - Alman Dansı - Piyano

    im_file = "tests/data/threshold/shot_04_alman_dansi_sparsebg_strong_longline.png"
    _, _, nr_image_input, detection_strategy = process_image_and_return_thr(im_file, None)

    expected = np.array([
        (-1, 1304, 14, 1373, 131, -1, 1, -999, -999),
        (-1, 549, 73, 617, 129, -1, 1, -999, -999),
        (-1, 1279, 130, 1320, 189, -1, 1, -999, -999),
        (-1, 549, 191, 617, 248, -1, 1, -999, -999),
        (-1, 1304, 192, 1373, 248, -1, 1, -999, -999),
        (-1, 1373, 250, 1442, 307, -1, 1, -999, -999),
        (-1, 1304, 309, 1373, 367, -1, 1, -999, -999),
        (-1, 549, 310, 617, 367, -1, 1, -999, -999),
        (-1, 1304, 428, 1373, 486, -1, 1, -999, -999),
        (-1, 687, 73, 755, 129, -1, 1, -999, -999),
        (-1, 688, 191, 755, 248, -1, 1, -999, -999),
        (-1, 687, 310, 753, 367, -1, 1, -999, -999),
    ], dtype=DT_RECT)
    actual, _, _ = asyncio.run(detection_strategy.detect(nr_image_input, None))
    ok, msg = diff_items(expected, actual)
    assert ok, msg

def test_edge_detection_shot_02_sparsebg_weak_longline():
    im_file = "tests/data/threshold/shot_02_sparsebg_weak_longline.png"
    _, _, nr_image_input, detection_strategy = process_image_and_return_thr(im_file, None)

    expected = np.array([
        (-1, 840, 27, 865, 150, -1, 1, -999, -999),
        (-1, 741, 39, 766, 71, -1, 1, -999, -999),
        (-1, 567, 46, 593, 110, -1, 1, -999, -999),
        (-1, 666, 106, 692, 150, -1, 1, -999, -999),
        (-1, 741, 136, 766, 189, -1, 1, -999, -999),
        (-1, 815, 153, 841, 190, -1, 1, -999, -999),
        (-1, 839, 184, 866, 229, -1, 1, -999, -999),
        (-1, 857, 219, 874, 269, -1, 1, -999, -999),
        (-1, 839, 260, 866, 308, -1, 1, -999, -999),
        (-1, 567, 173, 593, 308, -1, 1, -999, -999),
        (-1, 613, 175, 627, 227, -1, 1, -999, -999),
        (-1, 666, 224, 692, 268, -1, 1, -999, -999),
        (-1, 666, 307, 692, 347, -1, 1, -999, -999),
        (-1, 741, 335, 766, 382, -1, 1, -999, -999),
        (-1, 840, 358, 865, 384, -1, 1, -999, -999)
    ], dtype=DT_RECT)

    actual, _, _ = asyncio.run(detection_strategy.detect(nr_image_input, None))
    ok, msg = diff_items(expected, actual)
    assert ok, msg


def test_edge_detection_shot_05_sparsebg_strong_longline():
    im_file = "tests/data/threshold/shot_05_sparsebg_strong_longline.png"
    _, _, nr_image_input, detection_strategy = process_image_and_return_thr(im_file, None)

    expected = np.array([
        (-1, 328, 1, 377, 149, -1, 1, -999, -999),
        (-1, 1006, 0, 1057, 64, -1, 1, -999, -999),
        (-1, 1102, 69, 1153, 150, -1, 1, -999, -999),
        (-1, 1054, 141, 1106, 192, -1, 1, -999, -999),
        (-1, 1006, 178, 1058, 228, -1, 1, -999, -999),
        (-1, 1055, 230, 1106, 277, -1, 1, -999, -999),
        (-1, 1102, 268, 1151, 319, -1, 1, -999, -999),
        (-1, 1056, 312, 1106, 361, -1, 1, -999, -999)
    ], dtype=DT_RECT)
    actual, _, _ = asyncio.run(detection_strategy.detect(nr_image_input, None))
    ok, msg = diff_items(expected, actual)
    assert ok, msg

def test_edge_detection_shot_06_senorita_sparsebg_smoke_letters():
    # YouTube link: https://www.youtube.com/watch?v=QU7BZhDZ8zY
    # Channel: Piano Go Life
    # Title: Shawn Mendes, Camila Cabello - Señorita Piano Tutorial

    im_file = "tests/data/threshold/shot_06_senorita_sparsebg_smoke_letters.png"
    _, _, nr_image_input, detection_strategy = process_image_and_return_thr(im_file, None)

    expected = np.array([
        (-1, 442, 1, 479, 65, -1, 1, -999, -999),
        (-1, 959, 1, 996, 65, -1, 1, -999, -999),
        (-1, 1005, 94, 1027, 126, -1, 1, -999, -999),
        (-1, 812, 156, 851, 191, -1, 1, -999, -999),
        (-1, 922, 156, 961, 190, -1, 1, -999, -999),
        (-1, 1035, 158, 1071, 187, -1, 1, -999, -999),
        (-1, 1078, 214, 1109, 253, -1, 1, -999, -999),
        (-1, 1106, 242, 1143, 314, -1, 1, -999, -999),
        (-1, 740, 223, 776, 314, -1, 1, -999, -999),
        (-1, 998, 311, 1035, 376, -1, 1, -999, -999),
        (-1, 590, 344, 626, 375, -1, 1, -999, -999),
        (-1, 1006, 468, 1027, 499, -1, 1, -999, -999),
        (-1, 822, 469, 841, 499, -1, 1, -999, -999),
        (-1, 923, 468, 961, 501, -1, 1, -999, -999),
        (-1, 590, 474, 627, 625, -1, 1, -999, -999),
        (-1, 1071, 559, 1107, 626, -1, 1, -999, -999),
        (-1, 1106, 620, 1143, 688, -1, 1, -999, -999),
        (-1, 827, 660, 851, 684, -1, 1, -999, -999),
        (-1, 922, 655, 961, 689, -1, 1, -999, -999),
        (-1, 740, 695, 775, 814, -1, 1, -999, -999),
        (-1, 57, 770, 99, 807, -1, 1, -999, -999),
        (-1, 190, 770, 227, 814, -1, 1, -999, -999),
        (-1, 106, 773, 151, 807, -1, 1, -999, -999),
        (-1, 321, 773, 368, 814, -1, 1, -999, -999),
        (-1, 1815, 772, 1845, 808, -1, 1, -999, -999),
        (-1, 371, 775, 397, 807, -1, 1, -999, -999),
        (-1, 1744, 774, 1797, 814, -1, 1, -999, -999),
        (-1, 14, 776, 35, 815, -1, 1, -999, -999),
        (-1, 1526, 776, 1579, 814, -1, 1, -999, -999)
    ], dtype=DT_RECT)
    actual, _, _ = asyncio.run(detection_strategy.detect(nr_image_input, None))
    ok, msg = diff_items(expected, actual)
    assert ok, msg

def test_edge_detection_shot_11_alman_dansi_sparsebg_strong_longline_letters():
    # YouTube link: https://www.youtube.com/watch?v=79MFcQJizto
    # Channel: KolayNota
    # Title: Haydn - Alman Dansı - Piyano

    im_file = "tests/data/threshold/shot_11_alman_dansi_sparsebg_strong_longline_letters.png"
    _, _, nr_image_input, detection_strategy = process_image_and_return_thr(im_file, None)

    expected = np.array([
        (-1, 550, 0, 615, 25, -1, 1, -999, -999),
        (-1, 687, 0, 756, 25, -1, 1, -999, -999),
        (-1, 1030, 0, 1099, 26, -1, 1, -999, -999),
        (-1, 1000, 26, 1040, 84, -1, 1, -999, -999),
        (-1, 549, 86, 617, 142, -1, 1, -999, -999),
        (-1, 687, 86, 755, 143, -1, 1, -999, -999),
        (-1, 824, 86, 892, 143, -1, 1, -999, -999),
        (-1, 1030, 85, 1098, 143, -1, 1, -999, -999),
        (-1, 1000, 144, 1040, 202, -1, 1, -999, -999),
        (-1, 892, 204, 961, 262, -1, 1, -999, -999),
        (-1, 823, 263, 892, 499, -1, 1, -999, -999),
        (-1, 754, 322, 826, 381, -1, 1, -999, -999),
        (-1, 754, 441, 826, 499, -1, 1, -999, -999),
        (-1, 520, 323, 559, 380, -1, 1, -999, -999),
        (-1, 520, 441, 559, 499, -1, 1, -999, -999),
        (-1, 549, 560, 617, 617, -1, 1, -999, -999),
        (-1, 687, 560, 753, 617, -1, 1, -999, -999),
        (-1, 1030, 560, 1098, 618, -1, 1, -999, -999)
    ], dtype=DT_RECT)
    actual, _, _ = asyncio.run(detection_strategy.detect(nr_image_input, None))
    ok, msg = diff_items(expected, actual)
    assert ok, msg
