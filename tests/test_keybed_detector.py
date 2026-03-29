import numpy as np
from lumachords.runtime_config import AppMode, LogLevel, ProdMode, RuntimeConfig
from lumachords.preferences import Preferences
from lumachords.keybed_detector import KeybedDetector


def test_segment_selection():
    seg_scores = np.array([
        -394.1552801132202, -1.1608062982559204, -1.275471806526184, -0.7885557413101196, -0.44983944296836853,
        -0.32114875316619873, -0.6851812452077866, 267.35602259635925, 0.8003626763820648, 0.3732251822948456, 
        1.0183686017990112, 134.32968616485596, 1.3044178485870361, 34.01245188713074, -8.876487493515015, 
        -2.710684597492218, -np.inf, -7.695389270782471
    ])
    best_idx=7
    expected = (np.int64(7), np.int64(13))
    detector = KeybedDetector(Preferences(), RuntimeConfig(AppMode.NOTEBOOK, ProdMode.DEBUG, LogLevel.LOGLEVEL_VERBOSE))
    actual = detector.enhance_keybed_segment_selection(seg_scores, best_idx)
    assert expected == actual
