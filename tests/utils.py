import numpy as np


def pack(x):
    x = np.ascontiguousarray(x)
    if x.ndim == 1 or x.dtype.names is not None:
        return x.view(np.dtype((np.void, x.dtype.itemsize)))
    x = x.reshape(x.shape[0], -1)
    return x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))

def diff_items(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    e, a = pack(expected), pack(actual)
    expected_diff, actual_diff = expected[(~np.isin(e, a)).ravel()], actual[(~np.isin(a, e)).ravel()]
    if len(expected_diff) == 0 and len(actual_diff) == 0:
        return True, ""
    return False, f"Failed. Expected contains:\n{expected_diff}\nActual contains:\n{actual_diff}"
