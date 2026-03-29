from dataclasses import dataclass
from enum import IntEnum
import numpy as np

class AxisType(IntEnum):
    X = 0
    Y = 1

    def __str__(self):
        return "x" if self == AxisType.X else "y"

class BoxIsValid(IntEnum):
    Invalid = 0
    DetectedValid = 1
    EstimatedValid = 2

DT_LINE = np.dtype([
    ('id',   'i2'),
    ('x0',   'i2'),
    ('y0',   'i2'),
    ('x1',   'i2'),
    ('y1',   'i2'),
    ('tickness',   'i2'),
    ('is_start',   'bool'),
    ('axis', 'u1')
])

DT_TMP_RECT = np.dtype([
    ('id',   'i2'),
    ('x0',   'i2'),
    ('crop_y0',   'i2'),
    ('merged_y0',   'i2'),
    ('x1',   'i2'),
    ('crop_y1',   'i2'),
    ('merged_y1',   'i2'),
    ('top_line_idx', 'i2'),
    ('bottom_line_idx', 'i2'),
    ('snap_diff_top', 'i2'),
    ('snap_diff_bottom', 'i2'),
])

DT_RECT = np.dtype([
    ('id',   'i2'),
    ('x0',   'i2'),
    ('y0',   'i2'),
    ('x1',   'i2'),
    ('y1',   'i2'),
    ('key_idx', 'i2'),
    ('is_valid', 'i2'),
    ('snap_diff_top', 'i2'),
    ('snap_diff_bottom', 'i2'),
])

DT_BOX_EVENT = np.dtype([
    ('id',   'i2'),
    ('is_on', 'bool'),
    ('key_idx',   'i2'),
    ('midi_num', 'i2'),
    ('time_delta',   'f8'),
])

def as_structured_array(data, dtype, set_ids=False, id_start=0):
    if isinstance(data, np.ndarray) and data.dtype == dtype:
        arr = np.ascontiguousarray(data)
    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        arr = np.ascontiguousarray(data).astype(dtype, copy=False)
    else:
        arr = np.asarray(data)
        if arr.size == 0:
            arr = np.empty((0,), dtype=dtype)
        else:
            if arr.ndim == 1:
                if arr.shape[0] != len(dtype.names):
                    raise ValueError(f"Expected {len(dtype.names)} columns for {dtype}, got {arr.shape[0]}")
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != len(dtype.names):
                raise ValueError(f"Expected shape (N, {len(dtype.names)}) for {dtype}, got {arr.shape}")
            arr = np.ascontiguousarray(arr)
            out = np.empty(arr.shape[0], dtype=dtype)
            for i, name in enumerate(dtype.names):
                field_dtype = dtype.fields[name][0]
                out[name] = arr[:, i].astype(field_dtype, copy=False)
            arr = out

    if set_ids and arr.size:
        arr["id"] = np.arange(id_start, id_start + arr.shape[0], dtype=dtype.fields["id"][0])
    return arr

def as_dt_line(data, set_ids=False, id_start=0):
    return as_structured_array(data, DT_LINE, set_ids=set_ids, id_start=id_start)

def as_dt_tmp_rect(data, set_ids=False, id_start=0):
    return as_structured_array(data, DT_TMP_RECT, set_ids=set_ids, id_start=id_start)

def as_dt_rect(data, set_ids=False, id_start=0):
    return as_structured_array(data, DT_RECT, set_ids=set_ids, id_start=id_start)

@dataclass
class NoteEvent:
    time: float
    is_on: bool
    midi_num: int
    box_id: int

class NoteDuration(IntEnum):
    NONE = 0 # Special use
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHTH = 8
    SIXTEENTH = 16

class BackgroundType(IntEnum):
    TEXTURED = 1
    SPARSE = 2

    def __str__(self):
        if self == self.TEXTURED:
            return "TEXTURED"
        elif self == self.SPARSE:
            return "SPARSE"
        else:
            return "UNKNOWN"

@dataclass
class NoteRainBoundaryLimits:
    min_width: int
    max_width: int
    min_height: int
    final_min_width: int
    final_max_width: int
    final_min_height: int