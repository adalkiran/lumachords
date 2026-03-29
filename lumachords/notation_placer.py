import itertools
import numpy as np
import xml.etree.ElementTree as ET
import copy

from lumachords.staff_mapping import StaffMapping
from lumachords.hands_detector import HandsMidiNumRangesPerTime
from lumachords.data_types import NoteDuration
from lumachords.timing_estimator import TimingEstimator
from lumachords.notation_renderer import NotationRenderer


# XML namespace constants for xml:id handling
XML_NS = ""# "http://www.w3.org/XML/1998/namespace"
XML_ID = f"{{{XML_NS}}}id"

DT_PAIR_ABS_TICK_EXT = np.dtype([
    ("staff", "i1"),
    ("midi_num", "i2"),
    ("on_bar", "i4"),
    ("on_abs_tick", "i8"),
    ("off_bar", "i4"),
    ("off_abs_tick", "i8"),
])

DT_PAIR_SECS = np.dtype([
    ("hands_idx", "i8"),
    ("staff", "i1"),
    ("midi_num", "i2"),
    ("on_secs", "float"),
    ("off_secs", "float"),
])

DT_PAIR_ABS_TICK = np.dtype([
    ("staff", "i1"),
    ("midi_num", "i2"),
    ("on_abs_tick", "i8"),
    ("off_abs_tick", "i8"),
])


DT_PAIR_BBDT = np.dtype([
    ("staff", "i1"),
    ("midi_num", "i2"),
    ("on_abs_tick", "i8"),
    ("on_bar", "i4"),
    ("on_beat", "i1"),
    ("on_div", "i1"),
    ("on_tick", "i2"),
    ("off_abs_tick", "i8"),
    ("off_bar", "i4"),
    ("off_beat", "i1"),
    ("off_div", "i1"),
    ("off_tick", "i2"),
    ("tie", "u1"),
])


class NotationPlacer:
    SECONDS_PER_MINUTE = 60.0

    PITCHES = [ #name, note_num, is_sharp
        ("C",  0, False),
        ("C#", 0, True),
        ("D",  1, False),
        ("D#", 1, True),
        ("E",  2, False),
        ("F",  3, False),
        ("F#", 3, True),
        ("G",  4, False),
        ("G#", 4, True),
        ("A",  5, False),
        ("A#", 5, True),
        ("B",  6, False)
    ]

    DURATION_NAME = {
        16: (NoteDuration.WHOLE, 0),
        12: (NoteDuration.HALF, 1),       # dotted-half
        8:  (NoteDuration.HALF, 0),
        6:  (NoteDuration.QUARTER, 1),    # dotted-quarter
        4:  (NoteDuration.QUARTER, 0),
        3:  (NoteDuration.EIGHTH, 1),     # dotted-eighth
        2:  (NoteDuration.EIGHTH, 0),
        1:  (NoteDuration.SIXTEENTH, 0),
    }

    DURATION_NAME_INV = {v: k for k, v in DURATION_NAME.items()}

    def __init__(self, 
                 crop_silence_at_start: bool, 
                 take_latest_measures: int = None, 
                 default_split_midi_num=60,
                 always_use_latest_hands_range: bool=False,
                 min_measures: int = 4, 
                 time_sig: tuple[int, int] = (4, 4),
                 bpm=120.0, 
                 ppq=960,
                 divs_per_beat=4, 
                 onset_tolerance_beats=0.2,
                 include_image_alpha_channel=True,
                 print_timing: bool=True,
                 auto_timing: bool=True,
        ):
        self.crop_silence_at_start = crop_silence_at_start
        self.take_latest_measures = take_latest_measures
        self.default_split_midi_num = default_split_midi_num
        self.always_use_latest_hands_range = always_use_latest_hands_range
        self.min_measures = min_measures
        self.time_sig = time_sig
        self.bpm = bpm
        self.ppq = ppq
        self.divs_per_beat = divs_per_beat
        self.onset_tolerance_beats = onset_tolerance_beats
        self.include_image_alpha_channel = include_image_alpha_channel
        self.print_timing = print_timing

        # STATEFUL MEMBERS
        self.hands_midi_num_ranges_per_time: HandsMidiNumRangesPerTime = None
        self.pairs_bbdt: np.ndarray = None
        self.clefs_per_bar: dict[int, list[str]] = {}
        self.crop_abs_ticks = 0
        self.estimator = TimingEstimator() if auto_timing else None
        self.last_estimation_result = None

        # CALCULATED MEMBERS
        self.recalculate_timing_members(onset_tolerance_beats=onset_tolerance_beats)

    def recalculate_timing_members(self, onset_tolerance_beats: float = 0.2):
        self.onset_tolerance_beats = onset_tolerance_beats
        self.beats_per_bar = self.time_sig[0]
        self.ticks_per_bar = self.beats_per_bar * self.ppq
        self.ticks_per_div = self.ppq // self.divs_per_beat
        self.secs_per_beat = __class__.SECONDS_PER_MINUTE / self.bpm
        self.secs_per_bar = self.secs_per_beat * self.beats_per_bar
        self.onset_tolerance_ticks = round(onset_tolerance_beats * self.ppq)
    
    @staticmethod
    async def check_is_ready():
        await NotationRenderer.init()

    def copy(self):
        return copy.copy(self)

    def pitch_from_midi_num(self, midi_num: int):
        if not isinstance(midi_num, int) or not (0 <= midi_num <= 127):
            raise ValueError("midi_num must be an int in 0..127")

        pitch = midi_num % 12
        name, note_num, is_sharp = __class__.PITCHES[pitch]
        octave = (midi_num // 12) - 1

        return name, note_num, octave, is_sharp

    def abs_tick_to_bbdt(self, abs_tick: int):
        bar0, r = divmod(abs_tick, self.ticks_per_bar)
        beat0, r = divmod(r, self.ppq)
        div0, tick = divmod(r, self.ticks_per_div)

        # return 1-based fields like DAWs display
        return (bar0 + 1, beat0 + 1, div0 + 1, tick)
    
    def secs_to_abs_tick(self, time_secs: float):
        total_beats = time_secs / self.secs_per_beat
        abs_tick = round(total_beats * self.ppq)
        return abs_tick

    def resolve_staff_overlaps(self, pairs: np.ndarray) -> np.ndarray:
        if pairs.size <= 1:
            # Nothing to resolve when zero or one note
            return pairs

        tol = self.onset_tolerance_ticks
        resolved = pairs.copy()
        max_staff = 1

        for staff in range(max_staff + 1):
            staff_idx = np.where(resolved["staff"] == staff)[0]
            if staff_idx.size <= 1:
                continue

            j = 0
            while j < len(staff_idx):
                # Start a potential chord group at current onset (with tolerance)
                current_idx = staff_idx[j]
                current_on = resolved["on_abs_tick"][current_idx]
                chord_start = j
                j += 1
                while j < len(staff_idx) and abs(resolved["on_abs_tick"][staff_idx[j]] - current_on) <= tol:
                    j += 1

                chord_slice = staff_idx[chord_start:j]
                if len(chord_slice) > 1:
                    # Chord: unify off time to the latest note-off
                    shared_on = resolved["on_abs_tick"][chord_slice].min()
                    shared_off = resolved["off_abs_tick"][chord_slice].max()
                    resolved["on_abs_tick"][chord_slice] = shared_on
                    resolved["off_abs_tick"][chord_slice] = shared_off
                    j-=1
                elif j < len(staff_idx):
                    # Overlap but not chord: trim current note to the next onset (with tolerance)
                    next_on = resolved["on_abs_tick"][staff_idx[j]]
                    if resolved["off_abs_tick"][current_idx] > next_on:
                        old_off = resolved["off_abs_tick"][current_idx]
                        new_off = max(current_on, next_on)
                        same_mask = (
                            (resolved["staff"] == staff)
                            & (resolved["on_abs_tick"] == current_on)
                            & (resolved["off_abs_tick"] == old_off)
                        )
                        resolved["off_abs_tick"][same_mask] = new_off

        return resolved
    
    def midi_pairs_to_bbdt(self, pairs: np.ndarray) -> np.ndarray: # DT_PAIR_ABS_TICK
        pairs_bbdt_list = []
        for staff, midi_num, abs_tick_on, abs_tick_off in pairs:
            bbdt_on = self.abs_tick_to_bbdt(abs_tick_on)
            bbdt_off = self.abs_tick_to_bbdt(abs_tick_off)
            item = (staff, midi_num, abs_tick_on, *bbdt_on, abs_tick_off, *bbdt_off, False)
            pairs_bbdt_list.append(item)
        pairs_bbdt = np.array(pairs_bbdt_list, dtype=DT_PAIR_BBDT)
        return pairs_bbdt

    def midi_pairs_to_pairs_secs(self, pairs: list[tuple[int, float, float]]) -> np.ndarray: # Each tuple: (midi_num, time_on_secs, time_off_secs))
        pairs_secs_list = []
        for midi_num, time_on_secs, time_off_secs in pairs:
            hands_range_idx = (
                self.hands_midi_num_ranges_per_time.find_index(time_on_secs) 
                if not self.always_use_latest_hands_range 
                else self.hands_midi_num_ranges_per_time.get_latest_two_hands()
            )
            # We don't determine staff now, because we need to ensure the array is sorted.
            # (hands_idx, staff, midi_num, on_secs, off_secs)
            item = (hands_range_idx, -1, midi_num, time_on_secs, time_off_secs)
            pairs_secs_list.append(item)
        pairs_secs = np.array(pairs_secs_list, dtype=DT_PAIR_SECS)
        pairs_secs = pairs_secs[np.lexsort((pairs_secs["off_secs"], pairs_secs["on_secs"]))]

        current_hands_range = None
        for pair in pairs_secs:
            hands_idx = pair["hands_idx"]
            midi_num = pair["midi_num"]
            hands_range = self.hands_midi_num_ranges_per_time.items[hands_idx] if hands_idx > -1 else None
            if hands_range:
                if (len(hands_range) == 2 and all((hand_range is not None) for hand_range in hands_range )) or current_hands_range is None:
                    current_hands_range = hands_range
            staff_idx = StaffMapping.determine_staff_idx(current_hands_range, self.default_split_midi_num, midi_num)
            pair["staff"] = staff_idx
        return pairs_secs

    def midi_pairs_secs_to_abs_tick(self, pairs_secs: np.ndarray) -> np.ndarray: # DT_PAIR_SECS
        pairs_abs_tick_list = []
        for _, staff, midi_num, on_secs, off_secs in pairs_secs:
            abs_tick_on = self.secs_to_abs_tick(on_secs)
            abs_tick_off = self.secs_to_abs_tick(off_secs)
            # (staff, midi_num, on_bar, on_abs_tick, off_bar, off_abs_tick)
            item = (staff, midi_num, -1, abs_tick_on, -1, abs_tick_off)
            pairs_abs_tick_list.append(item)
        pairs_abs_tick = np.array(pairs_abs_tick_list, dtype=DT_PAIR_ABS_TICK_EXT)
        current_length_ticks = pairs_abs_tick["off_abs_tick"].max() if len(pairs_abs_tick) else 0
        if self.crop_silence_at_start and current_length_ticks > 0:
            crop_abs_ticks = pairs_abs_tick["on_abs_tick"].min()
            self.crop_abs_ticks = crop_abs_ticks
            pairs_abs_tick["on_abs_tick"] -= crop_abs_ticks
            pairs_abs_tick["off_abs_tick"] -= crop_abs_ticks
        pairs_abs_tick = pairs_abs_tick[np.lexsort((pairs_abs_tick["off_abs_tick"], pairs_abs_tick["on_abs_tick"]))]
        for pair in pairs_abs_tick:
            on_bar, _, _, _ = self.abs_tick_to_bbdt(pair["on_abs_tick"])
            off_bar, _, _, _ = self.abs_tick_to_bbdt(pair["off_abs_tick"])
            pair["on_bar"] = on_bar
            pair["off_bar"] = off_bar

        pairs_abs_tick = pairs_abs_tick[["staff", "midi_num", "on_abs_tick", "off_abs_tick"]].astype(DT_PAIR_ABS_TICK)
        pairs_abs_tick = pairs_abs_tick[np.lexsort((pairs_abs_tick["on_abs_tick"], pairs_abs_tick["staff"]))]
        return pairs_abs_tick
    
    def fill_rests(self, pairs_bbdt: np.ndarray) -> np.ndarray:
        if pairs_bbdt.size == 0:
            if not self.min_measures or self.min_measures < 1:
                return pairs_bbdt
            

        filled = []
        # find global earliest onset and latest offset across all staves
        first_on = 0
        last_off_bar = pairs_bbdt["off_bar"].max() if len(pairs_bbdt) else 0
        last_off_abs_tick = pairs_bbdt["off_abs_tick"].max() if len(pairs_bbdt) else 0
        last_off = max(1, self.min_measures, last_off_bar - 1) * self.ticks_per_bar
        if last_off < last_off_abs_tick:
            last_off = last_off_bar * self.ticks_per_bar

        staves = np.unique(pairs_bbdt["staff"])
        if len(staves) < 2:
            for staff in range(2):
                if staff not in staves:
                    item = (staff, 0, 0, 1, 1, 1, 1, self.ticks_per_bar, 2, 1, 1, 1, False)
                    pairs_bbdt = np.concatenate((pairs_bbdt, np.array([item], dtype=DT_PAIR_BBDT)))


        for staff in np.unique(pairs_bbdt["staff"]):
            staff_pairs = pairs_bbdt[pairs_bbdt["staff"] == staff]
            if staff_pairs.size == 0:
                continue

            # leading rest if staff starts after global start
            staff_first_on = staff_pairs["on_abs_tick"][0]
            if staff_first_on > first_on:
                on_bbdt = self.abs_tick_to_bbdt(first_on)
                off_bbdt = self.abs_tick_to_bbdt(staff_first_on)
                rest = (staff, 0, first_on, *on_bbdt, staff_first_on, *off_bbdt, False)
                filled.append(rest)

            for idx, note in enumerate(staff_pairs):
                filled.append(tuple(note))
                if idx + 1 >= len(staff_pairs):
                    continue

                next_note = staff_pairs[idx + 1]
                gap_on = note["off_abs_tick"]
                gap_off = next_note["on_abs_tick"]
                if gap_off > gap_on:
                    on_bbdt = self.abs_tick_to_bbdt(gap_on)
                    off_bbdt = self.abs_tick_to_bbdt(gap_off)
                    rest = (staff, 0, gap_on, *on_bbdt, gap_off, *off_bbdt, False)
                    filled.append(rest)

            # trailing rest if staff ends before global end
            staff_last_off = staff_pairs["off_abs_tick"][-1]
            if staff_last_off < last_off:
                on_bbdt = self.abs_tick_to_bbdt(staff_last_off)
                off_bbdt = self.abs_tick_to_bbdt(last_off)
                rest = (staff, 0, staff_last_off, *on_bbdt, last_off, *off_bbdt, False)
                filled.append(rest)
        return np.array(filled, dtype=DT_PAIR_BBDT)

    def split_spanning_segments(self, pairs_bbdt: np.ndarray) -> np.ndarray:
        if pairs_bbdt.size == 0:
            return pairs_bbdt

        separated = []
        for staff in np.unique(pairs_bbdt["staff"]):
            staff_pairs = pairs_bbdt[pairs_bbdt["staff"] == staff]
            for note in staff_pairs:
                note_off = note["off_abs_tick"]
                cur_on = note["on_abs_tick"]
                while True:
                    bar_start = (cur_on // self.ticks_per_bar) * self.ticks_per_bar
                    bar_end = bar_start + self.ticks_per_bar
                    seg_off = min(note_off, bar_end)
                    has_more = seg_off < note_off
                    on_bbdt = self.abs_tick_to_bbdt(cur_on)
                    off_bbdt = self.abs_tick_to_bbdt(seg_off)
                    tie_val = 1 if (has_more and note["midi_num"] > 0) else 0
                    separated.append(
                        (note["staff"], note["midi_num"], cur_on, *on_bbdt, seg_off, *off_bbdt, tie_val)
                    )
                    if not has_more:
                        break
                    cur_on = seg_off

        return np.array(separated, dtype=DT_PAIR_BBDT)

    def midi_to_full_bbdt(
        self,
        pairs: list[tuple[int, float, float]], # Each tuple: (midi_num, time_on_secs, time_off_secs))
    ):
        pairs_secs = self.midi_pairs_to_pairs_secs(pairs)

        time_sig_changed = False
        if self.estimator and len(pairs_secs):
            try:
                estimation_result = self.estimator.estimate(pairs_secs)
                self.last_estimation_result = estimation_result
                if estimation_result.bpm is not None:
                    self.bpm = float(estimation_result.bpm)
                if estimation_result.time_signature is not None:
                    if tuple(estimation_result.time_signature) != tuple(self.time_sig):
                        time_sig_changed = True
                    self.time_sig = estimation_result.time_signature
                self.recalculate_timing_members(onset_tolerance_beats=self.onset_tolerance_beats)
            except Exception:
                pass


        pairs_abs_tick = self.midi_pairs_secs_to_abs_tick(pairs_secs)
        pairs_abs_tick = self.resolve_staff_overlaps(pairs_abs_tick)
        pairs_bbdt = self.midi_pairs_to_bbdt(pairs_abs_tick)
        pairs_bbdt = self.fill_rests(pairs_bbdt)
        pairs_bbdt = self.split_spanning_segments(pairs_bbdt)
        if time_sig_changed:
            self.clefs_per_bar = self.determine_clefs_per_bar(pairs_bbdt)
        return pairs_bbdt

    def determine_clefs_per_bar(self, pairs_bbdt: np.ndarray) -> dict:
        max_staff = 1
        clefs_per_bar = {}
        unique_bars = np.unique(pairs_bbdt["on_bar"])
        for bar in unique_bars:
            if bar in self.clefs_per_bar:
                clefs_per_bar[int(bar)] = self.clefs_per_bar[int(bar)].copy()
                continue
            bar_pairs = pairs_bbdt[pairs_bbdt["on_bar"] == bar]
            clefs_per_bar[int(bar)] = ["G"] * (max_staff + 1)
            for staff in np.unique(bar_pairs["staff"]):
                staff_pairs = bar_pairs[(bar_pairs["staff"] == staff) & (bar_pairs["midi_num"] > 0)]
                include_mask = np.array([(self.calculate_duration(note)[0]["base"] != NoteDuration.NONE) for note in staff_pairs], dtype=bool)
                midi_nums = staff_pairs["midi_num"][include_mask]
                if bar > 1:
                    if (bar - 1) in clefs_per_bar:
                        prev_clef_key = clefs_per_bar[int(bar - 1)][int(staff)]
                    elif (bar - 1) in self.clefs_per_bar:
                        prev_clef_key = self.clefs_per_bar[int(bar - 1)][int(staff)]
                    else:
                        prev_clef_key = None
                else:
                    prev_clef_key = None
                clef_key = prev_clef_key if midi_nums.size == 0 else StaffMapping.determine_clef_key(midi_nums, prev_clef_key)
                clefs_per_bar[int(bar)][int(staff)] = clef_key if clef_key is not None else "G"
        if not clefs_per_bar:
            clefs_per_bar[1] = ["G", "F"]
        for bar, clefs in clefs_per_bar.items():
            if int(bar) in self.clefs_per_bar:
                continue
            has_future_note = np.any((pairs_bbdt["midi_num"] > 0) & (pairs_bbdt["on_bar"] > bar))
            if has_future_note:
                self.clefs_per_bar[int(bar)] = clefs.copy()
        return clefs_per_bar

    def filter_notation_measures(
        self,
        end_bar: int,
        for_file: bool,
    ):
        pairs_bbdt = self.pairs_bbdt
        if not for_file:
            if end_bar:
                pairs_bbdt = pairs_bbdt[pairs_bbdt["on_bar"] <= end_bar]
            if self.take_latest_measures:
                max_bar = pairs_bbdt["off_bar"].max()
                if max_bar > self.take_latest_measures + 1 and max_bar > self.min_measures + 1:
                    start_bar = max_bar - self.take_latest_measures - 1
                    pairs_bbdt = pairs_bbdt[pairs_bbdt["on_bar"] > start_bar]
        max_bar = pairs_bbdt["on_bar"].max()
        if self.min_measures and self.min_measures > 0 and max_bar < self.min_measures:
            pairs_bbdt = self.fill_rests(pairs_bbdt)
            pairs_bbdt = self.split_spanning_segments(pairs_bbdt)
        clefs_per_bar = self.determine_clefs_per_bar(pairs_bbdt)
        if end_bar:
            for bar in range(end_bar + 1, np.max(list(clefs_per_bar.keys())) + 1):
                clefs_per_bar[bar] = clefs_per_bar[max_bar]
        return pairs_bbdt, clefs_per_bar

    def build_mei_item(self, item_dict: dict, measure_num: int, staff_num: int, layer_id: str, id_counter: itertools.count, tied_note_ids: dict[tuple[int, int], str]):
        # See: https://music-encoding.org/guidelines/v5/content/cmn.html#cmnSlurTies
        tie_pairs = []
        if len(item_dict["pitches"]) > 0:
            result = []
            for midi_num in item_dict["pitches"]:
                note_el_id = f"{layer_id}n{next(id_counter)}"
                tie_key_tuple = (staff_num, midi_num)
                if tie_key_tuple in tied_note_ids:
                    tie_pairs.append((tied_note_ids[tie_key_tuple], {"measure_num": measure_num, "staff_num": staff_num, "note_id": note_el_id}))
                    if item_dict.get("tie", False):
                        # tie medial
                        tied_note_ids[tie_key_tuple] = {"measure_num": measure_num, "staff_num": staff_num, "note_id": note_el_id}
                    else:
                        # tie terminal
                        del tied_note_ids[tie_key_tuple]
                elif item_dict.get("tie", False):
                    # tie initial
                    tied_note_ids[tie_key_tuple] = {"measure_num": measure_num, "staff_num": staff_num, "note_id": note_el_id}
                name, note_num, octave, is_sharp = self.pitch_from_midi_num(midi_num)
                name = name[:1].lower()
                accid = "s" if is_sharp else None
                if len(item_dict["pitches"]) == 1:
                    attrs = {
                        "xml:id": note_el_id,
                        "pname": name,
                        "oct": str(octave),
                        "dur": str(item_dict["duration"]["base"]),
                    }
                    if accid:
                        attrs["accid"] = accid
                    if item_dict["duration"]["dots"]:
                        attrs["dots"] = str(item_dict["duration"]["dots"])
                    result.append(ET.Element("note", attrib=attrs))
                else:
                    attrs = {
                        "xml:id": note_el_id,
                        "pname": name,
                        "oct": str(octave),
                    }
                    if accid:
                        attrs["accid"] = accid
                    result.append(ET.Element("note", attrib=attrs))
            if len(result) > 1:
                chord_attrs = {"dur": str(item_dict["duration"]["base"])}
                if item_dict["duration"]["dots"]:
                    chord_attrs["dots"] = str(item_dict["duration"]["dots"])
                chord_el = ET.Element("chord", attrib=chord_attrs)
                for note_el in result:
                    chord_el.append(note_el)
                return chord_el, tie_pairs
            else:
                return result[0], tie_pairs
        else:
            rest_attrs = {
                "xml:id": f"{layer_id}n{next(id_counter)}",
                "dur": str(item_dict["duration"]["base"]),
            }
            if item_dict["duration"]["dots"]:
                rest_attrs["dots"] = str(item_dict["duration"]["dots"])
            return ET.Element("rest", attrib=rest_attrs), tie_pairs

    def calculate_duration(self, item):
        dur_divs = round((item["off_abs_tick"] - item["on_abs_tick"]) / self.ticks_per_div)
        if dur_divs < 1:
            return [{
                "base": NoteDuration.NONE,
                "dots": 0,
            }]

        if dur_divs in self.DURATION_NAME:
            base, dots = self.DURATION_NAME[dur_divs]
            return [{
                "base": base,
                "dots": dots,
            }]

        remaining = dur_divs
        pieces = []
        denom_values = sorted(self.DURATION_NAME.keys(), reverse=True)
        while remaining > 0:
            for denom in denom_values:
                if denom <= remaining:
                    pieces.append(denom)
                    remaining -= denom
                    break

        pieces.sort()
        return [
            {"base": self.DURATION_NAME[denom][0], "dots": self.DURATION_NAME[denom][1]}
            for denom in pieces
        ]
    
    def is_beamable(self, item_dict: dict) -> bool:
        dur = item_dict["duration"]
        return (
            dur.get("base") in (NoteDuration.EIGHTH, NoteDuration.SIXTEENTH)
            and len(item_dict["pitches"]) > 0
        )

    def align_measure(self, pairs_bbdt_measure: np.ndarray):
        k_on  = pairs_bbdt_measure["on_abs_tick"]
        k_off = pairs_bbdt_measure["off_abs_tick"]
        cuts = np.flatnonzero((k_on[1:] != k_on[:-1]) | (k_off[1:] != k_off[:-1])) + 1
        groups = np.split(pairs_bbdt_measure, cuts)
        result = []
        for group in groups:
            pitches = np.unique(group["midi_num"]).tolist()
            if 0 in pitches:
                pitches = []
            if len(pitches) > 1:
                pitches.sort()
            first_item = group[0]
            duration = self.calculate_duration(first_item)
            for i, duration_piece in enumerate(duration):
                item_dict = {
                    "pitches": pitches,
                    "duration": duration_piece,
                }
                if len(duration) > 0 and i < len(duration) - 1 and duration[i + 1]["base"] != NoteDuration.NONE:
                    item_dict["tie"] = True
                result.append(item_dict)
            if first_item["tie"] and result[-1]["duration"]["base"] != NoteDuration.NONE:
                result[-1]["tie"] = True
        return result
    
    def apply_beaming_measure(self, measure_dicts: list, measure_start_tick: int):
        # BEAMING respecting 4/4 accent rules: group within beats 1-2 or 3-4, avoid 2-3
        result = []
        beam_buf = []
        beam_window = None
        beat_ticks = self.ticks_per_bar // self.beats_per_bar

        current_tick = measure_start_tick
        for item_dict in measure_dicts:
            item_duration = item_dict["duration"]
            dur_ticks = __class__.DURATION_NAME_INV.get((item_duration["base"], item_duration["dots"]), 0) * self.ticks_per_div
            item_start = current_tick
            item_end = current_tick + dur_ticks
            current_tick = item_end

            if self.is_beamable(item_dict):
                window = ((item_start - measure_start_tick) // beat_ticks) // 2  # 0 for beats 1-2, 1 for beats 3-4
                if beam_window is None:
                    beam_window = window
                if window != beam_window:
                    if len(beam_buf) == 1:
                        result.append(beam_buf[0])
                    elif beam_buf:
                        result.append(beam_buf)
                    beam_buf = []
                    beam_window = window
                beam_buf.append(item_dict)
            else:
                if len(beam_buf) == 1:
                    result.append(beam_buf[0])
                    beam_buf = []
                    beam_window = None
                elif beam_buf:
                    result.append(beam_buf)
                    beam_buf = []
                    beam_window = None
                result.append(item_dict)

        if len(beam_buf) == 1:
            result.append(beam_buf[0])
        elif beam_buf:
            result.append(beam_buf)
        return result

    def build_mei_measure(self, measure_num: int, staff_num: int, parent_el: ET.Element, measure_dicts: list, layer_id: str, tied_note_ids: dict[int, str]):
        id_counter = itertools.count(1)
        measure_tie_pairs = []
        for item_dict in measure_dicts:
            if isinstance(item_dict, list):
                beam_el = ET.Element("beam")
                for beamed_item_dict in item_dict:
                    note_el, tie_pairs = self.build_mei_item(beamed_item_dict, measure_num, staff_num, layer_id, id_counter, tied_note_ids)
                    beam_el.append(note_el)
                    measure_tie_pairs += tie_pairs
                parent_el.append(beam_el)
            else:
                note_el, tie_pairs = self.build_mei_item(item_dict, measure_num, staff_num, layer_id, id_counter, tied_note_ids)
                parent_el.append(note_el)
                measure_tie_pairs += tie_pairs
        return measure_tie_pairs

    def pairs_bbdt_to_dicts(self, pairs_bbdt: np.ndarray):
        staff_indices = np.unique(pairs_bbdt["staff"])
        staff_indices.sort()

        min_bar = pairs_bbdt["on_bar"].min()
        max_bar = pairs_bbdt["off_bar"].max()
        measures = []

        for measure_num in range(min_bar, max_bar):
            measure = []
            for staff_idx in staff_indices:
                pairs_bbdt_measure = pairs_bbdt[(pairs_bbdt["staff"] == staff_idx) & (pairs_bbdt["on_bar"] == measure_num)]
                measure_dicts = self.align_measure(pairs_bbdt_measure)
                measure.append(measure_dicts)
                measure_start_tick = pairs_bbdt_measure["on_abs_tick"].min() if pairs_bbdt_measure.size else 0
            measures.append((measure, measure_start_tick))
        
        # CLEANUP
        # Untie notes tied with another note with NoteDuration.NONE
        for (current_measure, _), (next_measure, _) in zip(measures[0:-1], measures[1:]):
            for current_staff, next_staff in zip(current_measure, next_measure):
                if not len(current_staff):
                    continue
                current_staf_last_dict = current_staff[-1]
                if not current_staf_last_dict.get("tie", False):
                    continue
                if len(next_staff):
                    next_staff_first_dict = next_staff[0]
                    if next_staff_first_dict["duration"]["base"] == NoteDuration.NONE:
                        current_staf_last_dict["tie"] = False
                else:
                    current_staf_last_dict["tie"] = False
             
        # Remove notes with NoteDuration.NONE
        measures = [([[item_dict for item_dict in measure_staff if item_dict["duration"]["base"] != NoteDuration.NONE] for measure_staff in measure], measure_start_tick) for measure, measure_start_tick in measures]

        # APPLY BEAMING
        for measure_idx, (measure, measure_start_tick) in enumerate(measures):
            for staff_idx, measure_dicts in enumerate(measure):
                measure[staff_idx] = self.apply_beaming_measure(measure_dicts, measure_start_tick)
            measures[measure_idx] = measure

        return measures

    # STATE MANAGEMENT

    def set_state(
        self,
        pairs: list[tuple[int, float, float]], # Each tuple: (midi_num, time_on_secs, time_off_secs)
        hands_midi_num_ranges_per_time: HandsMidiNumRangesPerTime,
    ):
        self.hands_midi_num_ranges_per_time = hands_midi_num_ranges_per_time
        self.pairs_bbdt = self.midi_to_full_bbdt(pairs)

    # MEI GENERATION

    def clef_key_to_def(self, clef_key: str):
        if clef_key == "G":
            return ("G", "2")
        elif clef_key == "F":
            return ("F", "4")
        raise Exception(f'Invalid clef key: "{clef_key}"')

    def pairs_bbdt_to_mei(self, pairs_bbdt: np.ndarray, clefs_per_bar: dict, for_file: bool = False):
        """
<mei>
    <meiHead />
    <music>
        <body>
            <mdiv>
                <score>
                    <scoreDef>
                        <staffGrp>
                            <staffDef n="1" lines="5" meter.count="4" meter.unit="4">
                                <clef shape="G" line="2" />
                            </staffDef>
                            <staffDef n="2" lines="5" meter.count="4" meter.unit="4">
                                <clef shape="G" line="2" />
                            </staffDef>
                        </staffGrp>
                    </scoreDef>
                    <section id="s1">
                        <measure id="m1" n="1">
                            <staff id="m1s1" n="1">
                                <layer id="m1s1l1" n="1">
                                    <rest dur="1" />
                                </layer>
                            </staff>
                            <staff id="m1s2" n="2">
                                <layer id="m1s2l1" n="1">
                                    <note id="m1s2l1n1" dur="8" pname="c" oct="4" />
                                    <note id="m1s2l1n2" dur="8" pname="d" oct="4" />
                                    <note id="m1s2l1n3" dur="8" pname="d" oct="4">
                                </layer>
                            </staff>
                            <tie id="m1s2l1t1" startid="#m1s2l1n2" endid="#m1s2l1n3" />
                        </measure>
                    </section>
                </score>
            </mdiv>
        </body>
    </music>                
</mei>
        """

        mei_root = ET.Element("mei", attrib={
            "xmlns": "http://www.music-encoding.org/ns/mei",
            "meiversion": '5.1+basic',
        })
        ET.SubElement(mei_root, "meiHead")
        music_el = ET.SubElement(mei_root, "music")
        body_el = ET.SubElement(music_el, "body")
        mdiv_el = ET.SubElement(body_el, "mdiv")
        score_el = ET.SubElement(mdiv_el, "score")
        if self.print_timing:
            score_def_el = ET.SubElement(score_el, "scoreDef", attrib={
                "meter.count": str(self.time_sig[0]),
                "meter.unit": str(self.time_sig[1]),
                "midi.bpm": f"{self.bpm:.2f}",
            })
        else:
            score_def_el = ET.SubElement(score_el, "scoreDef", attrib={
                "midi.bpm": f"{self.bpm:.2f}",
            })
        staff_grp_el = ET.SubElement(score_def_el, "staffGrp", attrib={
            "symbol": "none",
            "bar.thru": "false",
        })
        
        min_bar = pairs_bbdt["on_bar"].min()

        if len(clefs_per_bar):
            last_clef_keys = clefs_per_bar[min_bar if min_bar in clefs_per_bar else 1]
        else:
            if self.default_split_midi_num <= 60:
                last_clef_keys = ["G", "G"]
            else:
                last_clef_keys = ["G", "F"]
        first_clefs = [self.clef_key_to_def(clef_key) for clef_key in last_clef_keys]
        staff_defs = [
            {"n": "1", "lines": "5", "clef.shape": first_clefs[0][0], "clef.line": first_clefs[0][1]},
            {"n": "2", "lines": "5", "clef.shape": first_clefs[1][0], "clef.line": first_clefs[1][1]},
        ]
        pass
        for attrs in staff_defs:
            ET.SubElement(staff_grp_el, "staffDef", attrib=attrs)

        section_el = ET.SubElement(score_el, "section", attrib={"xml:id": "s1"})

        measures_dicts = self.pairs_bbdt_to_dicts(pairs_bbdt)

        tied_note_ids: dict[tuple[int, int], str] = {}
        measure_elements: list[ET.Element] = []
        measure_tie_pairs = []
        for measure_num, measure_staves in enumerate(measures_dicts, start=min_bar):
            measure_id = f"m{measure_num}"
            measure_el = ET.SubElement(section_el, "measure", attrib={"xml:id": measure_id, "n": str(measure_num)})
            measure_elements.append(measure_el)
            if self.print_timing and measure_num == min_bar:
                """
                <tempo xml:id="t1ynpvpy" type="mscore-infer-from-text" midi.bpm="80" staff="1" tstamp="1">
                    <rend glyph.auth="smufl"></rend> = 80</tempo>
                """
                tempo_el = ET.SubElement(
                    measure_el,
                    "tempo",
                    attrib={
                        "xml:id": f"{measure_id}tempo",
                        #"place": "above",
                        "staff": "1",
                        "tstamp": "1",
                    },
                )
                tempo_el.text = f"♩ = {int(self.bpm)}"
                
            measure_clefs = clefs_per_bar.get(measure_num, None)
            for staff_num, measure_staff in enumerate(measure_staves, start=1):
                staff_el = ET.SubElement(measure_el, "staff", attrib={"xml:id": f"m{measure_num}s{staff_num}", "n": str(staff_num)})
                layer_id = f"m{measure_num}s{staff_num}l1"
                layer_el = ET.SubElement(staff_el, "layer", attrib={"xml:id": layer_id, "n": "1"})
                last_clef_key = last_clef_keys[staff_num - 1]
                clef_key = measure_clefs[staff_num - 1] if measure_clefs else last_clef_key
                if clef_key != last_clef_key:
                    if any(tied_note for tied_note in tied_note_ids.values() if tied_note.get("staff_num", None) == staff_num):
                        clef_key = last_clef_key
                    else:
                        clef_data = self.clef_key_to_def(clef_key)
                        clef_el = ET.SubElement(layer_el, "clef", attrib={"shape": clef_data[0], "line": clef_data[1]})
                        layer_el.append(clef_el)
                        last_clef_keys[staff_num - 1] = clef_key
                measure_tie_pairs += self.build_mei_measure(measure_num, staff_num, layer_el, measure_staff, layer_id, tied_note_ids)
        for tie_num, (tie_start_dict, tie_end_dict) in enumerate(measure_tie_pairs, start=1):
            attrs = {
                "xml:id": f"{measure_id}t{tie_num}",
                "startid": f"#{tie_start_dict["note_id"]}",
                "endid": f"#{tie_end_dict["note_id"]}",
            }
            tie_el = ET.Element("tie", attrib=attrs)
            measure_num_to_append = tie_start_dict["measure_num"]
            measure_el = measure_elements[measure_num_to_append - min_bar]
            measure_el.append(tie_el)
        return mei_root

    def midi_to_mei(
        self,
        end_bar: int,
        for_file: bool = False
    ):
        pairs_bbdt, clefs_per_bar = self.filter_notation_measures(end_bar, for_file)
        return self.pairs_bbdt_to_mei(pairs_bbdt, clefs_per_bar, for_file)

    def calculate_end_bar(self, pts_time: float):
        return int(np.floor(pts_time / self.secs_per_bar))

    # IMAGE CREATION

    async def midi_to_image(
        self,
        end_bar: int,
        output_height: int = None,
        background_color = None,
        foreground_color: str=None,
        alpha_rate: float = 1,
        fixed_size: bool=False,
        print_measure_nums_interval: int=None,
        margin_vertical_extra_units=None,
        margin_horizontal_extra_units=None,
        output_width: int=None,
        return_svg: bool = False,
    ):
        mei = self.midi_to_mei(end_bar)
        svg = await NotationRenderer.verovio_svg_from_mei(
            mei,
            fixed_size=fixed_size,
            print_measure_nums_interval=print_measure_nums_interval,
            margin_vertical_extra_units=margin_vertical_extra_units,
            margin_horizontal_extra_units=margin_horizontal_extra_units,
            output_width=output_width,
            output_height=output_height
        )
        bgra = await NotationRenderer.svg_string_to_rgba(
            svg, 
            background_color=background_color, 
            foreground_color=foreground_color, 
            alpha_rate=alpha_rate,
            include_image_alpha_channel=self.include_image_alpha_channel,
        )
        bgra[0:1, :, :] = 255
        bgra[-1:, :, :] = 255
        if return_svg:
            return bgra, svg
        else:
            return bgra

    def create_midi_to_image_fn(self, **kwargs):
        return lambda end_bar: self.midi_to_image(end_bar, **kwargs)
