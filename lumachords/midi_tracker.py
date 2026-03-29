import asyncio
from dataclasses import dataclass
from collections import deque
import threading
import numpy as np
import itertools
import mido

from lumachords.hands_detector import HandMidiNumRange, HandsMidiNumRangesPerTime
from lumachords.notation_placer import NotationPlacer
from lumachords.data_types import NoteEvent
from lumachords.utils import Utils
from lumachords.video import VideoWriter


@dataclass
class MidiTrackerSnapshot:
    pts: int
    time: float
    active_midi_num_groups: list[list[int]]
    all_note_events_count: int
    time_on: dict[int,float]

class MidiTracker:
    def __init__(self, actual_fps: int, default_split_midi_num=60, video_backend: str | None = None, midi_velocity: int=50):
        self.actual_fps = actual_fps
        self.default_split_midi_num = default_split_midi_num
        self.video_backend = video_backend
        self.midi_velocity = midi_velocity
        self.time_on: dict[int,float] = {}
        self.time_on_lock = threading.Lock()
        self.all_note_events: list[NoteEvent] = []
        self.event_pairs: list[tuple[int,float,float]] = []
        self.hands_midi_num_ranges_per_time = HandsMidiNumRangesPerTime()
        self.active_midi_num_groups = None
        self.last_pts = None
        self.last_pts_time = None
        self.snapshots_lock = threading.Lock()
        self.snapshots: deque[MidiTrackerSnapshot] = deque(maxlen=actual_fps * 5) # 5 seconds worth of snapshots

    def get_current_active_midi_nums(self):
        with self.time_on_lock:
            result = [midi_num for midi_num in self.time_on.keys()]
        result.sort()
        return result

    def set_current_note_state(self, pts_time: float, raw_events: np.ndarray):
        raw_events_parts = [
            raw_events[raw_events["time_delta"] <= 0],
            raw_events[raw_events["time_delta"] > 0]
        ]
        active_midi_num_groups = [self.get_current_active_midi_nums()]
        for raw_events_part in raw_events_parts:
            for ev in raw_events_part:
                midi_num = int(ev["midi_num"])
                is_on = bool(ev["is_on"])
                if is_on:
                    with self.time_on_lock:
                        if midi_num in self.time_on:
                            # An ON event came for an already ON note
                            raw_events = raw_events[raw_events != ev]
                        else:
                            self.time_on[midi_num] = float(pts_time + ev["time_delta"])
                elif midi_num in self.time_on:
                    with self.time_on_lock:
                        on_time = self.time_on[midi_num]
                        off_time = float(pts_time + ev["time_delta"])
                        del self.time_on[midi_num]
                    self.add_event_pair(midi_num, on_time, off_time)
                else:
                    # An OFF event came for an already OFF note
                    raw_events = raw_events[raw_events != ev]
            active_midi_num_groups.append(self.get_current_active_midi_nums())
        active_midi_num_groups.append(self.get_current_active_midi_nums())
        active_midi_num_groups = [k for k, _ in itertools.groupby(active_midi_num_groups)]
        return active_midi_num_groups, raw_events

    def step_frame(self, pts: int, raw_events: np.ndarray, hands_midi_num_ranges: list[HandMidiNumRange], play_y_lag_time_delta: float):
        pts_time = Utils.pts_to_pts_time(pts, self.actual_fps)
        self.active_midi_num_groups, raw_events = self.set_current_note_state(pts_time, raw_events)
        self.add_hands_ranges(pts_time - play_y_lag_time_delta, hands_midi_num_ranges)
        note_events = [NoteEvent(float(pts_time + ev["time_delta"] + play_y_lag_time_delta), bool(ev["is_on"]), int(ev["midi_num"]), int(ev["id"])) for ev in raw_events]
        self.all_note_events += note_events
        with self.snapshots_lock:
            self.snapshots.append(MidiTrackerSnapshot(pts, pts_time, self.active_midi_num_groups, len(self.all_note_events), dict(self.time_on)))
        self.last_pts = pts
        self.last_pts_time = pts_time
        return note_events
    
    def report_last_items(self, pts, pts_time, happening_pts: int = None, last_k=5):
        happening_note_events = self.get_happening_note_events(happening_pts=happening_pts)
        if not len(happening_note_events):
            return  "No notes have been played yet"
        active_midi_str = "Active Notes: " + ", ".join(map(str, [[f"{Utils.midi_num_to_name(midi_num)} ({midi_num})" for midi_num in midi_num_group] for midi_num_group in self.get_happening_active_midi_num_groups(happening_pts=happening_pts)]))
        last_events = happening_note_events[-last_k:]
        after_idx = np.argmax([ev.time > pts_time for ev in last_events])
        last_events_str_list = list(map(lambda ev: \
            f"TIME: {ev.time:10.3f} {"ON " if ev.is_on else "OFF"} {Utils.midi_num_to_name(ev.midi_num)} ({ev.midi_num:d}) BOX ID: {ev.box_id}", last_events))
        last_before_events_str_list = last_events_str_list[:after_idx]
        last_after_events_str_list = last_events_str_list[after_idx:]
        all_str_list = [active_midi_str, f"PTS: {pts:5d} TIME: {pts_time:10.3f}"] + last_before_events_str_list + ["------------------------"] + last_after_events_str_list
        return "\n".join(all_str_list)

    def add_event_pair(self, midi_num: int, on_time: float, off_time: float):
        self.event_pairs.append((midi_num, on_time, off_time))

    def get_event_pairs(self, include_active: bool, happening_time: float = 0.0) -> list[tuple[int, float, float]]:
        result = self.event_pairs.copy()
        cut_end_rev_i = -1
        snapshot = None
        current_time_on: dict[int, float] = {}

        if happening_time and happening_time > 0.0:
            happening_pts = Utils.pts_time_to_pts(happening_time, self.actual_fps)
            snapshot = self.get_snapshot(happening_pts)
            if snapshot:
                current_time_on = dict(snapshot.time_on)
            elif include_active:
                with self.time_on_lock:
                    current_time_on = dict(self.time_on)

        if len(result) and happening_time and happening_time > 0.0:
            for rev_i, (midi_num, on_time, off_time) in enumerate(reversed(result)):
                if on_time >= happening_time:
                    cut_end_rev_i = rev_i
                elif on_time < happening_time < off_time:
                    new_off_time = max(happening_time, on_time + 0.125)
                    idx = len(result) - 1 - rev_i
                    result[idx] = (midi_num, on_time, new_off_time)
                    # Only persist the trim when the note is not currently active.
                    if midi_num not in current_time_on:
                        self.event_pairs[idx] = result[idx]
                else:
                    break
        if cut_end_rev_i != -1:
            result = result[:len(result) - 1 - cut_end_rev_i]

        if include_active:
            if happening_time and happening_time > 0.0:
                last_pts_time = happening_time
            else:
                with self.time_on_lock:
                    current_time_on = dict(self.time_on)
                last_pts_time = self.last_pts_time

            active_events = [(midi_num, on_time, max(last_pts_time, on_time + 0.125)) for midi_num, on_time in current_time_on.items()]
            for active_event in active_events:
                active_midi_num, active_on_time, active_off_time = active_event
                existing_idx = next((i for i, (midi_num, on_time, _) in enumerate(result) if midi_num == active_midi_num and on_time == active_on_time), None)
                if existing_idx is not None:
                    result[existing_idx] = (active_midi_num, active_on_time, active_off_time)
                else:
                    result.append(active_event)

        result.sort(key=lambda ev: (ev[1], ev[0]))
        return result
    
    def add_hands_ranges(self, pts_time: float, hands_midi_num_ranges: list[HandMidiNumRange]):
        self.hands_midi_num_ranges_per_time.append(pts_time, hands_midi_num_ranges, check_if_last_same=True)

    def get_snapshot(self, pts: int) -> MidiTrackerSnapshot | None:
        with self.snapshots_lock:
            if not len(self.snapshots):
                return None
            first_snapshot_pts = self.snapshots[0].pts
            idx = pts - first_snapshot_pts
            if 0 <= idx < len(self.snapshots):
                for snapshot in itertools.islice(self.snapshots, idx, None):
                    if snapshot.pts == pts:
                        return snapshot
            # Fallback linear search if not found by index or idx is out of range
            for snapshot in self.snapshots:
                if snapshot.pts == pts:
                    return snapshot            
        return None
    
    def get_happening_active_midi_num_groups(self, happening_pts: int = None):
        if happening_pts and happening_pts > 0:
            snapshot = self.get_snapshot(happening_pts)
            return snapshot.active_midi_num_groups if snapshot else [[]]
        else:
            return self.active_midi_num_groups or [[]]

    def get_happening_note_events(self, happening_pts: int = None):
        if happening_pts and happening_pts > 0:
            snapshot = self.get_snapshot(happening_pts)
            return self.all_note_events[:snapshot.all_note_events_count] if snapshot else []
        else:
            return self.all_note_events

    def get_active_groups_as_measure_events(self, hands_midi_num_ranges: list[HandMidiNumRange], happening_pts: int = None) -> list[tuple[int, float, float]]:
        staff_indices = (0, 1)
        active_midi_num_groups = self.get_happening_active_midi_num_groups(happening_pts=happening_pts)
        total_duration_secs = 2.0
        part_duration_secs = total_duration_secs / len(active_midi_num_groups)
        result, last_index_for_note, previous_group = [], {}, []
        for i, current_group in enumerate(active_midi_num_groups):
            on_time, off_time = part_duration_secs * i, part_duration_secs * (i + 1)
            
            counts = []
            for group in (current_group, previous_group):
                idxs = [Utils.midi_num_to_staff_idx(midi_num, self.default_split_midi_num) for midi_num in group]
                counts.append({staff_idx: sum(i == staff_idx for i in idxs) for staff_idx in staff_indices})

            staff_counts, prev_staff_counts = counts
            ordered = [midi_num for midi_num in current_group if midi_num in previous_group] + [midi_num for midi_num in current_group if midi_num not in previous_group]
            for midi_num in ordered:
                staff = Utils.midi_num_to_staff_idx(midi_num, self.default_split_midi_num)
                can_merge = midi_num in previous_group and staff_counts[staff] == prev_staff_counts[staff] == 1
                last_index = last_index_for_note.get(midi_num)
                if can_merge and last_index is not None and result[last_index][2] == on_time:
                    result[last_index] = (midi_num, result[last_index][1], off_time)
                else:
                    result.append((midi_num, on_time, off_time))
                    last_index_for_note[midi_num] = len(result) - 1
            previous_group = current_group
        return result
    
    def save_midi(self, out_path: str):
        pairs = self.get_event_pairs(True)
        if not pairs:
            return
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        tempo = mido.bpm2tempo(120)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        track.append(mido.Message("program_change", program=0, channel=0, time=0))
        events = []
        for midi_num, on_time, off_time in pairs:
            if off_time <= on_time:
                continue
            events.append((on_time, True, int(midi_num)))
            events.append((off_time, False, int(midi_num)))
        events.sort(key=lambda e: (e[0], 0 if not e[1] else 1))
        last_time = events[0][0] if len(events) else 0.0
        for t, is_on, midi_num in events:
            delta = mido.second2tick(t - last_time, mid.ticks_per_beat, tempo)
            msg_type = "note_on" if is_on else "note_off"
            velocity = self.midi_velocity if is_on else 0
            track.append(mido.Message(msg_type, note=midi_num, velocity=velocity, time=int(delta)))
            last_time = t
        mid.save(out_path)

    async def save_video(
            self,
            progress_overlay: any,
            input_video_path: str,
            output_video_path: str,
            frame_size: tuple[int, int],
            until_pts: int,
            play_y_lag_time_delta: float,
            app_level_stop_event: asyncio.Event,
            user_level_stop_event: asyncio.Event,
            note_placer: NotationPlacer,
            reader_fps: int,
            writer_fps: int = 0,
            pix_fmt: str = "bgra"
        ) -> bool:
        local_stop_event = asyncio.Event()

        def check_any_stop_event():
            result = (app_level_stop_event and app_level_stop_event.is_set()) or (user_level_stop_event and user_level_stop_event.is_set())
            if result:
                local_stop_event.set()
            return result
        
        midi_to_image_fn =  note_placer.create_midi_to_image_fn(background_color="white",
                                alpha_rate=0.5,
                                fixed_size=True,
                                print_measure_nums_interval=1,
                                margin_horizontal_extra_units=2,
                                output_width=frame_size[1],
                                output_height=frame_size[0]*1//3
        )
        video_writer = VideoWriter(reader_fps, writer_fps=writer_fps, pix_fmt=pix_fmt, backend=self.video_backend)
        if progress_overlay and not video_writer.audio_supported:
            progress_overlay.set_progress(0, message="Saving video (no audio)...")
        overlay_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(10)
        if check_any_stop_event():
            return False
        writer_task = asyncio.get_running_loop().create_task(video_writer.write_frames(progress_overlay, input_video_path, output_video_path, overlay_queue, local_stop_event, until_pts, use_tqdm=True))
        await asyncio.sleep(0)
        end_bar = None
        im_overlay = None
        end_pts = until_pts + 1
        first_frame_processed = False
        if writer_task.done():
            if writer_task.exception() is not None:
                raise writer_task.exception()
            else:
                return False
        try:
            for pts in range(end_pts + 1):
                if writer_task.done():
                    if writer_task.exception() is not None:
                        raise writer_task.exception()
                    else:
                        break
                if check_any_stop_event():
                    break
                pts_time = Utils.pts_to_pts_time(pts, reader_fps)
                _, happening_pts_time = Utils.calculate_actual_happening_time(pts_time, True, play_y_lag_time_delta, reader_fps)
                end_bar = note_placer.calculate_end_bar(happening_pts_time)
                if not end_bar:
                    continue
                im_overlay = await midi_to_image_fn(end_bar)
                await overlay_queue.put((pts - 1, im_overlay))
                await asyncio.sleep(0)
                first_frame_processed = True
            if not check_any_stop_event():
                im_overlay = await midi_to_image_fn(end_bar)
                await overlay_queue.put((end_pts, im_overlay))
        except Exception as e:
            raise e
        finally:
            await asyncio.sleep(0)
            if not check_any_stop_event():
                if im_overlay is None:
                    im_overlay = await midi_to_image_fn(end_bar)
                await overlay_queue.put((end_pts, im_overlay))
            await asyncio.sleep(0)
            local_stop_event.set()
            await asyncio.sleep(0)
        await writer_task
        return first_frame_processed
