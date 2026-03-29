from collections import deque
from dataclasses import dataclass
from typing import Callable
import numpy as np

from lumachords.data_types import DT_BOX_EVENT, BoxIsValid, as_dt_rect
from lumachords.preferences import Preferences
from lumachords.rendering import TrackerRenderer
from lumachords.utils import Utils

@dataclass
class TrackEntry:
    cx: int
    cy: int
    last_seen: int = -1


class NoteRainTracker:
    def __init__(self, pref: Preferences, actual_fps: int, keybed_output, tracking_y_band_tuple, play_y, velocity_consensus_callback_fn: Callable[[float], None]=None):
        self.pref = pref
        self.actual_fps = actual_fps
        self.keybed_output = keybed_output
        self.keybed_top_y = keybed_output.keybed_bounds[1]
        self.tracking_y0 = tracking_y_band_tuple[0]
        self.tracking_y1 = tracking_y_band_tuple[1]
        self.play_y = play_y
        self.velocity_consensus_callback_fn = velocity_consensus_callback_fn
        self.prev_boxes = None          # structured array (x0,y0,x1,y1,...)
        self.prev_ids   = None          # np.ndarray[int], same length as prev_boxes
        self.next_id    = 1             # next ID to assign
        self.track_entries: dict[int, TrackEntry] = {} # id->TrackEntry
        self.last_pts = -1
        self.last_unadjusted_pts = -1
        require_secs_before_calculating_velocity_consensus = 2
        require_pts_before_calculating_velocity_consensus = require_secs_before_calculating_velocity_consensus * actual_fps
        self.latest_median_velocities = deque(maxlen=require_pts_before_calculating_velocity_consensus)
        self.velocity_consensus = -1
        self.last_median_velocity = -1

        self.prev_inside_play_scope_ids = None
    
    def match_boxes(self, prev_key_idx, prev_cx, prev_cy, prev_y0, prev_y1, cur_key_idx, cur_cx, cur_cy, cur_y0, cur_y1, max_dx, max_dy, alpha=0.5, unadjusted_pts_delta=-1):
        """
        Match boxes between two frames while preserving left-right order.

        max_dx: max allowed horizontal shift (pixels)
        max_dy: max allowed vertical shift (pixels)
        alpha:  vertical weight in distance (larger => care more about dy)
        """
        if self.velocity_consensus > 0:
            velocity = self.velocity_consensus
        elif self.last_median_velocity > 0:
            velocity = self.last_median_velocity
        elif len(self.latest_median_velocities):
            latest_median_velocities_list = list(self.latest_median_velocities)
            velocity = np.mean(latest_median_velocities_list)
        else:
            velocity = 0
        min_dy = (velocity * 0.5) * 0.7
        Np, Nn = len(prev_cx), len(cur_cx)

        prev2cur = np.full(Np, -1, dtype=int)
        cur2prev = np.full(Nn, -1, dtype=int)
        if Np == 0 or Nn == 0:
            return prev2cur, cur2prev

        # Sort left→right in each frame
        prev_order = np.lexsort((prev_cy, prev_key_idx)) #prev_cx
        cur_order = np.lexsort((cur_cy, cur_key_idx)) #cur_cx

        j_start = 0
        for pi in range(Np):
            i = prev_order[pi]
            if j_start >= Nn:
                break

            js = cur_order[j_start:]  # candidate current indices (sorted by x)
            dx = cur_cx[js] - prev_cx[i]

            use_y1_dy = (prev_y0[i] < 3) | (cur_y0[js] < 3)
            dy = np.where(use_y1_dy, cur_y1[js] - prev_y1[i], cur_cy[js] - prev_cy[i])

            # motion + already-matched constraints
            mask = (
                (np.abs(dx) <= max_dx) &
                (np.abs(dy) <= max_dy) &
                (cur2prev[js] == -1)
            )
            if not mask.any():
                continue

            valid = np.where(mask)[0]

            if len(valid) > 1:
                mask = mask & (dy >= min_dy)
                if mask.any():
                    valid = np.where(mask)[0]

            # distance-like cost: prefer small |dx| and |dy|
            # alpha controls how "expensive" vertical gap is
            cost = dx[valid]**2 + (alpha * dy[valid])**2
            best_loc = valid[np.argmin(cost)]
            best_j_idx = j_start + best_loc
            j = cur_order[best_j_idx]

            prev2cur[i] = j
            cur2prev[j] = i
            j_start = best_j_idx + 1  # non-crossing

        return prev2cur, cur2prev

    def track_boxes_once(self, prev_boxes, cur_boxes, im_vis):
        return TrackerRenderer.track_boxes_once(prev_boxes, cur_boxes, im_vis, self.match_boxes)

    def track_play_line(self, cur_boxes, cur_median_velocity, transpose_octaves):
        play_y0 = self.play_y - cur_median_velocity
        play_y1 = self.play_y
        cur_boxes = cur_boxes[cur_boxes["is_valid"] > 0]
        cur_lengths = cur_boxes["y1"] - cur_boxes["y0"]

        play_scope_mask = (cur_boxes["y1"] > play_y0)
        inside_play_scope_mask = play_scope_mask & (cur_boxes["y0"] < play_y0)
        out_of_play_scope_mask = play_scope_mask & (cur_boxes["y0"] >= play_y0)

        out_and_short_mask = out_of_play_scope_mask & (cur_lengths < cur_median_velocity) & ((cur_boxes["y0"] < play_y1))
        inside_play_scope_mask |= out_and_short_mask

        cur_inside_play_scope_ids = cur_boxes["id"][inside_play_scope_mask]
        cur_out_of_play_scope_ids = cur_boxes["id"][out_of_play_scope_mask]

        id_dtype = cur_out_of_play_scope_ids.dtype

        if self.prev_inside_play_scope_ids is not None:
            off_event_ids = np.intersect1d(self.prev_inside_play_scope_ids, cur_out_of_play_scope_ids)
            on_event_ids = np.setdiff1d(cur_inside_play_scope_ids, self.prev_inside_play_scope_ids)
        else:
            off_event_ids = np.array([], dtype=id_dtype)
            on_event_ids = cur_inside_play_scope_ids.copy()
        off_event_boxes = cur_boxes[np.isin(cur_boxes["id"], off_event_ids)]
        off_event_time_delta = -(off_event_boxes["y0"] - play_y1) / (cur_median_velocity * self.actual_fps)

        on_event_boxes = cur_boxes[np.isin(cur_boxes["id"], on_event_ids)]
        on_event_time_delta = (-(on_event_boxes["y1"] - play_y1) / (cur_median_velocity * self.actual_fps)) if cur_median_velocity != 0 else np.zeros_like(on_event_boxes["y1"])

        off_events = np.empty(len(off_event_boxes), dtype=DT_BOX_EVENT)
        off_events["id"], off_events["is_on"], off_events["key_idx"], off_events["time_delta"] = off_event_boxes["id"], False, off_event_boxes["key_idx"], off_event_time_delta
        on_events = np.empty(len(on_event_boxes), dtype=DT_BOX_EVENT)
        on_events["id"], on_events["is_on"], on_events["key_idx"], on_events["time_delta"] = on_event_boxes["id"], True, on_event_boxes["key_idx"], on_event_time_delta
        events = np.hstack((off_events, on_events))
        events = events[events["time_delta"].argsort()]
        event_midi_nums = []
        for key_idx in events["key_idx"]:
            key_data = self.keybed_output.all_keys_data[key_idx]
            midi_num = key_data["midi_num"] + transpose_octaves * 12
            event_midi_nums.append(midi_num)
        events["midi_num"] = event_midi_nums
        self.prev_inside_play_scope_ids = cur_inside_play_scope_ids
        return events
    
    def set_velocity_consensus(self, value):
        if self.velocity_consensus != value and self.velocity_consensus_callback_fn is not None:
            self.velocity_consensus_callback_fn(value)
        self.velocity_consensus = value

    def step_frame(self, pts: int, cur_boxes, coverage_tol, transpose_octaves, complete_occlusions=True, filter_for_tracking=True):
        unadjusted_pts_delta = pts - 1 - self.last_unadjusted_pts
        max_dx=4
        max_dy=50
        alpha=0.5
        cur_boxes = np.asarray(cur_boxes)
        N_cur = len(cur_boxes)
        cur_ids = np.full(N_cur, -1, dtype=int)

        # Compute centers for current frame (structured → 1D arrays)
        cur_cx = (cur_boxes["x0"] + cur_boxes["x1"]) * 0.5
        cur_cy = (cur_boxes["y0"] + cur_boxes["y1"]) * 0.5

        # First frame: just assign new IDs
        if self.prev_boxes is None or len(self.prev_boxes) == 0:
            if filter_for_tracking:
                cur_boxes = cur_boxes[cur_boxes["y0"] <= self.tracking_y0]
            N_cur = len(cur_boxes)
            cur_ids = np.full(N_cur, -1, dtype=int)

            if N_cur > 0:
                cur_ids = np.arange(self.next_id, self.next_id + N_cur, dtype=int)
                self.next_id += N_cur
            self.prev_boxes = cur_boxes.copy()
            self.prev_ids   = cur_ids.copy()
            cur_boxes["id"] = cur_ids
            events = self.track_play_line(cur_boxes, 0, transpose_octaves)
            self.last_pts = pts
            self.last_unadjusted_pts = pts
            return cur_boxes, events

        # Later frames: match to previous
        prev_cx = (self.prev_boxes["x0"] + self.prev_boxes["x1"]) * 0.5
        prev_cy = (self.prev_boxes["y0"] + self.prev_boxes["y1"]) * 0.5

        prev2cur, cur2prev = self.match_boxes(
            self.prev_boxes["key_idx"], prev_cx, prev_cy, self.prev_boxes["y0"], self.prev_boxes["y1"],
            cur_boxes["key_idx"], cur_cx, cur_cy, cur_boxes["y0"], cur_boxes["y1"],
            max_dx=max_dx, max_dy=max_dy, alpha=alpha, unadjusted_pts_delta=unadjusted_pts_delta,
        )

        # Matched: inherit previous IDs
        matched_prev = np.flatnonzero(prev2cur >= 0)
        matched_cur  = prev2cur[matched_prev]
        cur_ids[matched_cur] = self.prev_ids[matched_prev]

        prev_y0_matched = self.prev_boxes["y0"][matched_prev]
        prev_y1_matched = self.prev_boxes["y1"][matched_prev]
        cur_y0_matched = cur_boxes["y0"][matched_cur]
        cur_y1_matched = cur_boxes["y1"][matched_cur]
        use_y1_velocity = (prev_y0_matched < self.tracking_y0) | (cur_y0_matched < self.tracking_y0)
        velocity_y_delta = np.where(use_y1_velocity, cur_y1_matched - prev_y1_matched, cur_cy[matched_cur] - prev_cy[matched_prev])
        cur_velocities_arr_tmp = np.divide(velocity_y_delta, pts - self.last_pts) if (self.last_pts > -1 and (pts - self.last_pts)) > 0 else np.full((len(matched_cur)), -1)
        cur_velocities_arr_tmp = cur_velocities_arr_tmp[
            (cur_boxes["is_valid"][matched_cur] > 0) &
            ((cur_velocities_arr_tmp > 0) | ((cur_boxes["y0"][matched_cur] >= self.tracking_y0) & (cur_boxes["y0"][matched_cur] <= self.tracking_y1)))
        ]
        if len(cur_velocities_arr_tmp):
            cur_velocities_md = np.median(cur_velocities_arr_tmp)
            cur_velocities_std = np.std(cur_velocities_arr_tmp)
            cur_velocities_arr = cur_velocities_arr_tmp[(cur_velocities_arr_tmp >= (cur_velocities_md-cur_velocities_std)) & (cur_velocities_arr_tmp <= (cur_velocities_md+2*cur_velocities_std))]
            cur_median_velocity_calculated = int(np.ceil(np.median(cur_velocities_arr))) if len(cur_velocities_arr) > 0 else -1
        else:
            cur_median_velocity_calculated = -1

        if np.any(cur_velocities_arr_tmp <= 0):
            cur_median_velocity = self.velocity_consensus
        elif not len(cur_velocities_arr_tmp) or cur_median_velocity_calculated < 5:
            cur_median_velocity = -1
        else:
            cur_median_velocity = cur_median_velocity_calculated
        latest_median_velocities_mean = int(np.ceil(np.median(list(self.latest_median_velocities)))) if len(self.latest_median_velocities) else -1
        if cur_median_velocity > -1:
            self.latest_median_velocities.append(float(cur_median_velocity))
            play_y0 = self.play_y - cur_median_velocity
            any_box_over_play_y = len(cur_boxes[cur_boxes["y1"] >= play_y0])
            if (
                (self.velocity_consensus < 0) and 
                (
                    (len(self.latest_median_velocities) == self.latest_median_velocities.maxlen) or
                    (any_box_over_play_y and len(self.latest_median_velocities) >= 5)
                )
            ):
                latest_median_velocities_list = list(self.latest_median_velocities)
                latest_median_velocities_list = latest_median_velocities_list[len(latest_median_velocities_list)//4:] # take the last 75% of items
                self.set_velocity_consensus(np.mean(latest_median_velocities_list))
        adjust_cur_boxes = False
        if len(self.latest_median_velocities) and latest_median_velocities_mean > -1 and (latest_median_velocities_mean - cur_median_velocity_calculated) > 0.7 * latest_median_velocities_mean:
            cur_median_velocity = latest_median_velocities_mean
            adjust_cur_boxes = True

        # Unmatched current boxes: assign new IDs
        cur_tracking_y0 = self.tracking_y0
        if (self.last_unadjusted_pts > -1) and unadjusted_pts_delta > 0:
            cur_tracking_y0 += unadjusted_pts_delta * cur_median_velocity
        new_mask = (cur_ids < 0) & (cur_boxes["y0"] < cur_tracking_y0)
        cur_ids[new_mask] = -2

        unmatched_prev = np.flatnonzero(prev2cur < 0)
        unmatched_prev_ids  = self.prev_ids[unmatched_prev]

        is_adjusted = False
        if len(unmatched_prev):
            if complete_occlusions:
                is_adjusted = True
                for prev_id, (_, prev_x0, prev_y0, prev_x1, prev_y1, prev_key_idx, prev_is_valid, prev_snap_diff_top, prev_snap_diff_bottom) in zip(unmatched_prev_ids, self.prev_boxes[unmatched_prev]):
                    if prev_id not in self.track_entries:
                        continue
                    if prev_y0 < 3:
                        # if the box at previous frame has an invisible part behind the top edge
                        estimated_y0 = prev_y0
                        estimated_y1 = prev_y1 + cur_median_velocity // 2
                    else:
                        # if all edges of the box at previous frame are visible
                        estimated_y0 = prev_y0 + cur_median_velocity
                        estimated_y1 = prev_y1 + cur_median_velocity
                    if cur_median_velocity > 0 and estimated_y0 < self.tracking_y1:
                        is_valid = (prev_is_valid + 1) if prev_is_valid > 0 else prev_is_valid
                        if estimated_y0 > self.tracking_y0 or is_valid < 5:
                            possibly_y_delta = unadjusted_pts_delta * cur_median_velocity
                            possbly_same_cur_box_indices = np.flatnonzero((cur_boxes["key_idx"] == prev_key_idx) & (np.abs(cur_boxes["y0"] - prev_y0 + possibly_y_delta) < 3) & (np.abs(cur_boxes["y1"] - prev_y1 + possibly_y_delta) < 3))
                            possbly_same_cur_box_idx = possbly_same_cur_box_indices[0] if (len(possbly_same_cur_box_indices) == 1 and cur_ids[possbly_same_cur_box_indices[0]] == -2) else -1
                            estimated_box_data = (prev_id, prev_x0, estimated_y0, prev_x1, estimated_y1, prev_key_idx, is_valid, prev_snap_diff_top, prev_snap_diff_bottom)
                            if possbly_same_cur_box_idx > -1:
                                cur_boxes[possbly_same_cur_box_idx] = estimated_box_data
                                cur_ids[possbly_same_cur_box_idx] = prev_id
                            else:
                                cur_boxes = np.concatenate((cur_boxes, as_dt_rect([estimated_box_data])))
                                cur_ids = np.concatenate((cur_ids, [prev_id]))
                    else:
                        del self.track_entries[prev_id]
            else:
                for prev_id in unmatched_prev_ids:
                    if prev_id in self.track_entries:
                        del self.track_entries[prev_id]
        if adjust_cur_boxes and complete_occlusions:
            is_adjusted = True
            for cur_id, (_, prev_x0, prev_y0, prev_x1, prev_y1, prev_key_idx, prev_is_valid, prev_snap_diff_top, prev_snap_diff_bottom) in zip(cur_ids[matched_cur], self.prev_boxes[matched_prev]):
                if cur_id not in self.track_entries:
                    continue
                estimated_y0 = prev_y0 + cur_median_velocity
                estimated_y1 = prev_y1 + cur_median_velocity
                if estimated_y1 > self.keybed_top_y and estimated_y0 < self.keybed_top_y:
                    estimated_y1 = np.minimum(self.keybed_top_y, estimated_y1)
                if cur_median_velocity > 0:
                    is_valid = (prev_is_valid + 1) if prev_is_valid > 0 else prev_is_valid
                    cur_boxes[cur_ids == cur_id] = as_dt_rect([(cur_id, prev_x0, estimated_y0, prev_x1, estimated_y1, prev_key_idx, is_valid, prev_snap_diff_top, prev_snap_diff_bottom)])

        new_mask = (cur_ids == -2)
        new_count = int(new_mask.sum())
        if new_count > 0:
            new_ids = np.arange(self.next_id, self.next_id + new_count, dtype=int)
            self.next_id += new_count
            cur_ids[new_mask] = new_ids

        if is_adjusted:
            ord_after_adjust = np.lexsort((cur_boxes["y0"], cur_boxes["x0"]))
            cur_boxes = cur_boxes[ord_after_adjust]
            cur_ids = cur_ids[ord_after_adjust]
            # Compute centers for current frame (structured → 1D arrays)
            cur_cx = (cur_boxes["x0"] + cur_boxes["x1"]) * 0.5
            cur_cy = (cur_boxes["y0"] + cur_boxes["y1"]) * 0.5

        cur_boxes["id"] = cur_ids
        cur_boxes["is_valid"][cur_ids < 0] = BoxIsValid.Invalid

        Utils.invalidate_covered_rectangles(cur_boxes, coverage_tol)
        events = self.track_play_line(cur_boxes, cur_median_velocity, transpose_octaves)

        # Update state for next frame
        prev_boxes_mask = (cur_ids > -1)
        self.prev_boxes = cur_boxes.copy()[prev_boxes_mask]
        self.prev_ids   = cur_ids.copy()[prev_boxes_mask]

        # Update track history
        for id, cx, cy in zip(cur_ids, cur_cx, cur_cy):
            id = int(id)
            if id < 0:
                continue
            entry = self.track_entries.get(id, None)
            if entry is None:
                entry = TrackEntry(cx, cy)
                self.track_entries[id] = entry
            entry.cx = cx
            entry.cy = cy
            entry.last_seen = pts

        self.last_pts = pts
        if not adjust_cur_boxes:
            self.last_unadjusted_pts = pts
        self.last_median_velocity = cur_median_velocity

        return cur_boxes, events
