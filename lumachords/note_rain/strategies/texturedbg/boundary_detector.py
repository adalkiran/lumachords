import numpy as np

from lumachords.preferences import Preferences
from lumachords.data_types import DT_RECT, AxisType, as_dt_rect, as_dt_tmp_rect
from lumachords.keybed_detector import KeybedDetectorOutput
from lumachords.utils import Utils

from ...note_rain_utils import NoteRainUtils

class BoundaryDetector:
    """Pairs detected edges into rectangles and processes them into note boundaries."""

    def __init__(self, pref: Preferences, keybed_output: KeybedDetectorOutput):
        self.pref = pref
        self.note_rain_boundary_limits = NoteRainUtils.calculate_boundary_limits(pref, keybed_output)
    
    def rowwise_continuity(self, y0, y1, tol=10):
        s, e = np.minimum(y0, y1), np.maximum(y0, y1)
        v = (y0 >= 0) & (y1 >= 0)

        o = np.argsort(np.where(v, s, np.inf), axis=1)
        s = np.take_along_axis(s, o, 1); e = np.take_along_axis(e, o, 1); v = np.take_along_axis(v, o, 1)

        ec = np.maximum.accumulate(np.where(v, e, -np.inf), axis=1)
        new = np.zeros_like(v, bool)
        new[:, 0] = v[:, 0]
        new[:, 1:] = (s[:, 1:] > ec[:, :-1] + tol) & v[:, 1:]

        R, N = y0.shape
        if not new.any():
            return np.zeros(R, bool), np.full(R, -1), np.full(R, -1)

        idx = np.flatnonzero(new.ravel())
        rows = idx // N
        s_mask = np.where(v, s, np.inf).ravel()
        e_mask = np.where(v, e, -np.inf).ravel()
        y0b = np.minimum.reduceat(s_mask, idx)
        y1b = np.maximum.reduceat(e_mask, idx)
        L = y1b - y0b

        # pick longest block per row (tie-break: earliest)
        ord2 = np.lexsort((np.arange(idx.size), -L, rows))
        ur, first = np.unique(rows[ord2], return_index=True)
        best = ord2[first]

        y0m = np.full(R, np.nan, float); y1m = np.full(R, np.nan, float)
        y0m[ur] = y0b[best].astype(float)
        y1m[ur] = y1b[best].astype(float)

        is_cont = (new.sum(axis=1) == 1) & v.any(axis=1)
        return is_cont, y0m, y1m

    def group_coords_by_proximity(self, arr, proximity_limit):
        arr = np.asarray(arr)
        if arr.size == 0:
            return [], []
        cut = np.flatnonzero(np.diff(arr) > proximity_limit) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, arr.size]
        unique_groups = [np.unique(arr[s:e]) for s, e in zip(starts, ends)]
        idx_groups = [np.arange(s, e) for s, e in zip(starts, ends)]
        return unique_groups, idx_groups

    def pair_lines(self, lines, lim_edge_tickness, im_shape):
        axis_lines_y = lines[lines["axis"] == AxisType.Y]

        if len(axis_lines_y) == 0:
            return np.array([], dtype=DT_RECT), np.array([]), 0
        
        start_lines = axis_lines_y[axis_lines_y["is_start"]]
        end_lines = axis_lines_y[~axis_lines_y["is_start"]]

        if not (len(start_lines) and len(end_lines)):
            return np.array([], dtype=DT_RECT), np.array([]), 0
        
        grouping_proximity_limit = max(5, int(np.ceil(im_shape[1] * self.pref.engine.note_rain_grouping_proximity_limit_rate)))
        tol = int(np.ceil(im_shape[1] * self.pref.engine.note_rain_vertical_continuity_gap_tolerance_rate))
        vert_tol = int(np.ceil(im_shape[1] * self.pref.engine.note_rain_horizontal_line_vertical_snap_tolerance_rate))
        horiz_x_snap_rate = self.pref.engine.note_rain_horizontal_line_horizontal_snap_tolerance_rate
        obs_blocking_overlap_rate = self.pref.engine.note_rain_obstacle_blocking_overlap_rate
        coverage_tol = int(np.ceil(im_shape[1] * self.pref.engine.note_rain_coverage_tolerance_rate))
        blims = self.note_rain_boundary_limits

        # Calculations of start and end lines of vertical lines
        sx = ((start_lines["x0"] + start_lines["x1"]) // 2).astype(np.int32)
        ex = ((end_lines["x0"] + end_lines["x1"]) // 2).astype(np.int32)

        sy0, sy1 = start_lines["y0"], start_lines["y1"]
        ey0, ey1 = end_lines["y0"], end_lines["y1"]

        dx   = ex[None,:] - sx[:,None]
        ok   = (dx >= blims.min_width) & (dx <= blims.max_width)
        y0i  = np.maximum(sy0[:,None], ey0[None,:])
        y1i  = np.minimum(sy1[:,None], ey1[None,:])
        Lint = (y1i - y0i)
        ok  &= (Lint > 0)
        # Avoid pairing very short overlaps against long edges (likely false positives).
        sy_len = (sy1 - sy0)[:, None]
        ey_len = (ey1 - ey0)[None, :]
        ey_sy_max_len = np.maximum(np.maximum(sy_len, ey_len), 1)
        ey_sy_whole_overlap_rate = ey_sy_max_len / im_shape[0]
        overlap_rate = Lint / ey_sy_max_len
        ok &= ((overlap_rate >= 0.25) | (ey_sy_whole_overlap_rate >= 0.75))

        close_matrix = (dx > 0) & (dx < blims.min_width) & (Lint > 0)
        sclose = np.where(np.any(close_matrix, axis=1), np.argmax(close_matrix, axis=1), -1)
        eclose = np.where(np.any(close_matrix, axis=0), np.argmax(close_matrix, axis=0), -1)
        start_best_overlap = np.where(sclose > -1, Lint[np.arange(len(sx)), sclose], -1)
        end_best_overlap = np.where(eclose > -1, Lint[eclose, np.arange(len(ex))], -1)
        strong_start = start_best_overlap >= (sy1 - sy0) * 0.9
        strong_end = end_best_overlap >= (ey1 - ey0) * 0.9
        start_pref_x = np.where(sclose > -1, ex[sclose], np.inf)
        end_pref_x = np.where(eclose > -1, sx[eclose], np.inf)
        start_pref_near = np.abs(ex[None, :] - start_pref_x[:, None]) < blims.min_width
        end_pref_near = np.abs(sx[:, None] - end_pref_x[None, :]) < blims.min_width
        mutual_close = (
            (sclose[:, None] == np.arange(len(ex))[None, :]) &
            (eclose[None, :] == np.arange(len(sx))[:, None])
        )
        start_pref_inside = (sclose[:, None] > -1) & (start_pref_x[:, None] > sx[:, None]) & (start_pref_x[:, None] < ex[None, :])
        end_pref_inside = (eclose[None, :] > -1) & (end_pref_x[None, :] > sx[:, None]) & (end_pref_x[None, :] < ex[None, :])
        spans_two_close_pairs = (
            strong_start[:, None] &
            strong_end[None, :] &
            start_pref_inside &
            end_pref_inside &
            (start_pref_x[:, None] < end_pref_x[None, :])
        )
        no_better_than_close_pairs = (Lint <= start_best_overlap[:, None]) & (Lint <= end_best_overlap[None, :])
        much_worse_overlap = (Lint * 2 < start_best_overlap[:, None]) & (Lint * 2 < end_best_overlap[None, :])
        block_matrix = (
            mutual_close |
            (strong_start[:, None] & start_pref_near) |
            (strong_end[None, :] & end_pref_near) |
            (spans_two_close_pairs & no_better_than_close_pairs) |
            much_worse_overlap
        )
        ok &= ~block_matrix

        filtered_ex_indices = np.flatnonzero(np.any(ok, axis=0)).astype(int)
        filtered_ex = ex[filtered_ex_indices]

        ex_groups, ex_groups_idx = self.group_coords_by_proximity(filtered_ex, proximity_limit=grouping_proximity_limit)

        start_y0_matrix, end_y0_matrix = np.meshgrid(sy0, ey0, indexing='ij')
        start_y0_matrix = np.where(ok, start_y0_matrix, np.nan)
        end_y0_matrix = np.where(ok, end_y0_matrix, np.nan)

        start_y1_matrix, end_y1_matrix = np.meshgrid(sy1, ey1, indexing='ij')
        start_y1_matrix = np.where(ok, start_y1_matrix, np.nan)
        end_y1_matrix = np.where(ok, end_y1_matrix, np.nan)

        # Calculations of start and end lines of horizontal lines
        axis_lines_x = lines[lines["axis"] == AxisType.X]
        h_start_lines = axis_lines_x[axis_lines_x["is_start"]]
        h_end_lines = axis_lines_x[~axis_lines_x["is_start"]]
        h_start_widths = h_start_lines["x1"] - h_start_lines["x0"]
        h_end_widths = h_end_lines["x1"] - h_end_lines["x0"]

        box_candidates = []
        obstacle_lines = []
        all_local_boxes_list = []  # Store all boxes to create obstacle_mask later
        
        for ex_group, ex_group_idx in zip(ex_groups, ex_groups_idx):
            ex_group_indices = filtered_ex_indices[ex_group_idx]
            
            # Extract the relevant columns for this group
            group_end_y0 = end_y0_matrix[:, ex_group_indices]
            group_end_y1 = end_y1_matrix[:, ex_group_indices]
            
            group_is_cont, ex_group_y0m, ex_group_y1m = self.rowwise_continuity(
                group_end_y0, group_end_y1, tol=tol
            )
            
            # Map back to original indices: find which column in the group corresponds to the merged range
            # We need to find which column index (within ex_group_indices) matches the merged y0
            eq = (group_end_y0 == ex_group_y0m[:, None])  # Compare within group
            ex_group_y0midx_local = np.where(np.any(eq, axis=1), np.argmax(eq, axis=1), np.nan)
            
            # Convert local group indices to global end_lines indices
            idx_mask = ~np.isnan(ex_group_y0midx_local)
            start_indices = np.flatnonzero(idx_mask).astype(int)
            
            # Map local group column indices to global end_lines indices
            ex_group_y0midx_global = ex_group_indices[ex_group_y0midx_local[idx_mask].astype(int)]
            
            filtered_sx = sx[start_indices]

            local_boxes_list = []
            sx_groups, sx_groups_idx = self.group_coords_by_proximity(filtered_sx, proximity_limit=grouping_proximity_limit)
            
            for sx_group, sx_group_idx in zip(sx_groups, sx_groups_idx):
                sx_group_indices = start_indices[sx_group_idx]
                end_indices = ex_group_y0midx_global[sx_group_idx]  # Use global indices

                merged_start_y0 = start_y0_matrix[sx_group_indices, end_indices]
                merged_start_y1 = start_y1_matrix[sx_group_indices, end_indices]
                local_end_y0 = ex_group_y0m[sx_group_indices]
                local_end_y1 = ex_group_y1m[sx_group_indices]
                n_candidates = len(local_end_y0)

                merged_x0_val  = int(np.median(sx[sx_group_indices]))  # anchor all boxes in this group to a common x
                local_x0       = np.full((n_candidates), merged_x0_val, dtype="i2")
                local_x1       = ex[end_indices]
                local_overlap_rate = overlap_rate[sx_group_indices, end_indices]
                extend_mask = local_overlap_rate >= 0.7
                local_crop_y0  = np.where(
                    (np.abs(merged_start_y0 - local_end_y0) <= vert_tol) | extend_mask,
                    np.minimum(merged_start_y0, local_end_y0),
                    np.maximum(merged_start_y0, local_end_y0),
                )
                local_crop_y1  = np.where(
                    (np.abs(merged_start_y1 - local_end_y1) <= vert_tol) | extend_mask,
                    np.maximum(merged_start_y1, local_end_y1),
                    np.minimum(merged_start_y1, local_end_y1),
                )

                uniq_stack = np.stack([
                    local_x0, # x0
                    local_crop_y0, # y0
                    local_x1, # x1
                    local_crop_y1, #y1
                ], axis=1).astype('i2', copy=False)
                uniq_stack, uniq_idx = np.unique(uniq_stack, axis=0, return_index=True)
                local_x0, local_crop_y0, local_x1, local_crop_y1 = uniq_stack.T
                local_merged_y0 = merged_start_y0[uniq_idx]
                local_merged_y1 = merged_start_y1[uniq_idx]

                n_candidates = len(local_x0)    

                sx_group_boxes = as_dt_tmp_rect(np.stack([
                    np.full((n_candidates), -1, dtype="i2"), # id
                    local_x0, # x0
                    local_crop_y0, # crop_y0
                    local_merged_y0, # merged_y0
                    local_x1, # x1
                    local_crop_y1, #crop_y1
                    local_merged_y1, # merged_y1
                    np.full((n_candidates), -1, dtype="i2"), # top line index, if exists
                    np.full((n_candidates), -1, dtype="i2"), # bottom line index, if 
                    np.full((n_candidates), -999, dtype="i2"), # top line snap diff, if exists
                    np.full((n_candidates), -999, dtype="i2") # bottom line snap diff, if exists
                ], axis=1).astype('i2', copy=False))
                if len(h_start_lines) and len(h_end_lines):
                    for sb_idx, sx_group_box in enumerate(sx_group_boxes):
                        # Snap top
                        ix0 = np.maximum(sx_group_box["x0"], h_start_lines["x0"])
                        ix1 = np.minimum(sx_group_box["x1"], h_start_lines["x1"])
                        Lx = ix1 - ix0
                        Lx_limit = (sx_group_box["x1"]-sx_group_box["x0"]) * horiz_x_snap_rate
                        for try_y0 in (sx_group_box["crop_y0"], sx_group_box["merged_y0"]):
                            dy = try_y0 - h_start_lines["y1"]
                            mask = (Lx > Lx_limit) & (dy >= -vert_tol) & (dy <= vert_tol)
                            if np.any(mask):
                                candidate_indices = np.flatnonzero(mask)
                                dy_candidates = dy[candidate_indices]
                                cand_widths = h_start_widths[candidate_indices]
                                # Lines above the box top or width criteria (is_narrower) for curved/elliptic rectangles
                                is_above = dy_candidates >= 0
                                is_narrower = cand_widths <= (sx_group_box["x1"] - sx_group_box["x0"]) * 0.9
                                is_preferred_candidate = is_above | is_narrower  # Above lines or narrower curved/elliptic candidates rank first
                                # (above or narrower) gets 0, otherwise gets 1 -> (above or narrower) preferred.
                                # Preferred candidates stay ahead of fallback candidates, while wider lines rank first within each bucket
                                combined_rank = (~is_preferred_candidate).astype(cand_widths.dtype) * (np.max(cand_widths) + 1) - cand_widths
                                order = np.lexsort((np.abs(dy_candidates), combined_rank))
                                chosen_idx = candidate_indices[order[0]]
                                top_line = h_start_lines[chosen_idx]
                                sx_group_box["top_line_idx"] = chosen_idx
                                if (try_y0 > top_line["y0"]) or (top_line["y0"] >= try_y0 - vert_tol):
                                    sx_group_box["snap_diff_top"] = int(try_y0 - top_line["y0"])
                                    sx_group_box["crop_y0"] = top_line["y0"]
                                    break
                        # Snap bottom
                        ix0 = np.maximum(sx_group_box["x0"], h_end_lines["x0"])
                        ix1 = np.minimum(sx_group_box["x1"], h_end_lines["x1"])
                        Lx = ix1 - ix0
                        for try_y1 in (sx_group_box["crop_y1"], sx_group_box["merged_y1"]):
                            dy = h_end_lines["y0"] - try_y1
                            mask = (Lx > Lx_limit) & (dy >= -vert_tol) & (dy <= vert_tol)
                            if np.any(mask):
                                candidate_indices = np.flatnonzero(mask)
                                dy_candidates = dy[candidate_indices]
                                cand_widths = h_end_widths[candidate_indices]
                                # Lines below the box bottom or width criteria (is_narrower) for curved/elliptic rectangles
                                is_below = dy_candidates >= 0
                                is_narrower = cand_widths <= (sx_group_box["x1"] - sx_group_box["x0"]) * 0.9
                                is_preferred_candidate = is_below | is_narrower   # below lines or narrower curved/elliptic candidates rank first
                                # (below or narrower) gets 0, otherwise gets 1 -> (below or narrower) preferred.
                                # Preferred candidates stay ahead of fallback candidates, while wider lines rank first within each bucket
                                combined_rank = (~is_preferred_candidate).astype(cand_widths.dtype) * (np.max(cand_widths) + 1) - cand_widths
                                order = np.lexsort((np.abs(dy_candidates), combined_rank))
                                chosen_idx = candidate_indices[order[0]]
                                bottom_line = h_end_lines[chosen_idx]
                                sx_group_box["bottom_line_idx"] = chosen_idx
                                if (try_y1 < bottom_line["y1"]) or (bottom_line["y1"] <= try_y1 + vert_tol):
                                    sx_group_box["snap_diff_bottom"] = int(bottom_line["y1"] - try_y1)
                                    sx_group_box["crop_y1"] = bottom_line["y1"]
                                    break

                    # Split vertically merged spans when internal horizontal boundaries indicate a gap.
                    split_boxes = []
                    for sx_group_box in sx_group_boxes:
                        width = sx_group_box["x1"] - sx_group_box["x0"]
                        Lx_limit = width * horiz_x_snap_rate
                        split_Lx_limit = width * 0.75
                        qy0, qy1 = sx_group_box["crop_y0"], sx_group_box["crop_y1"]

                        ix0 = np.maximum(sx_group_box["x0"], h_end_lines["x0"])
                        ix1 = np.minimum(sx_group_box["x1"], h_end_lines["x1"])
                        Lx_end = ix1 - ix0
                        internal_bottoms = h_end_lines["y1"][
                            (Lx_end > Lx_limit) &
                            (Lx_end > split_Lx_limit) &
                            (h_end_lines["y1"] > qy0 + vert_tol) &
                            (h_end_lines["y1"] < qy1 - vert_tol)
                        ]

                        ix0 = np.maximum(sx_group_box["x0"], h_start_lines["x0"])
                        ix1 = np.minimum(sx_group_box["x1"], h_start_lines["x1"])
                        Lx_start = ix1 - ix0
                        internal_tops = h_start_lines["y0"][
                            (Lx_start > Lx_limit) &
                            (Lx_start > split_Lx_limit) &
                            (h_start_lines["y0"] > qy0 + vert_tol) &
                            (h_start_lines["y0"] < qy1 - vert_tol)
                        ]

                        did_split = False
                        if len(internal_bottoms) and len(internal_tops):
                            bottoms = np.unique(np.asarray(internal_bottoms, dtype=np.int32))
                            tops = np.unique(np.asarray(internal_tops, dtype=np.int32))
                            pair_candidates = np.transpose(np.nonzero(tops[None, :] - bottoms[:, None] > vert_tol))
                            if len(pair_candidates):
                                gaps = tops[pair_candidates[:, 1]] - bottoms[pair_candidates[:, 0]]
                                best_pair = pair_candidates[np.argmin(gaps)]
                                split_y1 = int(bottoms[best_pair[0]])
                                split_y0 = int(tops[best_pair[1]])
                                if ((split_y1 - qy0) >= blims.min_height) and ((qy1 - split_y0) >= blims.min_height):
                                    top_box = sx_group_box.copy()
                                    bottom_box = sx_group_box.copy()
                                    top_box["crop_y1"] = split_y1
                                    bottom_box["crop_y0"] = split_y0
                                    split_boxes.append(top_box)
                                    split_boxes.append(bottom_box)
                                    did_split = True

                        if not did_split:
                            split_boxes.append(sx_group_box.copy())

                    sx_group_boxes = np.asarray(split_boxes, dtype=sx_group_boxes.dtype)
                height_mask = ((sx_group_boxes["crop_y1"] - sx_group_boxes["crop_y0"]) >= blims.min_height)
                sx_group_boxes = sx_group_boxes[height_mask]

                # Merge vertically contiguous boxes that share (almost) the same x-span
                if len(sx_group_boxes) > 1:
                    # Use pre-snap extents to decide continuity so horizontal snapping does not artificially bridge gaps
                    ref_boxes = np.stack([
                        sx_group_boxes["x0"],
                        sx_group_boxes["crop_y0"],
                        sx_group_boxes["x1"],
                        sx_group_boxes["crop_y1"],
                    ], axis=1).astype('i2', copy=False)
                    order = sorted(range(len(sx_group_boxes)), key=lambda i: (sx_group_boxes["x1"][i], sx_group_boxes["x0"][i], sx_group_boxes["crop_y0"][i]))
                    merged_boxes = []
                    merged_refs = []
                    (dim_ref_x0, dim_ref_y0, dim_ref_x1, dim_ref_y1) = range(0, 4)
                    for idx in order:
                        box = sx_group_boxes[idx]
                        ref = ref_boxes[idx]
                        if merged_boxes:
                            last = merged_boxes[-1]
                            last_ref = merged_refs[-1]
                            same_x = (abs(int(last["x0"]) - int(box["x0"])) <= grouping_proximity_limit) and (abs(int(last["x1"]) - int(box["x1"])) <= grouping_proximity_limit)
                            y_gap = max(int(ref[dim_ref_y0]) - int(last_ref[dim_ref_y1]), int(last_ref[dim_ref_y0]) - int(ref[dim_ref_y1]), 0)
                            contig_y = y_gap < vert_tol  # allow small gap but keep clearly separated spans apart
                            # Do not merge past explicit boundaries
                            blocked = (last["bottom_line_idx"] > -1) or (box["top_line_idx"] > -1)
                            if same_x and contig_y and not blocked:
                                last["x0"] = min(last["x0"], box["x0"])
                                last["crop_y0"] = min(last["crop_y0"], box["crop_y0"])
                                last["x1"] = max(last["x1"], box["x1"])
                                last["crop_y1"] = max(last["crop_y1"], box["crop_y1"])
                                # Update reference span as well
                                last_ref[dim_ref_x0] = min(last_ref[dim_ref_x0], ref[dim_ref_x0])
                                last_ref[dim_ref_x1] = max(last_ref[dim_ref_x1], ref[dim_ref_x1])
                                continue
                        merged_boxes.append(box.copy())
                        merged_refs.append(ref.copy())
                    sx_group_boxes = np.asarray(merged_boxes, dtype=sx_group_boxes.dtype)

                local_boxes_list.append(sx_group_boxes)

            if len(local_boxes_list) == 0:
                continue
                
            local_boxes = np.hstack(local_boxes_list)
            all_local_boxes_list.append(local_boxes)

        # Process obstacles after collecting all boxes
        if len(all_local_boxes_list) == 0:
            return np.array([], dtype=DT_RECT), np.array([]), coverage_tol
            
        all_boxes = np.hstack(all_local_boxes_list)
        obstacle_mask = np.full(len(all_boxes), False)
        
        # Now process obstacles with correct global indexing
        for ex_group, ex_group_idx in zip(ex_groups, ex_groups_idx):
            ex_group_indices = filtered_ex_indices[ex_group_idx]
            
            # Find which boxes belong to this ex_group
            group_box_mask = np.isin(all_boxes["x1"], ex[ex_group_indices])  # boxes where x1 is in this ex_group
            group_box_indices = np.flatnonzero(group_box_mask)
            
            for global_idx in group_box_indices:
                local_box = all_boxes[global_idx]
                local_obstacle_lines = []
                
                qx0, qy0, qx1, qy1 = local_box[["x0", "crop_y0", "x1", "crop_y1"]]
                
                # Check end line obstacles
                obs_ex_mask = ((ex<qx1)[:,None] & (ex[:,None]>(qx0 + 0)) & ~(np.isin(ex, ex_group)[:, None])).ravel()
                obs_ey_intersection = np.where(obs_ex_mask.ravel(), (
                    np.minimum(ey1, qy1) -
                    np.maximum(ey0, qy0)
                ), np.nan)
                obs_e_mask = obs_ey_intersection > (qy1-qy0)*obs_blocking_overlap_rate
                obs_eclose_mask = obs_e_mask & (eclose > -1)
                obs_eclose_indices = np.flatnonzero(obs_eclose_mask)
                obs_e_sclose_x = sx[eclose[obs_eclose_indices]]
                obs_eclose_indices = obs_eclose_indices[obs_e_sclose_x == qx0]
                obs_eclose_lines = start_lines[eclose[obs_eclose_indices]]
                obs_e_mask[obs_eclose_indices] = (qy1-qy0) < (obs_eclose_lines["y1"]-obs_eclose_lines["y0"])*obs_blocking_overlap_rate
                obstacle_candidates = end_lines[obs_e_mask]
                if len(obstacle_candidates):
                    local_obstacle_lines.append(obstacle_candidates)

                # Check start line obstacles
                obs_sx_mask = ((sx<(qx1 - 0))[:,None] & (sx[:,None]>qx0)).ravel()
                obs_sy_intersection = np.where(obs_sx_mask.ravel(), (
                    np.minimum(sy1, qy1) -
                    np.maximum(sy0, qy0)
                ), np.nan)
                obs_s_mask = obs_sy_intersection > (qy1-qy0)*obs_blocking_overlap_rate
                obs_sclose_mask = obs_s_mask & (sclose > -1)
                obs_sclose_indices = np.flatnonzero(obs_sclose_mask)
                obs_s_eclose_x = ex[sclose[obs_sclose_indices]]
                obs_sclose_indices = obs_sclose_indices[obs_s_eclose_x == qx1]
                obs_sclose_lines = end_lines[sclose[obs_sclose_indices]]
                obs_s_mask[obs_sclose_indices] = (qy1-qy0) < (obs_sclose_lines["y1"]-obs_sclose_lines["y0"])*obs_blocking_overlap_rate
                obstacle_candidates = start_lines[obs_s_mask]
                
                if len(obstacle_candidates):
                    non_edge_obstacles = []
                    for obstacle_candidate in obstacle_candidates:
                        boxes_with_obstacle_is_edge = (
                            (np.abs(all_boxes["x0"] - obstacle_candidate["x0"]) < grouping_proximity_limit) &
                            (np.abs(all_boxes["crop_y0"] - obstacle_candidate["y0"]) < vert_tol) &
                            (np.abs(all_boxes["crop_y1"] - obstacle_candidate["y1"]) < vert_tol)
                        )
                        if not np.any(boxes_with_obstacle_is_edge):
                            non_edge_obstacles.append(obstacle_candidate)
                    if len(non_edge_obstacles):
                        local_obstacle_lines.append(np.array(non_edge_obstacles, dtype=start_lines.dtype))
                            
                if len(local_obstacle_lines):
                    obstacle_lines.extend(local_obstacle_lines)
                    obstacle_mask[global_idx] = True

        # Build final boxes array
        if len(all_boxes):
            key_idx_col = np.full((len(all_boxes), 1), -1, dtype="i2")
            is_valid_col = (~obstacle_mask).astype("i2").reshape(-1, 1)
            box_candidates = np.hstack((
                np.array(all_boxes[["id", "x0", "crop_y0", "x1", "crop_y1"]].tolist(), dtype="i2"), 
                key_idx_col, 
                is_valid_col, 
                np.array(all_boxes[["snap_diff_top", "snap_diff_bottom"]].tolist(), dtype="i2")
            ))
        else:
            box_candidates = np.array([])
            
        boxes = as_dt_rect(box_candidates) if len(box_candidates) else np.array([], dtype=DT_RECT)
        boxes = Utils.invalidate_covered_rectangles(boxes, coverage_tol)
        boxes = boxes[np.lexsort((boxes["y1"], boxes["x1"], boxes["y0"], boxes["x0"]))]
        
        obstacle_lines = [arr for arr in obstacle_lines if (isinstance(arr, list) and len(arr[0]))]
        obstacle_lines = np.hstack(obstacle_lines) if len(obstacle_lines) else np.array([])
        obstacle_lines = obstacle_lines[0:0]
        return boxes, obstacle_lines, coverage_tol
