from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from lumachords.processing_state import ProcessingState

@dataclass
class Viewport:
    x: int
    y: int
    w: int
    h: int
    dynw: int = None
    dynh: int = None

    @property
    def right(self):
        return self.x + (self.dynw or self.w)

    @property
    def bottom(self):
        return self.y

    def as_tuple(self):
        return (self.x, self.y, self.w, self.h)

class PositionRule(Enum):
    RIGHT_OF = auto()
    LEFT_OF  = auto()
    BELOW    = auto()
    ABOVE    = auto()
    CENTER_IN = auto()
    ABOVE_BOTTOM_LEFT_OF = auto()


class TitlePosition(Enum):
    OUTSIDE_TOP_CENTER = auto()
    INSIDE_TOP_LEFT = auto()


@dataclass
class Panel:
    panel_id: int
    panel_name: str
    title: Optional[str]
    title_pos: TitlePosition = TitlePosition.OUTSIDE_TOP_CENTER
    viewport: Optional[Viewport] = None
    image_tex: Optional[int] = None                      # OpenGL texture id
    title_tex: Optional[tuple[int, int, int]] = None     # (tex_id, w, h)
    has_variable_size: bool = False

    ref_panel: Optional['Panel'] = None
    rule: Optional[PositionRule] = None
    offset_to_ref: int = 0

    def set_constraint(self, ref_panel: 'Panel', rule: PositionRule, gap: int = 10):
        self.ref_panel = ref_panel
        self.rule = rule
        self.gap = gap
        return self

    def update_size(self, new_w: int, new_h: int):
        if self.has_variable_size:
            self.viewport.dynw = new_w
            self.viewport.dynh = new_h
        else:
            self.viewport.w = new_w
            self.viewport.h = new_h
        return self
    
    def calculate_pos(self):
        if not self.ref_panel or not self.rule:
            return

        ref_vp = self.ref_panel.viewport
        my_vp = self.viewport
        ref_w = ref_vp.dynw or ref_vp.w
        ref_h = ref_vp.dynh or ref_vp.h
        my_w = my_vp.dynw or my_vp.w
        my_h = my_vp.dynh or my_vp.h

        if self.rule == PositionRule.RIGHT_OF:
            my_vp.x = ref_vp.x + ref_w + self.gap
            my_vp.y = ref_vp.y
        elif self.rule == PositionRule.LEFT_OF:
            my_vp.x = ref_vp.x - my_w - self.gap
            my_vp.y = ref_vp.y
        elif self.rule == PositionRule.BELOW:
            my_vp.x = ref_vp.x
            my_vp.y = ref_vp.y - my_h - self.gap
        elif self.rule == PositionRule.ABOVE:
            my_vp.x = ref_vp.x
            my_vp.y = ref_vp.y + ref_h + self.gap
        elif self.rule == PositionRule.CENTER_IN:
            my_vp.x = int(ref_vp.x + (ref_w - my_w) // 2)
            my_vp.y = int(ref_vp.y + (ref_h - my_h) // 2)
        elif self.rule == PositionRule.ABOVE_BOTTOM_LEFT_OF:
            my_vp.x = int(ref_vp.x)
            my_vp.y = int(ref_vp.y)

@dataclass
class WindowDef:
    window_size: tuple[int, int]
    frame_size: tuple[int, int]
    gap: int
    panel_titles: list[str]
    progress_bar_size_rates: tuple[float, float]

class WindowGeometryHelper:
    @staticmethod
    def build_layout(wdef: WindowDef) -> dict:
        panels: dict[int, Panel] = {}
        win_width, win_height = wdef.window_size
        viewports_dict = __class__.compute_viewports(wdef)
        if wdef.panel_titles and len(wdef.panel_titles) > 0:
            for i, (title, vp) in enumerate(zip(wdef.panel_titles, viewports_dict["panel_viewports"])):
                p = Panel(panel_id=i, panel_name=(title or f"dynamic_panel_{i}"), title=title, viewport=vp, has_variable_size=False)
                panels[i] = p

        midi_events_y = win_height - wdef.gap // 2
        panels[ProcessingState.IDX_MIDI_EVENTS] = Panel(
            panel_id=ProcessingState.IDX_MIDI_EVENTS,
            panel_name="midi_events",
            title=None,
            viewport=Viewport(
                wdef.gap,
                midi_events_y,
                None,
                None,
            ),
            has_variable_size=True,
        )
        active_notes_h = int(win_height * 0.3)
        active_notes_w = int(active_notes_h * 1)
        active_notes_panel = Panel(
            panel_id=ProcessingState.IDX_ACTIVE_NOTES,
            panel_name="active_notes",
            title="Active Notes Transition",
            title_pos=TitlePosition.INSIDE_TOP_LEFT,
            viewport=Viewport(
                None,
                None,
                active_notes_w,
                active_notes_h,
            ),
            has_variable_size=True,
        )
        if viewports_dict["rows"] == 1 and (0 in panels):
            active_notes_panel.set_constraint(panels[0], PositionRule.ABOVE_BOTTOM_LEFT_OF, gap=0)
        else:
            active_notes_panel.set_constraint(panels[ProcessingState.IDX_MIDI_EVENTS], PositionRule.BELOW, gap=0)
        panels[ProcessingState.IDX_ACTIVE_NOTES] = active_notes_panel

        pbar_viewport = viewports_dict["pbar_viewport"]
        info_y = win_height - wdef.gap // 2
        panels[ProcessingState.IDX_INFO] = Panel(
            panel_id=ProcessingState.IDX_INFO,
            panel_name="info",
            title=None,
            viewport=Viewport(
                pbar_viewport.right + wdef.gap,
                info_y,
                None,
                None,
            ),
            has_variable_size=True,
        )

        return {
            "panels": panels,
            **viewports_dict,
        }

    @staticmethod
    def compute_viewports(wdef: WindowDef) -> dict:
        window_width, window_height = wdef.window_size
        frame_width, frame_height = wdef.frame_size

        pbar_width_rate, pbar_height_rate = wdef.progress_bar_size_rates
        pbar_width, pbar_height = int(np.ceil(pbar_width_rate * window_width)), int(np.ceil(pbar_height_rate * window_height))
        pbar_x = (window_width - pbar_width) // 2
        pbar_y = window_height - pbar_height
        pbar_viewport = Viewport(pbar_x, pbar_y, pbar_width, pbar_height)

        count = len(wdef.panel_titles) if wdef.panel_titles else 0
        if count <= 0:
            return {
                "panel_viewports": [],
                "pbar_viewport": pbar_viewport,
                "cell_size": None,
                "rows": 0,
                "cols": 0,
            }

        content_x = 0
        content_y = 0
        content_width = window_width - content_x
        content_height = window_height - content_y - pbar_height

        # grid shape: almost square
        cols = np.ceil(np.sqrt(count))
        rows = np.ceil(count / cols)
        remaining = cols * rows - count
        top_cols = cols - remaining  # first row may have fewer cells

        # usable space (inside outer gaps) for scaling the whole grid
        avail_w = max(1, content_width - (cols + 1) * wdef.gap)
        avail_h = max(1, content_height - (rows + 1) * wdef.gap)

        # cell count per row: first row uses top_cols, others use full cols
        row_counts = np.array([top_cols] + [cols] * int(rows - 1), dtype=int)
        max_cols = row_counts.max()

        # scale frame size to fit the whole grid (keep aspect ratio)
        scale = np.minimum(
            avail_w / (max_cols * frame_width),
            avail_h / (rows * frame_height),
        )

        cell_w = max(int(frame_width * scale), 1)
        cell_h = max(int(frame_height * scale), 1)

        # --- build slots with numpy ---
        idx = np.arange(count)
        row_bounds = row_counts.cumsum()
        row_idx = np.searchsorted(row_bounds, idx, side="right")
        prev_bounds = np.concatenate(([0], row_bounds[:-1]))
        col_idx = idx - prev_bounds[row_idx]

        # per-row horizontal centering
        group_w = row_counts * cell_w + (row_counts - 1) * wdef.gap
        base_x_rows = np.maximum(wdef.gap, wdef.gap + (avail_w - group_w) // 2)

        # vertical centering of the whole grid
        inner_h = max(1, content_height - 2 * wdef.gap)
        grid_h = rows * cell_h + (rows - 1) * wdef.gap
        extra_h = max(0, inner_h - grid_h)
        bottom_y0 = wdef.gap + extra_h // 2  # bottom of the bottom row

        row_from_bottom = rows - 1 - np.arange(rows)
        base_y_rows = bottom_y0 + row_from_bottom * (cell_h + wdef.gap)

        x = content_x + base_x_rows[row_idx] + col_idx * (cell_w + wdef.gap)
        y = content_y + base_y_rows[row_idx]

        return {
            "panel_viewports": [
                Viewport(int(cell_x), int(cell_y), int(cell_w), int(cell_h))
                for cell_x, cell_y in zip(x, y)
            ],
            "pbar_viewport": pbar_viewport,
            "cell_size": (cell_w, cell_h),
            "cols": cols,
            "rows": rows,
        }
