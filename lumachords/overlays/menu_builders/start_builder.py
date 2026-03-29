from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
import webbrowser

import pygame_menu

from lumachords.utils import Utils
from lumachords.runtime_config import AppMode
from lumachords.midi_rt import MidiRt
from lumachords.profile_store import ProfileStore
from lumachords.ui_types import UICommand
from lumachords.video import YoutubeDownloadService

from lumachords.gui.file_dialog_utils import FileDialogUtils
from .common import CARD_BORDER, CARD_TITLE, MUTED, add_button, card_width, create_card, muted_font_size, pack_card_label

if TYPE_CHECKING:  # pragma: no cover
    from lumachords.overlays.menu_overlay import MenuOverlay


def build_menu_start(menu_overlay: "MenuOverlay", menu: pygame_menu.Menu) -> None:
    MenuStartBuilder(menu_overlay, menu).build()


class MenuStartBuilder:
    TRANSPOSE_OCTAVE_OPTIONS = [None, -2, -1, 0, 1, 2]
    ABOUT_GITHUB_URL = "https://github.com/adalkiran"
    ABOUT_LINKEDIN_URL = "https://www.linkedin.com/in/alper-dalkiran/"
    LICENSE_PAGE_LINES = 180

    def __init__(self, menu_overlay: "MenuOverlay", menu: pygame_menu.Menu):
        self.menu_overlay = menu_overlay
        self.menu = menu
        self.file_input_row = None
        self.youtube_input_widget = None
        self.youtube_help_label = None
        self.transpose_octaves_widget = None
        self.split_note_name_widget = None
        self.split_note_oct_widget = None
        self.midirt_velocity_widget = None
        self.midirt_velocity_name_label = None
        self.midirt_use_pedal_widget = None
        self.midi_rt_pref_row = None
        self.last_loaded_profile_key = None

    @staticmethod
    def open_url(url: str) -> None:
        webbrowser.open_new_tab(url)

    def _path_label_max_len(self) -> int:
        if self.menu_overlay.menu:
            return max(24, int(self.menu_overlay.menu.get_width() * 0.12))
        return 40

    def format_path_label(self, path: str, max_len: int | None = None) -> str:
        if not path:
            return "Not set"
        max_len = self._path_label_max_len() if max_len is None else max_len
        if len(path) <= max_len:
            return path
        keep = max_len - 3
        return f"...{path[-keep:]}" if keep > 0 else "..."

    def _muted_font_size(self) -> int:
        return muted_font_size(self.menu)

    def _card_width(self) -> int:
        return card_width(self.menu, min_width=280, horizontal_padding=26)

    def _create_card(self, title: str, body: str | None, height: int) -> pygame_menu.widgets.Frame:
        return create_card(self.menu, title, body, height, min_width=280, horizontal_padding=26)

    def _pack_card_label(self, card, text: str, *, color=None, size: int | None = None, margin: int = 0):
        return pack_card_label(self.menu, card, text, color=color, size=size)

    def _make_button(self, title: str, action, **kwargs):
        return add_button(self.menu, title, action, **kwargs)

    # -------------- START MENU --------------

    def build(self) -> None:
        self._build_header()
        self._build_video_backend_row()
        self._build_input_card()
        self._build_detection_card()
        self._build_mode_card()
        self._build_midi_card()
        self.menu.add.vertical_margin(20)
        self._make_button(
            "Start Session",
            self.menu_overlay.wrap_action(
                self.on_start_button,
                immediate=True,
                toggle_menu=True,
            ),
        )

    def apply_input_source_visibility(self) -> None:
        use_youtube = self.menu_overlay.settings.input_source == "youtube"
        if self.file_input_row:
            self.file_input_row.hide() if use_youtube else self.file_input_row.show()
        if self.youtube_input_widget:
            self.youtube_input_widget.show() if use_youtube else self.youtube_input_widget.hide()
        if self.youtube_help_label:
            self.youtube_help_label.show() if use_youtube else self.youtube_help_label.hide()

    def _set_default_profile_widgets(self) -> None:
        if self.transpose_octaves_widget:
            try:
                self.transpose_octaves_widget.set_value(self._transpose_octaves_index(None))
            except Exception:
                pass
        if self.split_note_name_widget:
            try:
                self.split_note_name_widget.set_value(Utils.NOTE_ALL_NAMES.index("C"))
            except Exception:
                pass
        if self.split_note_oct_widget:
            try:
                self.split_note_oct_widget.set_value(Utils.NOTE_ALL_OCTAVES.index("4"))
            except Exception:
                pass

    def load_input_profile(self, input_source: str, input_value: str) -> None:
        source = (input_source or "").strip().lower()
        value = (input_value or "").strip()
        if not source or not value:
            return
        input_key = f"{source}:{value}"
        if input_key == self.last_loaded_profile_key:
            return
        profile = ProfileStore.get_input_profile(source, value)
        if not profile:
            self._set_default_profile_widgets()
            self.last_loaded_profile_key = input_key
            return
        transpose_octaves = profile.get("transpose_octaves")
        split_note = profile.get("split_note")
        if transpose_octaves is not None and self.transpose_octaves_widget:
            try:
                transpose_value = int(transpose_octaves)
                if transpose_value < -2 or transpose_value > 2:
                    transpose_value = 0
                self.transpose_octaves_widget.set_value(self._transpose_octaves_index(transpose_value))
            except Exception:
                pass
        if split_note:
            split_name, split_oct = Utils.split_note_name_to_parts(str(split_note))
            if self.split_note_name_widget:
                try:
                    self.split_note_name_widget.set_value(Utils.NOTE_ALL_NAMES.index(split_name))
                except Exception:
                    pass
            if self.split_note_oct_widget:
                try:
                    self.split_note_oct_widget.set_value(Utils.NOTE_ALL_OCTAVES.index(split_oct))
                except Exception:
                    pass
        self.last_loaded_profile_key = input_key

    def open_input_file_dialog(self) -> None:
        filename = FileDialogUtils.show_open_dialog("Select Input Video", FileDialogUtils.FILTER_VIDEO_FILES)
        if not filename:
            return
        if self.menu_overlay.input_path_label:
            self.menu_overlay.input_path_label.set_title(f"Selected: {self.format_path_label(filename)}")
        self.menu_overlay.settings.input_video_path = filename
        self.menu_overlay.settings.input_source = "file"
        self.load_input_profile("file", filename)
        self.menu_overlay.force_render = True

    # -------------- START MENU ACTION EVENTS --------------

    def on_input_source_change(self, value: bool) -> None:
        self.menu_overlay.settings.input_source = "youtube" if value else "file"
        if self.menu_overlay.settings.input_source == "youtube":
            self.load_input_profile("youtube", self.menu_overlay.settings.youtube_input)
        else:
            self.load_input_profile("file", self.menu_overlay.settings.input_video_path)
        self.apply_input_source_visibility()
        self.menu_overlay.force_render = True

    def on_youtube_input_change(self, value: str) -> None:
        self.menu_overlay.settings.youtube_input = value.strip()
        self.load_input_profile("youtube", self.menu_overlay.settings.youtube_input)
        self.menu_overlay.force_render = True

    def on_transpose_octaves_change(self, value: int | None) -> None:
        if value is None:
            self.menu_overlay.settings.transpose_octaves = None
            return
        try:
            transpose_value = int(value)
            if transpose_value < -2 or transpose_value > 2:
                return
            self.menu_overlay.settings.transpose_octaves = transpose_value
        except (TypeError, ValueError):
            pass

    def _transpose_octaves_index(self, value: int | None) -> int:
        if value is None:
            return 0
        try:
            return self.TRANSPOSE_OCTAVE_OPTIONS.index(int(value))
        except Exception:
            return 0

    def on_split_note_change(self, value: str) -> None:
        try:
            self.menu_overlay.settings.split_note = value if value else "C4"
            self.menu_overlay.settings.split_midi_num = Utils.parse_split_note_to_midi_num(self.menu_overlay.settings.split_note) or 60
        except ValueError:
            pass

    def on_start_paused_change(self, value: bool) -> None:
        self.menu_overlay.settings.start_paused = value

    def on_auto_timing_change(self, value: bool) -> None:
        self.menu_overlay.settings.auto_timing = value

    def on_midirt_option_change(self, value) -> None:
        self.menu_overlay.settings.midirt_option = value
        self._set_midi_rt_pref_enabled()

    @staticmethod
    def _velocity_name(velocity: int) -> str:
        if velocity < 40:
            return "pianissimo (near silent)"
        if velocity < 55:
            return "piano (soft)"
        if velocity < 75:
            return "mezzo piano (light)"
        if velocity < 95:
            return "mezzo forte (balanced)"
        if velocity < 112:
            return "forte (strong)"
        return "fortissimo (very loud)"

    def _read_velocity_widget(self) -> int:
        if not self.midirt_velocity_widget:
            return int(self.menu_overlay.settings.midirt_velocity or 30)
        value = self.midirt_velocity_widget.get_value()
        if isinstance(value, (list, tuple)):
            value = value[0]
        try:
            value = int(round(float(value)))
        except Exception:
            value = MidiRt.DEFAULT_MIDIRT_VELOCITY
        return max(20, min(127, value))

    def on_midirt_velocity_change(self, *args) -> None:
        value = args[0] if args else self._read_velocity_widget()
        try:
            velocity = int(round(float(value)))
        except Exception:
            velocity = self._read_velocity_widget()
        if self.midirt_velocity_name_label:
            self.midirt_velocity_name_label.set_title(self._velocity_name(velocity))

    def _set_midi_rt_pref_enabled(self) -> None:
        option = self.menu_overlay.settings.midirt_option
        backend = getattr(option, "backend", None) or getattr(option, "backed", None) or "dummy"
        enabled = backend != "dummy"
        for widget in self.midi_rt_pref_row.get_widgets():
            if widget:
                try:
                    widget.readonly = not enabled
                    widget.is_selectable = enabled
                except Exception:
                    pass

    def on_start_button(self) -> None:
        from lumachords.overlays.overlay_manager import OverlayManager
        ProfileStore.set_global_mode("advanced" if self.menu_overlay.settings.app_mode == AppMode.GUI_ADVANCED else "basic")
        self.menu_overlay.settings.midirt_velocity = self._read_velocity_widget()
        self.menu_overlay.settings.midirt_use_pedal = bool(self.midirt_use_pedal_widget.get_value())
        ProfileStore.set_midirt_device(self.menu_overlay.settings.midirt_option.backend,  self.menu_overlay.settings.midirt_option.title)
        if self.menu_overlay.settings.midirt_option.backend != "dummy":
            ProfileStore.set_midirt_velocity(self.menu_overlay.settings.midirt_velocity)
            ProfileStore.set_midirt_use_pedal(self.menu_overlay.settings.midirt_use_pedal)

        youtube_input = (self.menu_overlay.settings.youtube_input or "").strip()
        use_youtube_source = self.menu_overlay.settings.input_source == "youtube"
        if use_youtube_source:
            if not youtube_input:
                OverlayManager.show_toast("YouTube URL or video ID must be specified.")
                return False
            try:
                YoutubeDownloadService.normalize_youtube_input(youtube_input)
            except Exception as e:
                OverlayManager.show_error(e)
                return False
            ProfileStore.set_input_profile("youtube", youtube_input, self.menu_overlay.settings.transpose_octaves, self.menu_overlay.settings.split_note)
            self.menu_overlay.settings.input_source = "youtube"
            self.menu_overlay.command_callback_fn(UICommand.DOWNLOAD_YOUTUBE)
            return False
        if not self.menu_overlay.settings.input_video_path:
            OverlayManager.show_toast("Input video file must be selected.")
            return False
        if not Path(self.menu_overlay.settings.input_video_path).is_file():
            OverlayManager.show_error(f"Input video file does not exist:\n{self.menu_overlay.settings.input_video_path}")
            return False
        ProfileStore.set_input_profile("file", self.menu_overlay.settings.input_video_path, self.menu_overlay.settings.transpose_octaves, self.menu_overlay.settings.split_note)
        self.menu_overlay.command_callback_fn(UICommand.START)

    # -------------- START MENU WIDGET BUILDER METHODS --------------

    def _build_video_backend_row(self) -> None:
        row = self.menu.add.frame_h(self._card_width(), 28, margin=(0, 4))
        row._relax = True
        row._pack_margin_warning = False
        row.pack(
            self.menu.add.label("Video backend", font_color=MUTED, align=pygame_menu.locals.ALIGN_LEFT, margin=(0, 0)),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        row.pack(
            self.menu.add.label(
                self.menu_overlay.settings.get_video_backend_title(),
                font_color=CARD_TITLE,
                align=pygame_menu.locals.ALIGN_RIGHT,
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_RIGHT,
        )
        self.menu.add.vertical_margin(14)

    def _build_header(self) -> None:
        header = self.menu.add.frame_h(self._card_width(), 44, margin=(0, 0))
        header._relax = True
        header._pack_margin_warning = False
        header.pack(
            self.menu.add.label("Start A Session", font_color=CARD_TITLE, align=pygame_menu.locals.ALIGN_LEFT, margin=(0, 0)),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        header.pack(
            self._make_button(
                "About",
                self.menu_overlay.wrap_action(self.open_about_menu, immediate=False, toggle_menu=False),
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_RIGHT,
        )
        self.menu.add.label(
            "Choose a source, review the interpretation settings, then start.",
            max_char=70,
            font_color=MUTED,
            align=pygame_menu.locals.ALIGN_LEFT,
            font_size=self._muted_font_size(),
            margin=(0, 0),
        )
        self.menu.add.vertical_margin(8)

    def _build_input_card(self) -> None:
        card = self._create_card(
            "Source",
            "Pick a local video file or paste a YouTube link as a piano tutorial source video.",
            236,
        )

        source_toggle = self.menu.add.toggle_switch(
            "Source Type",
            default=(self.menu_overlay.settings.input_source == "youtube"),
            state_text=("FILE", "YOUTUBE"),
            onchange=self.menu_overlay.wrap_action(self.on_input_source_change, immediate=False, toggle_menu=False),
            toggleswitch_id="input_source",
            margin=(0, 0),
        )
        card.pack(source_toggle, align=pygame_menu.locals.ALIGN_LEFT)

        path_label_max_len = max(24, int(self.menu.get_width() * 0.12))
        self.file_input_row = self.menu.add.frame_v(self._card_width() - 28, 62, margin=(0, 0))
        self.file_input_row._relax = True
        self.file_input_row._pack_margin_warning = False
        file_action_row = self.menu.add.frame_h(self._card_width() - 28, 36, margin=(0, 0))
        file_action_row._relax = True
        file_action_row._pack_margin_warning = False
        file_action_row.pack(self.menu.add.label("Video file", font_color=CARD_TITLE, margin=(0, 0)), align=pygame_menu.locals.ALIGN_LEFT)
        file_action_row.pack(
            self._make_button(
                "Choose File...",
                self.menu_overlay.wrap_action(self.open_input_file_dialog, immediate=False, toggle_menu=False),
                margin=(20, 0),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        self.file_input_row.pack(file_action_row, align=pygame_menu.locals.ALIGN_LEFT)
        self.file_input_row.pack(self.menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_LEFT)
        self.file_input_row.pack(
            self.menu.add.label(
                f"Selected: {self.format_path_label(self.menu_overlay.settings.input_video_path, path_label_max_len)}",
                max_char=max(30, path_label_max_len + 10),
                font_color=MUTED,
                align=pygame_menu.locals.ALIGN_LEFT,
                font_size=self._muted_font_size(),
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        self.menu_overlay.input_path_label = self.file_input_row.get_widgets()[-1]
        card.pack(self.file_input_row, align=pygame_menu.locals.ALIGN_LEFT)

        self.youtube_input_widget = self.menu.add.text_input(
            "YouTube Link Or Video ID here: ",
            default=self.menu_overlay.settings.youtube_input or "",
            maxchar=300,
            copy_paste_enable=True,
            repeat_keys=False,
            onchange=self.menu_overlay.wrap_action(self.on_youtube_input_change, immediate=False, toggle_menu=False),
            textinput_id="youtube_input",
            margin=(0, 0),
        )
        card.pack(self.youtube_input_widget, align=pygame_menu.locals.ALIGN_LEFT)
        self.youtube_help_label = self.menu.add.label(
            "Paste a full YouTube link or only the video ID.",
            max_char=70,
            font_color=MUTED,
            align=pygame_menu.locals.ALIGN_LEFT,
            font_size=self._muted_font_size(),
            margin=(0, 0),
        )
        card.pack(self.youtube_help_label, align=pygame_menu.locals.ALIGN_LEFT)
        self.apply_input_source_visibility()

    def _build_detection_card(self) -> None:
        card = self._create_card(
            "Detection",
            "These options change how LumaChords interprets the played notes from the video.",
            350,
        )
        transpose_options = [("Auto", None)] + [(str(v), v) for v in self.TRANSPOSE_OCTAVE_OPTIONS if v is not None]
        transpose_row = self.menu.add.frame_h(self._card_width() - 28, 40, margin=(0, 0))
        transpose_row._relax = True
        transpose_row._pack_margin_warning = False
        transpose_row.pack(self.menu.add.label("Transpose Octaves", font_color=CARD_TITLE, margin=(0, 0)), align=pygame_menu.locals.ALIGN_LEFT)
        self.transpose_octaves_widget = self.menu.add.dropselect(
            "",
            transpose_options,
            default=self._transpose_octaves_index(self.menu_overlay.settings.transpose_octaves),
            selection_option_font_color=CARD_TITLE,
            selection_box_width=int(self.menu.get_width() * 0.15),
            onchange=(lambda _, val: self.menu_overlay.wrap_action(self.on_transpose_octaves_change, immediate=False, toggle_menu=False)(val)),
            dropselect_id="transpose_octaves",
        )
        transpose_row.pack(self.transpose_octaves_widget, align=pygame_menu.locals.ALIGN_LEFT)
        card.pack(transpose_row, align=pygame_menu.locals.ALIGN_LEFT)

        split_name, split_oct = Utils.split_note_name_to_parts(self.menu_overlay.settings.split_note)
        row_width = max(240, self._card_width() - 28)
        row = self.menu.add.frame_h(row_width, 42, margin=(0, 0))
        row._relax = True
        row._pack_margin_warning = False
        row.pack(self.menu.add.label("Default Split MIDI Note", font_color=CARD_TITLE, margin=(0, 0)), align=pygame_menu.locals.ALIGN_LEFT)

        self.split_note_name_widget = row.pack(
            self.menu.add.dropselect(
                "",
                [(n, n) for n in Utils.NOTE_ALL_NAMES],
                default=Utils.NOTE_ALL_NAMES.index(split_name),
                selection_option_font_color=CARD_TITLE,
                dropselect_id="split_note_name",
                selection_box_width=int(row.get_width() * 0.24),
                onchange=(lambda _, val: self.menu_overlay.wrap_action(self.on_split_note_change, immediate=False, toggle_menu=False)(f"{val}{self.split_note_oct_widget.get_value()[0][1]}")),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )

        self.split_note_oct_widget = row.pack(
            self.menu.add.dropselect(
                "",
                [(o, o) for o in Utils.NOTE_ALL_OCTAVES],
                default=Utils.NOTE_ALL_OCTAVES.index(split_oct),
                selection_option_font_color=CARD_TITLE,
                dropselect_id="split_note_oct",
                selection_box_width=int(row.get_width() * 0.2),
                onchange=(lambda _, val: self.menu_overlay.wrap_action(self.on_split_note_change, immediate=False, toggle_menu=False)(f"{self.split_note_name_widget.get_value()[0][1]}{val}")),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        card.pack(row, align=pygame_menu.locals.ALIGN_LEFT)
        self._pack_card_label(card, "The default split note will be used when the hands/colors not detected from the video.")
        auto_timing = self.menu.add.toggle_switch(
            "Auto Timing (experimental)",
            default=False,
            onchange=self.menu_overlay.wrap_action(self.on_auto_timing_change, immediate=False, toggle_menu=False),
            toggleswitch_id="auto_timing",
            margin=(0, 0),
        )
        card.pack(auto_timing, align=pygame_menu.locals.ALIGN_LEFT)
        self._pack_card_label(card, "When enabled, tempo and time signature are inferred automatically from note flow, but the results may be inaccurate.")

        if self.menu_overlay.settings.input_source == "youtube":
            self.load_input_profile("youtube", self.menu_overlay.settings.youtube_input)
        else:
            self.load_input_profile("file", self.menu_overlay.settings.input_video_path)

    def _build_mode_card(self) -> None:
        card = self._create_card(
            "Playback And View",
            "Session behavior controls. App Mode changes the live analysis view during processing.",
            280,
        )
        start_paused = self.menu.add.toggle_switch(
            "Start Paused",
            default=self.menu_overlay.settings.start_paused,
            onchange=self.menu_overlay.wrap_action(self.on_start_paused_change, immediate=False, toggle_menu=False),
            toggleswitch_id="start_paused",
            margin=(0, 0),
        )
        card.pack(start_paused, align=pygame_menu.locals.ALIGN_LEFT)
        app_mode = self.menu.add.toggle_switch(
            "App Mode",
            default=(self.menu_overlay.settings.app_mode == AppMode.GUI_ADVANCED),
            state_text=("BASIC", "ADVANCED"),
            onchange=self.menu_overlay.wrap_action(
                self.menu_overlay.toggle_gui_mode,
                immediate=False,
                toggle_menu=False,
            ),
            margin=(0, 0),
        )
        card.pack(app_mode, align=pygame_menu.locals.ALIGN_LEFT)
        self._pack_card_label(card, "BASIC shows only the main output panel. ADVANCED keeps extra analysis panels visible.")

    def _build_midi_card(self) -> None:
        card = self._create_card(
            "MIDI Output",
            "Choose where detected notes are sent. Velocity and pedal only apply when a MIDI output is selected.",
            290,
        )
        midi_rt_options = MidiRt.create_backend_options()
        midi_rt_dropselect_items = [(option.title, option) for option in midi_rt_options]
        default_midirt_option_idx = 0
        if self.menu_overlay.settings.midirt_option:
            opt = self.menu_overlay.settings.midirt_option
            for i, (_, item) in enumerate(midi_rt_dropselect_items):
                if item.backend == opt.backend and (opt.backend != "mido" or item.title == opt.title):
                    default_midirt_option_idx = i
                    break

        midi_rt_device_row = self.menu.add.frame_h(self._card_width() - 28, 40, margin=(0, 0))
        midi_rt_device_row._relax = True
        midi_rt_device_row._pack_margin_warning = False
        midi_rt_device_row.pack(self.menu.add.label("Output Device", font_color=CARD_TITLE, margin=(0, 0)), align=pygame_menu.locals.ALIGN_LEFT)
        midi_rt_device_dropdown = self.menu.add.dropselect(
            "",
            midi_rt_dropselect_items,
            default=default_midirt_option_idx,
            selection_option_font_color=CARD_TITLE,
            dropselect_id="midirt_option",
            selection_box_width=int(midi_rt_device_row.get_width() * 0.5),
            onchange=(lambda _, val: self.menu_overlay.wrap_action(self.on_midirt_option_change, immediate=False, toggle_menu=False)(val)),
            margin=(0, 0),
        )
        midi_rt_device_row.pack(midi_rt_device_dropdown, align=pygame_menu.locals.ALIGN_LEFT)
        card.pack(midi_rt_device_row, align=pygame_menu.locals.ALIGN_LEFT)

        self.midi_rt_pref_row = self.menu.add.frame_h(self._card_width() - 28, 58, margin=(0, 0))
        self.midi_rt_pref_row._relax = True
        self.midi_rt_pref_row._pack_margin_warning = False

        midi_rt_pref_left = self.menu.add.frame_v(int(self.midi_rt_pref_row.get_width() * 0.52), 58, margin=(0, 0))
        midi_rt_pref_left._relax = True
        midi_rt_pref_left._pack_margin_warning = False

        velocity_row = self.menu.add.frame_h(int(midi_rt_pref_left.get_width() * 0.9), 34, margin=(0, 0))
        velocity_row._relax = True
        velocity_row._pack_margin_warning = False
        self.midirt_velocity_widget = velocity_row.pack(
            self.menu.add.range_slider(
                "Velocity",
                default=max(20, min(127, int(self.menu_overlay.settings.midirt_velocity or MidiRt.DEFAULT_MIDIRT_VELOCITY))),
                range_values=(20, 127),
                increment=1,
                value_format=lambda x: str(int(x)),
                rangeslider_id="midirt_velocity",
                onchange=self.menu_overlay.wrap_action(self.on_midirt_velocity_change, immediate=False, toggle_menu=False),
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        midi_rt_pref_left.pack(velocity_row, align=pygame_menu.locals.ALIGN_LEFT)
        self.midirt_velocity_name_label = midi_rt_pref_left.pack(
            self.menu.add.label(
                self._velocity_name(self._read_velocity_widget()),
                font_color=MUTED,
                selectable=False,
            ),
            align=pygame_menu.locals.ALIGN_CENTER,
        )
        self.midirt_velocity_name_label.is_selectable = False

        self.midi_rt_pref_row.pack(midi_rt_pref_left, align=pygame_menu.locals.ALIGN_LEFT)
        self.midirt_use_pedal_widget = self.midi_rt_pref_row.pack(
            self.menu.add.toggle_switch(
                "Use pedal",
                default=bool(self.menu_overlay.settings.midirt_use_pedal),
                state_text=("OFF", "ON"),
                toggleswitch_id="midirt_use_pedal",
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        card.pack(self.midi_rt_pref_row, align=pygame_menu.locals.ALIGN_LEFT)

        midi_rt_device_dropdown._onchange(None, midi_rt_dropselect_items[default_midirt_option_idx][1])

    # -------------- ABOUT BOX --------------

    def open_about_menu(self) -> None:
        about_width = int(self.menu.get_width() * 0.7)
        about_height = int(self.menu.get_height() * 0.6)
        about_menu, _ = self.menu_overlay.create_empty_menu("About", about_width, about_height)
        about_menu.add.vertical_margin(int(about_height * 0.08))
        about_menu.add.label("Developed by")
        about_menu.add.label("Adil Alper DALKIRAN")
        about_menu.add.vertical_margin(12)
        linkedin_btn = about_menu.add.button("LinkedIn", self.menu_overlay.wrap_action(lambda: self.open_url(self.ABOUT_LINKEDIN_URL), immediate=False, toggle_menu=False))
        linkedin_btn.set_border(1, CARD_BORDER)
        github_btn = about_menu.add.button("GitHub", self.menu_overlay.wrap_action(lambda: self.open_url(self.ABOUT_GITHUB_URL), immediate=False, toggle_menu=False))
        github_btn.set_border(1, CARD_BORDER)
        about_menu.add.vertical_margin(12)
        licenses_btn = about_menu.add.button(
            "Open Source Licenses...",
            self.menu_overlay.wrap_action(self.open_licenses_menu, immediate=False, toggle_menu=False),
        )
        licenses_btn.set_border(1, CARD_BORDER)
        about_menu.add.vertical_margin(8)
        about_menu.add.label("LumaChords is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.", max_char=70, font_color="white", align=pygame_menu.locals.ALIGN_LEFT, font_size=10)
        about_menu.add.vertical_margin(8)
        close_btn = about_menu.add.button("Close", pygame_menu.events.BACK)
        close_btn.set_border(1, CARD_BORDER)
        self.menu._open(about_menu)

    # -------------- LICENSES BOX --------------

    def open_licenses_menu(self, page_index: int = 0) -> None:
        pages, source_path = self._read_license_notice_pages()
        page_index = max(0, min(page_index, len(pages) - 1))
        self.menu._open(self._create_licenses_menu(page_index, pages, source_path))

    def _create_licenses_menu(self, page_index: int, pages: list[str], source_path: str) -> pygame_menu.Menu:
        licenses_width = int(self.menu.get_width() * 0.9)
        licenses_height = int(self.menu.get_height() * 0.85)
        licenses_menu, _ = self.menu_overlay.create_empty_menu("Open Source Licenses", licenses_width, licenses_height)

        text_view_width = max(licenses_width - 30, 280)
        text_view_height = max(licenses_height - 150, 180)
        text_max_chars = max(40, int(text_view_width / 9))
        content = pages[page_index]
        line_count = max(1, content.count("\n") + 1)
        estimated_row_height = 24
        content_height = max(text_view_height, line_count * estimated_row_height)

        licenses_menu.add.label(f"Page {page_index + 1}/{len(pages)}")
        licenses_menu.add.label(self.format_path_label(source_path, max_len=max(30, int(licenses_width * 0.09))), max_char=max(30, int(licenses_width * 0.09)))

        licenses_frame = licenses_menu.add.frame_v(
            text_view_width,
            content_height,
            max_height=text_view_height,
            margin=(0, 0),
            scrollbars=pygame_menu.locals.POSITION_EAST,
        )
        licenses_frame._relax = True

        license_label = licenses_menu.add.label(
            content,
            max_char=text_max_chars,
            wordwrap=True,
            selectable=False,
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        if isinstance(license_label, list):
            for label_item in license_label:
                licenses_frame.pack(label_item, align=pygame_menu.locals.ALIGN_LEFT)
        else:
            licenses_frame.pack(license_label, align=pygame_menu.locals.ALIGN_LEFT)

        licenses_menu.add.vertical_margin(8)
        nav_row = licenses_menu.add.frame_h(text_view_width, 36, margin=(0, 0))
        nav_row._relax = True
        if page_index > 0:
            prev_btn = licenses_menu.add.button(
                "< Prev",
                self.menu_overlay.wrap_action(
                    lambda: self.open_licenses_menu(page_index - 1),
                    immediate=False,
                    toggle_menu=False,
                ),
            )
            prev_btn.set_border(1, CARD_BORDER)
            nav_row.pack(
                prev_btn,
                align=pygame_menu.locals.ALIGN_LEFT,
            )
        else:
            nav_row.pack(licenses_menu.add.label(""), align=pygame_menu.locals.ALIGN_LEFT)
        if page_index < len(pages) - 1:
            next_btn = licenses_menu.add.button(
                "Next >",
                self.menu_overlay.wrap_action(
                    lambda: self.open_licenses_menu(page_index + 1),
                    immediate=False,
                    toggle_menu=False,
                ),
            )
            next_btn.set_border(1, CARD_BORDER)
            nav_row.pack(
                next_btn,
                align=pygame_menu.locals.ALIGN_RIGHT,
            )
        else:
            nav_row.pack(licenses_menu.add.label(""), align=pygame_menu.locals.ALIGN_RIGHT)

        licenses_menu.add.vertical_margin(4)
        close_btn = licenses_menu.add.button("Close", pygame_menu.events.BACK)
        close_btn.set_border(1, CARD_BORDER)
        return licenses_menu

    @staticmethod
    def _read_license_notice_pages() -> tuple[list[str], str]:
        notice_paths = [
            Path(Utils.resource_path("THIRD_PARTY_LICENSES.txt", "resources", return_as_str=False)),
            Path("THIRD_PARTY_LICENSES.txt"),
        ]
        for notice_path in notice_paths:
            if notice_path.is_file():
                pages: list[str] = []
                page_lines: list[str] = []
                with notice_path.open("r", encoding="utf-8", errors="replace") as f:
                    for raw_line in f:
                        page_lines.append(raw_line.rstrip("\n"))
                        if len(page_lines) >= MenuStartBuilder.LICENSE_PAGE_LINES:
                            pages.append("\n".join(page_lines))
                            page_lines = []
                if page_lines:
                    pages.append("\n".join(page_lines))
                if not pages:
                    pages = ["(License file is empty)"]
                return pages, str(notice_path)
        return [f"License notice file was not found.\n{notice_paths[0]}"], str(notice_paths[0])
