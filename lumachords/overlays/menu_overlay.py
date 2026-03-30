
from enum import IntEnum
from typing import Callable
import subprocess
import sys

import pygame
import pygame_menu

from lumachords.runtime_config import AppMode, AppSettings
from lumachords.overlays.menu_builders import retry_search_ffmpeg, build_menu_ffmpeg, build_menu_processing, build_menu_start, style_button
from lumachords.ui_types import UICommand

from .base_overlay import BaseOverlay


class MenuType(IntEnum):
    MENU_START = 1
    MENU_PROCESSING = 2
    MENU_FINISH = 3
    MENU_FFMPEG = 4


class MenuOverlay(BaseOverlay):
    MENU_TITLES = {
        MenuType.MENU_START: "Start Session",
        MenuType.MENU_PROCESSING: "Session",
        MenuType.MENU_FINISH: "Session Complete",
        MenuType.MENU_FFMPEG: "FFmpeg Setup"
    }

    def __init__(self, window_width: int, window_height: int, command_callback_fn: Callable[[UICommand, any], any], post_command_callback_fn: Callable, settings: AppSettings):    
        self.menu: pygame_menu.Menu = None
        self.menu_type: MenuType = None
        self.settings = settings
        self.input_path_label = None
        super().__init__(window_width, window_height, pygame_menu.themes.THEME_BLUE)
        self.command_callback_fn = command_callback_fn
        self.post_command_callback_fn = post_command_callback_fn

        self.theme.background_color = (6, 10, 16, 212)
        self.theme.title_background_color = (23, 84, 106, 255)
        self.theme.title_font_color = (241, 247, 255, 255)
        self.theme.widget_alignment = pygame_menu.locals.ALIGN_CENTER
        self.theme.widget_font_color = (234, 240, 255, 255)
        self.theme.readonly_color = (164, 181, 204, 255)
        self.theme.readonly_selected_color = (205, 220, 240, 255)
        self.theme.selection_color = (88, 194, 178, 255)
        self.theme.widget_box_background_color = (18, 24, 36, 255)
        self.theme.widget_box_border_color = (70, 96, 126, 255)
        self.theme.widget_box_arrow_color = (150, 210, 202, 255)
        self.theme.widget_margin = (0, 6)
        self.theme.widget_padding = (5, 10)

        self.confirm_menu: pygame_menu.Menu = None

    def create_empty_menu(self, menu_title: str, menu_width: int, menu_height: int):
        window_width, window_height = self.window_width, self.window_height
        enabled = False if self.menu is None else self.menu.is_enabled()
        menu = pygame_menu.Menu(
            menu_title, 
            menu_width, 
            menu_height,
            theme=self.theme,
            enabled=enabled,
            screen_dimension=(window_width, window_height),
        )
        return menu, enabled

    def _style_button(self, button: pygame_menu.widgets.Button) -> pygame_menu.widgets.Button:
        return style_button(button, border_color=self.theme.widget_box_border_color)

    def build_menu(self, menu_type: MenuType):
        if menu_type is None:
            return
        menu_title = __class__.MENU_TITLES[menu_type]
        window_width, window_height = self.window_width, self.window_height
        self.menu_type = menu_type

        if menu_type == MenuType.MENU_START:
            menu_width = int(window_width * 0.56)
            menu_height = int(min(window_height * 0.94, menu_width * 1.35))
            menu, enabled = self.create_empty_menu(menu_title, menu_width, menu_height)
            build_menu_start(self, menu)
        elif menu_type == MenuType.MENU_FFMPEG:
            menu_width = int(window_width * 0.72)
            menu_height = int(min(window_height * 0.92, menu_width * 1.2))
            menu, enabled = self.create_empty_menu(menu_title, menu_width, menu_height)
            build_menu_ffmpeg(self, menu)
        else:
            menu_width = int(window_width * 0.4)
            menu_height = int(window_height * 0.48)
            menu, enabled = self.create_empty_menu(menu_title, menu_width, menu_height)
            build_menu_processing(self, menu, menu_type)

        confirm_menu = pygame_menu.Menu(
            "Confirm Quit",
            menu_width,
            menu_height,
            theme=self.theme,
            enabled=enabled,
            screen_dimension=(window_width, window_height),
        )
        confirm_menu.add.label("Are you sure?")
        self._style_button(confirm_menu.add.button("Yes", (lambda: pygame.event.post(pygame.event.Event(pygame.QUIT)))))
        self._style_button(confirm_menu.add.button("No", pygame_menu.events.BACK))
        self._style_button(menu.add.button('Quit', confirm_menu))

        self.menu = menu
        self.confirm_menu = confirm_menu

    def resize(self, window_width: int, window_height: int, force: bool=False):
        if not super().resize(window_width, window_height, force=force):
            return
        selected_index = self.menu.get_index() if self.menu else None
        self.build_menu(self.menu_type)
        if selected_index is not None and selected_index >= 0:
            try:
                self.menu.select_widget(self.menu._widgets[selected_index])
            except Exception:
                pass

    def rebuild_and_show(self, menu_type: MenuType):
        if self.menu and self.menu.is_enabled():
            self.toggle()
        self.build_menu(menu_type)
        if not self.menu.is_enabled():
            self.toggle()

    def is_enabled(self) -> bool:
        return self.menu and self.menu.is_enabled()

    def requires_pause(self) -> bool:
        return self.is_enabled() or (self.confirm_menu and self.confirm_menu.is_enabled())

    def toggle(self):
        if self.menu.is_enabled():
            if self.menu_type == MenuType.MENU_FINISH:
                return
            if self.confirm_menu:
                self.confirm_menu.disable()
            self.menu.full_reset()
            self.menu.disable()
        else:
            self.menu.full_reset()
            self.menu.enable()
        self.force_render = True

    def show_confirm_quit(self):
        if not self.confirm_menu:
            return
        if not self.menu.is_enabled():
            self.menu.enable()
        self.menu._open(self.confirm_menu)

    def process_event(self, event) -> bool:
        processed = False
        if self.is_enabled():
            if (
                self.menu_type == MenuType.MENU_START
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_v
                and (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
            ):
                selected_widget = self.menu.get_selected_widget()
                if selected_widget is not None and selected_widget.get_id() == "youtube_input":
                    text = self._read_clipboard_text()
                    if text:
                        current = str(selected_widget.get_value())
                        selected_widget.set_value(current + text)
                        selected_widget.change()
                        return True
            try:
                processed = self.menu.update([event])
            except RecursionError:
                # This try-catch block exists because of the text inputs (added using self.menu.add.text_input) cause max recursion limit exceeded
                # exception when pressed arrow keys on the keyboard while a text input box is focused.
                # Seems like it's a bug of pygame-menu, because it thinks the text input is in text seletction mode, but it's empty.
                processed = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_F1):
                if self.menu_type != MenuType.MENU_START:
                    self.toggle()
                    processed = True
                elif self.confirm_menu.is_enabled():
                    self.confirm_menu.disable()
                    self.menu.full_reset()
                    self.menu.enable()
                    processed = True
            if self.menu_type == MenuType.MENU_FFMPEG and event.type == pygame.WINDOWFOCUSGAINED:
                try:
                    retry_search_ffmpeg(self, show_error=False)
                except Exception:
                    pass
        return processed

    @staticmethod
    def _read_clipboard_text() -> str:
        # Try SDL clipboard first to avoid extra dependencies.
        try:
            if not pygame.scrap.get_init():
                pygame.scrap.init()
            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
            if raw:
                if isinstance(raw, bytes):
                    return raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
                return str(raw).strip()
        except Exception:
            pass

        # Fallback to native clipboard commands to avoid Tk/SDL conflicts.
        try:
            if sys.platform == "darwin":
                cp = subprocess.run(
                    ["pbpaste"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if cp.returncode == 0 and cp.stdout:
                    return cp.stdout.strip()
            elif sys.platform.startswith("linux"):
                for cmd in (["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
                    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if cp.returncode == 0 and cp.stdout:
                        return cp.stdout.strip()
            elif sys.platform.startswith("win"):
                cp = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if cp.returncode == 0 and cp.stdout:
                    return cp.stdout.strip()
        except Exception:
            pass
        return ""

    def draw(self) -> pygame.Surface:
        if not self.is_enabled():
            return None
        self.surface.fill(self.menu.get_theme().background_color)
        self.menu.draw(self.surface)
        return self.surface
    
    def call_wrapped_action(self, action_fn: Callable, args: any, kwargs: dict, immediate=False, toggle_menu=False):
        if not action_fn:
            return
        action_result = action_fn(*args)
        exc: Exception = None
        if immediate:
            if self.post_command_callback_fn:
                try:
                    self.post_command_callback_fn()
                except Exception as e:
                    exc = e
        if toggle_menu:
            if action_result != False: # noqa: E712
                self.toggle() 
        if exc:
            raise exc    

    def wrap_action(self, action_fn: Callable, immediate=False, toggle_menu=False):
        if not action_fn:
            return
        return lambda *args, **kwargs: self.call_wrapped_action(action_fn, args, kwargs, immediate, toggle_menu)

    def toggle_gui_mode(self, value: bool) -> None:
        self.settings.app_mode = AppMode.GUI_ADVANCED if value else AppMode.GUI_BASIC
        self.command_callback_fn(UICommand.CHANGE_SETTINGS)
