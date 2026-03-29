
import asyncio

import pygame
import pygame_menu

from .base_overlay import BaseOverlay


class ProgressOverlay(BaseOverlay):
    def __init__(self, window_width: int, window_height: int):
        self.progress = 0.0
        self.message = ""
        self.stop_event: asyncio.Event = None
        self.progress_widget: pygame_menu.widgets.ProgressBar | None = None
        self.message_label: pygame_menu.widgets.Label | None = None
        self.menu: pygame_menu.Menu | None = None
        super().__init__(window_width, window_height, pygame_menu.themes.THEME_BLUE, global_scale=0.5)
        self.theme.widget_alignment = pygame_menu.locals.ALIGN_CENTER

    def set_progress(self, progress: float, message: str | None = None):
        self.progress = max(0.0, min(100.0, progress))
        if message is not None:
            self.message = message
        self.force_render = True

    def show(self, progress: float, message: str | None = None, stop_event: asyncio.Event=None):
        self.stop_event = stop_event
        self.set_progress(progress, message=message)
        if not self.menu:
            self.build_menu()
        self.menu.enable()

    def hide(self):
        if self.menu:
            self.menu.disable()
        self.force_render = True

    def build_menu(self) -> None:
        window_width, window_height = self.window_width, self.window_height
        bar_width = int(window_width * 0.6)
        bar_width = max(120, bar_width)
        bar_height = max(12, int(window_height * 0.025))
        menu_width, menu_height = bar_width + 40, bar_height * 6

        self.menu = pygame_menu.Menu(
            title="",
            width=menu_width,
            height=menu_height,
            theme=self.theme,
            center_content=True,
            position=(50, 50, True),
            screen_dimension=(window_width, window_height),
        )

        self.message_label = self.menu.add.label(self.message or "")
        self.menu.add.vertical_margin(10)
        self.progress_widget = self.menu.add.progress_bar(
            title="",
            default=self.progress,
            max_value=100,
            width=bar_width,
            height=bar_height,
            progress_text_enabled=True,
            progress_text_fontsize=max(12, int(self.theme.widget_font_size * 1.0)),
        )
        if self.stop_event:
            self.menu.add.button("Cancel", self.close_dialog_with_cancel)
        self.force_render = True

    def close_dialog(self):
        self.menu = None
        self.progress_widget = None
        self.message_label = None
        self.force_render = True

    def close_dialog_with_cancel(self):
        self.close_dialog()
        if self.stop_event:
            self.stop_event.set()

    def on_resize(self) -> None:
        # Restore current values on resize
        if self.progress_widget:
            self.build_menu()
            self.progress_widget.set_value(self.progress)
            if self.is_enabled():
                self.progress_widget.enable()
        if self.message_label is not None and self.message:
            self.message_label.set_title(self.message)

    def process_event(self, event) -> bool:
        processed = False
        if self.is_enabled():
            processed = self.menu.update([event])
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE,):
                if self.stop_event:
                    self.close_dialog_with_cancel()
                processed = True
        return processed

    def is_enabled(self) -> bool:
        return self.menu and self.menu.is_enabled()

    def requires_pause(self) -> bool:
        return False

    def draw(self) -> pygame.Surface:
        if not self.is_enabled():
            return None
        self.surface.fill((0, 0, 0, 0))
        if self.message_label:
            self.message_label.set_title(self.message or "")
        if self.progress_widget:
            self.progress_widget.set_value(self.progress)
        if self.menu:
            self.menu.draw(self.surface)
        return self.surface

