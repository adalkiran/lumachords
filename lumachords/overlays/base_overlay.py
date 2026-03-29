
from abc import abstractmethod
import pygame
import pygame_menu


class BaseOverlay:
    BASE_W, BASE_H = 1920, 1080

    def __init__(self, window_width: int, window_height: int, base_theme: pygame_menu.Theme, global_scale: float=1.0):  
        self.window_width = window_width
        self.window_height = window_height
        self.global_scale = global_scale

        theme: pygame_menu.Theme = base_theme.copy()
        self.base_title_font_size = theme.title_font_size
        self.base_widget_font_size = theme.widget_font_size
        self.base_widget_padding = theme.widget_padding
        self.scale: float = 1.0
        self.theme = theme
        self.surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        self.force_render = False
        self.resize(window_width, window_height, force=True)

    def resize(self, window_width: int, window_height: int, force: bool=False):
        if (window_width, window_height) == (self.window_width, self.window_height) and not force:
            return False
        self.window_width = window_width
        self.window_height = window_height

        scale = max(window_width / __class__.BASE_W, window_height / __class__.BASE_H)
        self.scale = scale = scale * self.global_scale

        self.theme.title_font_size = max(12, int(self.base_title_font_size * scale))
        self.theme.widget_font_size = max(10, int(self.base_widget_font_size * scale))
        self.theme.base_widget_padding = (int(self.base_widget_padding[0] * scale), int(self.base_widget_padding[1] * scale))
        self.surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        self.on_resize()

        return True

    def on_resize(self) -> None:
        pass

    @abstractmethod
    def process_event(self, event) -> bool:
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        pass

    def requires_pause(self) -> bool:
        return self.is_enabled()
        
    def requires_force_render(self) -> bool:
        if self.force_render:
            self.force_render = False
            return True
        return self.force_render

    @abstractmethod
    def draw(self) -> pygame.Surface:
        pass

