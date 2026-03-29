
import pygame
import pygame_menu

from .base_overlay import BaseOverlay


class ToastOverlay(BaseOverlay):
    def __init__(self, window_width: int, window_height: int):
        self.message = ""
        super().__init__(window_width, window_height, pygame_menu.themes.THEME_BLUE)
        self.duration_ms = 0
        self.fade_ms = 0
        self.start_ms = 0
        self.text_surface = None
        self.text_rect = None

    def show_toast(self, message, duration_ms=5_000):
        self.message = message
        self.duration_ms = max(200, duration_ms) if duration_ms > 0 else 0
        self.fade_ms = max(150, min(400, int(self.duration_ms * 0.25)))
        self.start_ms = pygame.time.get_ticks()
        self.build_menu()
        self.force_render = True

    def build_menu(self) -> None:
        if not self.message:
            self.close_dialog()
        font_size = max(12, int(self.theme.widget_font_size * 1.25))
        font = pygame.font.SysFont(None, font_size)
        lines = self.message.splitlines() or [""]
        line_surfaces = [font.render(line, True, (255, 255, 255)) for line in lines]
        max_w = max((s.get_width() for s in line_surfaces), default=0)
        total_h = sum(s.get_height() for s in line_surfaces)
        text_surface = pygame.Surface((max_w, total_h), pygame.SRCALPHA)
        y = 0
        for s in line_surfaces:
            text_surface.blit(s, (0, y))
            y += s.get_height()
        self.text_surface = text_surface
        self.text_rect = self.text_surface.get_rect(
            center=(self.window_width // 2, self.window_height // 2)
        )
    
    def close_dialog(self):
        self.message = ""
        self.force_render = True

    def process_event(self, event):
        processed = False
        if self.is_enabled() and event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE,):
            self.close_dialog()
            processed = True
        return processed

    def is_enabled(self) -> bool:
        if not self.message:
            return False
        if self.duration_ms > 0 and (pygame.time.get_ticks() - self.start_ms) >= self.duration_ms:
            self.close_dialog()
            return False
        return True

    def requires_pause(self) -> bool:
        return False

    def draw(self) -> pygame.Surface:
        if not self.is_enabled():
            return None

        elapsed_ms = pygame.time.get_ticks() - self.start_ms
        if self.duration_ms > 0 and elapsed_ms < self.fade_ms:
            alpha = int(255 * (elapsed_ms / self.fade_ms))
        elif self.duration_ms > 0 and (elapsed_ms > (self.duration_ms - self.fade_ms)):
            alpha = int(255 * ((self.duration_ms - elapsed_ms) / self.fade_ms))
        else:
            alpha = 255
        alpha = max(0, min(255, alpha))

        self.surface.fill((0, 0, 0, 0))
        self.surface.fill((0, 0, 0, 180), rect=self.text_rect)
        if self.text_surface and self.text_rect:
            self.text_surface.set_alpha(alpha)
            self.surface.blit(self.text_surface, self.text_rect)
        self.force_render = True
        return self.surface

    def on_resize(self) -> None:
        if self.message:
            self.build_menu()


