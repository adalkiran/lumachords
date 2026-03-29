import pygame
import pygame_menu

from .base_overlay import BaseOverlay


class ErrorOverlay(BaseOverlay):
    def __init__(self, window_width: int, window_height: int):
        self.exception = None
        self.current_dialog = None
        super().__init__(window_width, window_height, pygame_menu.themes.THEME_ORANGE, global_scale=0.7)

    def show_error(self, exception):
        self.exception = exception
        self.build_menu()
        return self.current_dialog

    def build_menu(self) -> None:
        window_width, window_height = self.window_width, self.window_height
        menu_width, menu_height = int(window_width * 0.8), int(window_height * 0.5)

        self.current_dialog = pygame_menu.Menu(
            title='Error',
            width=menu_width,
            height=menu_height,
            theme=self.theme,
            center_content=True,
            position=(50, 50, True),
            screen_dimension=(window_width, window_height),
        )

        self.current_dialog.add.label('An error occurred:', font_size=int(self.theme.widget_font_size * 1.25))
        self.current_dialog.add.label(str(self.exception), max_char=70, margin=(0, 0))
        self.current_dialog.add.vertical_margin(20)
        self.current_dialog.add.button('OK', self.close_dialog)
        self.force_render = True

    def close_dialog(self):
        self.current_dialog = None
        self.exception = None
        self.force_render = True

    def process_event(self, event) -> bool:
        processed = False
        if self.is_enabled():
            processed = self.current_dialog.update([event])
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE,):
                self.close_dialog()
                processed = True
        return processed

    def is_enabled(self) -> bool:
        return self.current_dialog and self.current_dialog.is_enabled()

    def draw(self) -> pygame.Surface:
        if not self.is_enabled():
            return None

        self.current_dialog.draw(self.surface)
        return self.surface

    def on_resize(self) -> None:
        if self.current_dialog:
            self.build_menu()
        
