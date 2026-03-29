
import asyncio

from .base_overlay import BaseOverlay
from .error_overlay import ErrorOverlay
from .menu_overlay import MenuOverlay, MenuType
from .progress_overlay import ProgressOverlay
from .toast_overlay import ToastOverlay


class OverlayManager:
    OVERLAY_MENU = 1
    OVERLAY_TOAST = 2
    OVERLAY_ERROR = 3
    OVERLAY_PROGRESS = 4

    _instance: "OverlayManager | None" = None

    def __init__(self):
        self.overlays: dict[int, BaseOverlay] = {}
        self.window: 'Window' = None # type: ignore  # noqa: F821

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = OverlayManager()
        return cls._instance

    def attach_window(self, window: 'Window'): # type: ignore  # noqa: F821
        window_width, window_height = window.wdef.window_size
        # self.overlays must be in order of z-index (higher first)
        # because overlays with higher z-index should catch the process_event(...) call first
        self.overlays = {
            self.OVERLAY_ERROR: ErrorOverlay(window_width, window_height),
            self.OVERLAY_TOAST: ToastOverlay(window_width, window_height),
            self.OVERLAY_PROGRESS: ProgressOverlay(window_width, window_height),
            self.OVERLAY_MENU: MenuOverlay(window_width, window_height, window.enqueue_command, window.post_command_callback_fn, window.settings),
        }
        self.window = window
    
    def resize(self, window_width: int, window_height: int):
        for overlay in self.overlays.values():
            overlay.resize(window_width, window_height)

    def get_enabled_overlays(self, reverse=False) -> list[MenuOverlay]:
        result = [overlay for overlay in self.overlays.values() if overlay.is_enabled()]
        if reverse:
            return reversed(result)
        return result

    def get_any_overlay_require_pause(self) -> bool:
        return any(overlay.requires_pause() for overlay in self.overlays.values())

    def get_any_overlay_require_force_render(self) -> bool:
        return any(overlay.requires_force_render() for overlay in self.overlays.values())

    def get(self, key: int):
        return self.overlays.get(key, None)
    
    def rebuild_menu(self, menu_type: MenuType):
        self.get(self.OVERLAY_MENU).build_menu(menu_type)

    def rebuild_and_show_menu(self, menu_type: MenuType):
        self.get(self.OVERLAY_MENU).rebuild_and_show(menu_type)

    def show_error_internal(self, exception):
        try:
            self.get(self.OVERLAY_ERROR).show_error(exception)
        except:
            print(f"Error: {exception}")

    def show_toast_internal(self, message: str, duration_ms=5_000):
        self.get(self.OVERLAY_TOAST).show_toast(message, duration_ms=duration_ms)

    def show_progress_internal(self, message: str, stop_event: asyncio.Event):
        overlay: ProgressOverlay = self.get(self.OVERLAY_PROGRESS)
        overlay.show(0, message=message, stop_event=stop_event)
        overlay.set_progress(0, message=message)
        return overlay

    @staticmethod
    def show_error(exception: Exception):
        manager = OverlayManager.instance()
        print(f"Error: {exception}")
        if not manager.window:
            return
        manager.show_error_internal(exception)

    @staticmethod
    def show_toast(message: str, duration_ms=5_000):
        manager = OverlayManager.instance()
        if not manager.window and message:
            print(f"INFO: {message}")
            return
        manager.show_toast_internal(message, duration_ms=duration_ms)

    @staticmethod
    def show_progress(message: str, stop_event: asyncio.Event=None): # type: ignore
        manager = OverlayManager.instance()
        if not manager.window:
            return None
        return manager.show_progress_internal(message, stop_event)
