import asyncio
import os
import threading
import time
from typing import Callable

import pygame
import numpy as np

from lumachords.runtime_config import AppSettings
from lumachords.processing_state import ProcessingState
from lumachords.overlays import BaseOverlay, OverlayManager
from lumachords.ui_types import PlaybackMode, UICommand, UIUtils
from lumachords.utils import Utils

from .window_geometry_helper import Panel, WindowDef, WindowGeometryHelper
from .window_gl_helper import WindowGLHelper


class Window:
    USEREVENT_INTERRUPT = pygame.USEREVENT + 1
    USEREVENT_FRAME_SEEK = pygame.USEREVENT + 2
    USEREVENT_TAKE_SHOT = pygame.USEREVENT + 3
    USEREVENT_SAVE_MIDI = pygame.USEREVENT + 4
    USEREVENT_SAVE_MEI = pygame.USEREVENT + 5



    INTERRUPT_EVENTS = [USEREVENT_INTERRUPT, USEREVENT_FRAME_SEEK, USEREVENT_TAKE_SHOT, USEREVENT_SAVE_MIDI, USEREVENT_SAVE_MEI]


    def __init__(self, title: str, settings: AppSettings, window_size: tuple[int, int], frame_size: tuple[int, int], gap: int = 30, progress_bar_size_rates: tuple[float, float] = (0.6, 0.02), post_command_callback_fn: Callable = None):
        self.title = title
        self.settings = settings
        self.post_command_callback_fn = post_command_callback_fn
        self.wdef = WindowDef(window_size, frame_size, gap, None, progress_bar_size_rates)
        self.state: ProcessingState = None

        self.panels: dict[int, Panel] = None
        self.pbar_viewport = None

        self.title_font = None
        self.pbar_font = None
        self.clock = None
        self.thread = None

        self.frame_seek_timeout_ms = 100


        self.playback_mode = PlaybackMode.NORMAL
        self.progress_current = 0
        self.progress_max = 0
        self.progress_text = None
        self.progress_post_text = None

        self.command_queue: asyncio.Queue[UICommand] = asyncio.Queue()
            
    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def bind_state(self, state: ProcessingState):
        # TODO: Cleanup/free state-related variables, buffers, etc...
        if self.state:
            state.inherit_from(self.state)
        self.state = state
        self.wdef.panel_titles = self.state.panel_titles
        if pygame.get_init():
            self.rebuild_layout_and_gl()

    def progresss_init(self, progress_max):
        self.progress_current = 0
        self.progress_max = progress_max
        self.progress_text = "Initializing..."

    def enqueue_command(self, command: UICommand):
        self.command_queue.put_nowait(command)

    def handle_events(self, wait_for_events: list[int]=None, wait_ms=None):
        # Handle window events (quit / ESC / Q / resize)
        events = pygame.event.wait(wait_ms) if (wait_for_events is not None and wait_ms) else pygame.event.get()
        if not isinstance(events, list):
            events = [events]
        overlay_manager = OverlayManager.instance()
        menu_overlay = overlay_manager.get(OverlayManager.OVERLAY_MENU)
        for event in events:
            if event.type == pygame.QUIT:
                self.enqueue_command(UICommand.QUIT)
                return False

            event_processed = False
            for overlay in overlay_manager.get_enabled_overlays():
                event_processed = overlay.process_event(event)
                if event_processed:
                    break
                if overlay.requires_pause() and event.type != pygame.VIDEORESIZE and event.type != pygame.QUIT:
                    event_processed = True
                    break
            if event_processed:
                continue
            

            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                if menu_overlay:
                    menu_overlay.toggle()
                return True

            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                if menu_overlay:
                    menu_overlay.show_confirm_quit()
                    return True
                self.enqueue_command(UICommand.QUIT)
                return False
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_p or event.key == pygame.K_SPACE):
                self.playback_mode = PlaybackMode.NORMAL if self.playback_mode != PlaybackMode.NORMAL else PlaybackMode.PAUSED
                if self.playback_mode == PlaybackMode.PAUSED:
                    # Stop the repeating timer when the key is released
                    pygame.time.set_timer(__class__.USEREVENT_FRAME_SEEK, 0)
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                # Start a repeating timer to emit step-forward events every frame_seek_timeout_ms while key is held
                pygame.time.set_timer(__class__.USEREVENT_FRAME_SEEK, self.frame_seek_timeout_ms, loops=0)
                self.playback_mode = PlaybackMode.FRAME_FWD
                return __class__.USEREVENT_FRAME_SEEK
            if event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
                # Stop the repeating timer when the key is released
                pygame.time.set_timer(__class__.USEREVENT_FRAME_SEEK, 0)
                self.playback_mode = PlaybackMode.FRAME_FWD_PAUSED
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.enqueue_command(UICommand.TAKE_SHOT)
                return __class__.USEREVENT_TAKE_SHOT
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.enqueue_command(UICommand.SAVE_MIDI)
                return __class__.USEREVENT_SAVE_MIDI
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.enqueue_command(UICommand.SAVE_MEI)
                return __class__.USEREVENT_SAVE_MEI
            if event.type == pygame.VIDEORESIZE:
                new_size = (int(event.w), int(event.h))
                self.apply_window_size(new_size)
                return True
            if event.type == __class__.USEREVENT_FRAME_SEEK and self.playback_mode not in (PlaybackMode.FRAME_FWD_PAUSED, PlaybackMode.FRAME_FWD):
                return True
            if wait_for_events is not None and event.type in wait_for_events:
                return event.type
        return True

    async def sleep_if_paused_async(self, callback_fn: Callable[[], None] = None):
        is_first = True
        overlay_manager = OverlayManager.instance()
        if overlay_manager.get_any_overlay_require_pause():
            callback_fn()
        while (require_pause := overlay_manager.get_any_overlay_require_pause()) \
            or (require_force_render := overlay_manager.get_any_overlay_require_force_render()) \
            or self.playback_mode != PlaybackMode.NORMAL:
            if is_first:
                is_first = False
                if self.playback_mode == PlaybackMode.PAUSED and callback_fn:
                    callback_fn()
            force_render_applicable = (self.playback_mode not in [PlaybackMode.NORMAL, PlaybackMode.FRAME_FWD] and require_force_render)
            just_render = require_pause or force_render_applicable
            if just_render:
                if not await self.present_async(30):
                    return False
            else:
                if force_render_applicable:
                    if not await self.present_async(30):
                        return False
                handle_event_result = self.handle_events(wait_for_events=__class__.INTERRUPT_EVENTS, wait_ms=0)
                if not handle_event_result:
                    return False
                elif handle_event_result != True: # noqa: E712
                    return True
                await asyncio.sleep(0)
        return True

    async def controlled_sleep_async(self, sleep_ms):
        """Non-blocking sleep that still pumps events."""
        if sleep_ms <= 0:
            await asyncio.sleep(0)
            return True
        deadline = time.time() + (sleep_ms / 1_000.0)
        while time.time() < deadline:
            event_result = self.handle_events(wait_for_events=__class__.INTERRUPT_EVENTS, wait_ms=0)
            if event_result is False:
                return False
            await asyncio.sleep(0)
        return True

    # ------------------------------------------------------------------
    # OpenGL + pygame setup
    # ------------------------------------------------------------------

    def create_window(self) -> bool:
        self.thread = threading.current_thread()
        if not UIUtils.ensure_video_system_init():
            return False
        self.clock = pygame.time.Clock()

        # Ask for vsync if available (safe no-op if unsupported)
        try:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        except Exception:
            pass

        pygame.display.set_caption(self.title)
        icon = pygame.image.load(Utils.resource_path("icon_64x64.png", "resources"))
        pygame.display.set_icon(icon)

        self.apply_window_size(self.wdef.window_size, force=True, apply_center=True)
        OverlayManager.instance().attach_window(self)
        return True

    def update_panel_image(self, panel_id: int, frame_bgr: np.ndarray) -> None:
        WindowGLHelper.update_panel_image(self.panels[panel_id], frame_bgr)

    def restore_viewport(self):
        WindowGLHelper.restore_viewport(self.wdef)

    def begin_gl_context(self):
        WindowGLHelper.begin_gl_context(self.wdef)

    def end_gl_context(self):
        WindowGLHelper.end_gl_context()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_panels(self) -> None:
        for panel in self.panels.values():
            panel.calculate_pos()
        WindowGLHelper.draw_panel_images(self.panels)
        self.restore_viewport()
        self.begin_gl_context()
        WindowGLHelper.draw_panel_titles(self.panels)
        self.end_gl_context()

    def draw_note_result(self) -> None:
        # RENDER MIDI EVENTS TEXT
        surf_rgba, tw, th = WindowGLHelper.render_multiline_text(self.state.midi_events_str, self.title_font, (255, 255, 255), min_line_count=8)
        midi_events_panel = self.panels[ProcessingState.IDX_MIDI_EVENTS]
        midi_events_panel.viewport.y = self.wdef.window_size[1] - self.wdef.gap // 2 - th
        midi_events_panel.update_size(tw, th)
        midi_events_panel.image_tex = WindowGLHelper.build_surface_texture(surf_rgba, tw=tw, th=th, existing_tex_id=midi_events_panel.image_tex)

        # RENDER INFO PANEL TEXT
        surf_rgba, tw, th = WindowGLHelper.render_multiline_text(self.state.info_panel_str, self.title_font, (255, 255, 255), min_line_count=8)
        info_panel = self.panels[ProcessingState.IDX_INFO]
        info_panel.viewport.y = self.wdef.window_size[1] - self.wdef.gap // 2 - th
        info_panel.update_size(tw, th)
        info_panel.image_tex = WindowGLHelper.build_surface_texture(surf_rgba, tw=tw, th=th, existing_tex_id=info_panel.image_tex)

    def draw_progress_bar(self) -> None:
        progress_rate = (self.progress_current / self.progress_max) if self.progress_max else 0
        self.begin_gl_context()
        WindowGLHelper.draw_progress_bar(self.pbar_viewport, self.pbar_font, progress_rate, f"{progress_rate*100: 3.1f}% | Frame: {self.progress_current:>6} / {self.progress_max:>6} {"Paused." if self.playback_mode == PlaybackMode.PAUSED else self.progress_text}{(" " + self.progress_post_text) if self.progress_post_text else ""}")
        self.end_gl_context()

    def draw_overlay(self, overlay: BaseOverlay) -> None:
        if overlay and overlay.is_enabled():
            surf = overlay.draw()
            if surf:
                self.begin_gl_context()
                WindowGLHelper.begin_gl_blend()
                WindowGLHelper.draw_surface(surf, 0, 0)
                WindowGLHelper.end_gl_blend()
                self.end_gl_context()
            
    def draw_overlays(self) -> None:
        for overlay in OverlayManager.instance().get_enabled_overlays(reverse=True):
            self.draw_overlay(overlay)

    def fill_demo_content(self):
        # Example: fill each dynamic panel with a simple gradient once
        for pid, panel in self.panels.items():
            if panel.has_variable_size:
                continue
            if panel.viewport:
                w, h = panel.viewport.w, panel.viewport.h
                img = np.zeros((h, w, 3), dtype=np.uint8)
                img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
                img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
                img[:, :, 2] = 80
                self.update_panel_image(pid, img)

    def delete_panel_textures(self):
        try:
            WindowGLHelper.delete_panel_textures(self.panels)
        except Exception:
            pass

    def apply_window_size(self, new_size: tuple[int, int], force: bool=False, apply_center: bool=False):
        """Set the window size and restore base GL state."""
        if not force and self.wdef.window_size == new_size:
            return
        screen_w, screen_h = pygame.display.get_desktop_sizes()[0]
        w, h = new_size
        w = min(screen_w, w)
        h = min(screen_h, h)
        new_size = (w, h)

        self.wdef.window_size = new_size
        if apply_center:
            #x, y = (screen_w - w) // 2, (screen_h - h) // 2
            #os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)
            os.environ['SDL_VIDEO_WINDOW_POS'] = "center"
        # Cocoa (macOS) behaves best with a plain 2D surface:
        pygame.display.set_mode(
            self.wdef.window_size,
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        self.restore_viewport()
        WindowGLHelper.reset_gl_state()
        self.rebuild_layout_and_gl()
        self.present(30)

    def rebuild_layout_and_gl(self):
        """Recompute layout and rebuild GL resources for the current window size."""
        # Drop old GL resources before reallocating
        self.delete_panel_textures()

        layout = WindowGeometryHelper.build_layout(self.wdef)
        self.panels = layout["panels"]
        self.pbar_viewport = layout["pbar_viewport"]

        # Scale title font with window height to keep proportional sizing
        title_px = max(12, int(self.wdef.window_size[1] * 0.018))
        self.title_font = pygame.font.SysFont(None, title_px)
        self.pbar_font = pygame.font.SysFont(None, int(self.pbar_viewport.h * 0.95))

        WindowGLHelper.allocate_panel_textures(self.panels)

        WindowGLHelper.create_title_textures(self.panels, self.title_font)
        OverlayManager.instance().resize(*self.wdef.window_size)


    def present(self, preview_fps: float, throttle: bool = True) -> bool:
        if not self.handle_events():
            return False
        if self.state:
            for i, state in self.state.states.items():
                self.update_panel_image(i, state.image)

        WindowGLHelper.begin_frame()
        if self.playback_mode != PlaybackMode.NOT_STARTED:
            self.draw_note_result()
            self.draw_panels()
            self.draw_progress_bar()
        self.draw_overlays()

        pygame.display.flip()
        if throttle and preview_fps:
            self.clock.tick(preview_fps)
        return True

    async def present_async(self, preview_fps: float) -> bool:
        """Async-friendly present that yields instead of ticking the clock."""
        if not self.present(preview_fps, throttle=False):
            return False
        if preview_fps:
            await asyncio.sleep(1.0 / preview_fps)
        else:
            await asyncio.sleep(0)
        return True

    def demo_main_loop(self) -> None:
        """Simple event loop that draws all panels."""
        self._running = True
        fps = 10

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False
                elif event.type == pygame.VIDEORESIZE:
                    new_size = (int(event.w), int(event.h))
                    self.apply_window_size(new_size)
            self.present(fps)

        self.cleanup()

    def cleanup(self) -> None:
        if threading.current_thread() is not self.thread:
            raise RuntimeError("gl_destroy must be called from the same thread as gl_init")
        try:
            self.delete_panel_textures()
            WindowGLHelper.cleanup_gl()
        except Exception:
            pass

        try:
            pygame.display.set_mode((1, 1), 0)  # drop GL context
            pygame.display.flip()
        except Exception:
            pass

        # Ask nicely
        try:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        except Exception:
            pass
        time.sleep(0.01)

        # Pump a couple times to let SDL process QUIT
        for _ in range(2):
            try:
                pygame.event.pump()
            except Exception:
                pass
            time.sleep(0.01)

            # Tear down video subsystem first, then the rest
            try:
                pygame.display.quit()
            except Exception:
                pass

            time.sleep(0.02)  # tiny grace helps Cocoa finalize the window

            try:
                pygame.quit()
            except Exception:
                pass

            # final pump (some macOS builds like one last tick)
            try:
                pygame.event.pump()
            except Exception:
                pass
            time.sleep(0.005)

    async def cleanup_async(self) -> None:
        """Async version of cleanup using non-blocking sleeps."""
        if threading.current_thread() is not self.thread:
            raise RuntimeError("gl_destroy must be called from the same thread as gl_init")
        try:
            self.delete_panel_textures()
            WindowGLHelper.cleanup_gl()
        except Exception:
            pass

        try:
            pygame.display.set_mode((1, 1), 0)  # drop GL context
            pygame.display.flip()
        except Exception:
            pass

        try:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        except Exception:
            pass
        await asyncio.sleep(0.01)

        for _ in range(2):
            try:
                pygame.event.pump()
            except Exception:
                pass
            await asyncio.sleep(0.01)

            try:
                pygame.display.quit()
            except Exception:
                pass

            await asyncio.sleep(0.02)

            try:
                pygame.quit()
            except Exception:
                pass

            try:
                pygame.event.pump()
            except Exception:
                pass
            await asyncio.sleep(0.005)
    
"""
if __name__ == "__main__":
    # Example usage
    #titles = ["Top Panel"] # 1
    #titles = ["Top Panel", "X C"] # 2
    titles = ["Top Panel", "Bottom Left", "Bottom Right"] # 3
    #titles = ["Top Panel", "Bottom Left", "Bottom Right", "AB"] # 4
    #titles = ["Top Panel", "Bottom Left", "Bottom Right", "AB", "CD"] # 5
    #titles = ["Top Panel", "Bottom Left", "Bottom Right", "AB", "CD", "EF"] # 6
    win = Window((1280, 720), (1280, 720), titles)
    win.create_window()
    win.demo_main_loop()
"""
