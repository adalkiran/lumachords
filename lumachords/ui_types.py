from enum import IntEnum, auto
from typing import Callable

import pygame

from lumachords.runtime_config import AppSettings

class PlaybackMode(IntEnum):
    NOT_STARTED       = 0
    NORMAL            = 1
    PAUSED            = 2
    FRAME_FWD         = 3
    FRAME_FWD_PAUSED  = 4

class UICommand(IntEnum):
    START     = auto()
    DOWNLOAD_YOUTUBE = auto()
    QUIT      = auto()
    TAKE_SHOT = auto()
    SAVE_MIDI = auto()
    SAVE_MEI  = auto()
    SAVE_VIDEO = auto()
    CHANGE_SETTINGS = auto()

class UIUtils:
    has_video_system = None

    @staticmethod
    def ensure_video_system_init():
        if pygame.get_init():
            return __class__.has_video_system
        pygame.init()
        pygame.font.init()
        __class__.has_video_system = pygame.display.get_driver() not in ["dummy", "offscreen"]
        return __class__.has_video_system
    
    @staticmethod
    def create_window_object(settings: AppSettings, post_command_callback_fn: Callable = None):
        if not __class__.ensure_video_system_init():
            return None
        
        from lumachords.gui.window import Window
        
        screen_w, screen_h = pygame.display.get_desktop_sizes()[0]
        window = Window(
            "LumaChords",
            settings,
            (int(screen_w * 0.7), int(screen_h * 0.7)),
            (1, 1),
            post_command_callback_fn=post_command_callback_fn,
        )
        return window
