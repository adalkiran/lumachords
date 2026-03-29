from __future__ import annotations

from typing import TYPE_CHECKING

import pygame_menu

from lumachords.ui_types import UICommand
from lumachords.runtime_config import AppMode
from .common import CARD_BORDER, add_button, create_card

if TYPE_CHECKING:  # pragma: no cover
    from lumachords.overlays.menu_overlay import MenuOverlay


def build_menu_processing(menu_overlay: "MenuOverlay", menu: pygame_menu.Menu, menu_type) -> None:
    from lumachords.overlays.menu_overlay import MenuType  # local import to avoid cycles

    def pack_export_button(title: str, command: UICommand) -> None:
        export_card.pack(
            add_button(
                menu,
                title,
                menu_overlay.wrap_action(
                    lambda: menu_overlay.command_callback_fn(command),
                    immediate=True,
                    toggle_menu=False,
                ),
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_CENTER,
        )

    if menu_type == MenuType.MENU_PROCESSING:
        app_mode_toggle_width = max(190, int(menu.get_width() * 0.48))
        menu.add.button('Resume', menu_overlay.toggle, margin=(0, 0)).set_border(1, CARD_BORDER)
        menu.add.vertical_margin(8)
        menu.add.toggle_switch(
            "App Mode",
            default=(menu_overlay.settings.app_mode == AppMode.GUI_ADVANCED),
            state_text=("BASIC", "ADVANCED"),
            width=app_mode_toggle_width,
            onchange=menu_overlay.wrap_action(
                menu_overlay.toggle_gui_mode,
                immediate=True,
                toggle_menu=False,
            ),
            margin=(0, 0),
        )
        menu.add.vertical_margin(12)
    export_card = create_card(menu, "Export", "Save the result in the format you need.", 320 if menu_type == MenuType.MENU_PROCESSING else 280)
    pack_export_button("Save MIDI", UICommand.SAVE_MIDI)
    export_card.pack(menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_CENTER)
    pack_export_button("Save MEI", UICommand.SAVE_MEI)
    export_card.pack(menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_CENTER)
    pack_export_button("Save Video", UICommand.SAVE_VIDEO)
    if menu_type == MenuType.MENU_PROCESSING:
        export_card.pack(menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_CENTER)
        pack_export_button("Take Screenshot", UICommand.TAKE_SHOT)
    menu.add.vertical_margin(12)
