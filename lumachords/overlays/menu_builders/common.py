from __future__ import annotations

import pygame_menu


CARD_BG = (15, 20, 30, 225)
CARD_BORDER = (56, 78, 102, 255)
CARD_TITLE = (236, 242, 255, 255)
CURSOR_COLOR = (236, 242, 255, 255)
CURSOR_SELECTION_COLOR =  (236, 242, 255, 120)
MUTED = (173, 188, 209, 255)
CARD_RIGHT_BORDER_INSET = 18


def style_button(
    button: pygame_menu.widgets.Button,
    *,
    border_width: int = 1,
    border_color=CARD_BORDER,
) -> pygame_menu.widgets.Button:
    button.set_border(border_width, border_color)
    return button


def card_width(menu: pygame_menu.Menu, *, min_width: int = 240, horizontal_padding: int = 18) -> int:
    return max(min_width, menu.get_width() - horizontal_padding)


def muted_font_size(menu: pygame_menu.Menu, *, scale: float = 0.74, min_size: int = 10) -> int:
    return max(min_size, int(menu.get_theme().widget_font_size * scale))


def create_card(
    menu: pygame_menu.Menu,
    title: str,
    body: str | None,
    height: int,
    *,
    min_width: int = 240,
    horizontal_padding: int = 18,
) -> pygame_menu.widgets.Frame:
    width = max(1, card_width(menu, min_width=min_width, horizontal_padding=horizontal_padding) - CARD_RIGHT_BORDER_INSET)
    card = menu.add.frame_v(width, height, margin=(0, 8))
    card._relax = True
    card._pack_margin_warning = False
    card.set_background_color(CARD_BG)
    card.set_border(1, CARD_BORDER)
    card.set_padding((14, 12))
    card.pack(
        menu.add.label(title, font_color=CARD_TITLE, align=pygame_menu.locals.ALIGN_LEFT, margin=(0, 0)),
        align=pygame_menu.locals.ALIGN_LEFT,
    )
    if body:
        card.pack(
            menu.add.label(
                body,
                max_char=70,
                font_color=MUTED,
                align=pygame_menu.locals.ALIGN_LEFT,
                font_size=muted_font_size(menu),
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
    return card


def pack_card_label(
    menu: pygame_menu.Menu,
    card: pygame_menu.widgets.Frame,
    text: str,
    *,
    color=None,
    size: int | None = None,
) -> pygame_menu.widgets.Label:
    label = menu.add.label(
        text,
        max_char=70,
        font_color=color or MUTED,
        align=pygame_menu.locals.ALIGN_LEFT,
        font_size=size or muted_font_size(menu),
        margin=(0, 0),
    )
    card.pack(label, align=pygame_menu.locals.ALIGN_LEFT)
    return label


def add_button(menu: pygame_menu.Menu, title: str, action, **kwargs) -> pygame_menu.widgets.Button:
    return style_button(menu.add.button(title, action, **kwargs))
