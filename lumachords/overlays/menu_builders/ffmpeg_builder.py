from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING
import webbrowser

import pygame
import pygame_menu

from lumachords.profile_store import ProfileStore
from lumachords.ui_types import UICommand

from lumachords.gui.file_dialog_utils import FileDialogUtils
from lumachords.utils import Utils
from lumachords.video.backends.ffmpeg_video_backends import FfmpegVideoUtils
from .common import CARD_TITLE, MUTED, add_button, create_card, muted_font_size

if TYPE_CHECKING:  # pragma: no cover
    from lumachords.overlays.menu_overlay import MenuOverlay

INSTALLING_FFMPEG_DOC_URL = "https://adalkiran.github.io/lumachords/installing-ffmpeg.html"
INSTALLING_FFMPEG_DOC_LOCAL_PATH = "docs/installing-ffmpeg.html"
TERMINAL_MENU_ITEMS: list[tuple[str, str, str | None]] = []

def _get_ffmpeg_doc_url():
    local_path = Path(INSTALLING_FFMPEG_DOC_LOCAL_PATH)
    if local_path.is_file():
        local_path = str(local_path.resolve())
        return "file://" + ("" if local_path.startswith("/") else "/") + local_path
    else:
        return INSTALLING_FFMPEG_DOC_URL

def _process_url_or_anchor(url_or_anchor: str):
    if url_or_anchor.startswith("#"):
        url_prefix = _get_ffmpeg_doc_url()
        return f"{url_prefix}{url_or_anchor}"
    else:
        return url_or_anchor


def _open_url(url: str):
    opened = False
    try:
        opened = webbrowser.open(url, new=2)
    except Exception:
        opened = False
    if not opened:
        from lumachords.overlays.overlay_manager import OverlayManager

        OverlayManager.show_error(f"Website couldn't be opened using your web browser, please visit this website using your browser manually:\n{url}")

def continue_without_ffmpeg(menu_overlay: "MenuOverlay") -> None:
    menu_overlay.settings.video_backend = "opencv"
    ProfileStore.set_continue_without_ffmpeg(True)
    menu_overlay.command_callback_fn(UICommand.CHANGE_SETTINGS)
    menu_overlay.toggle()

def open_ffmpeg_file_dialog(menu_overlay: "MenuOverlay") -> None:
    from lumachords.video.backends.ffmpeg_video_backends import FfmpegVideoUtils

    binary_file_path = FileDialogUtils.show_open_dialog("Select FFmpeg Executable File", FileDialogUtils.FILTER_VIDEO_FILES, start_dir="/")
    if binary_file_path:
        binary_file_dir = FfmpegVideoUtils.check_ffmpeg_binary(binary_file_path, check_exists=False)
        if binary_file_dir is not None:
            FfmpegVideoUtils.set_ffmpeg_binary_path(binary_file_dir)
            menu_overlay.toggle()
        else:
            raise Exception(f"The selected file is not an FFmpeg executable file:\n {binary_file_path}")

def retry_search_ffmpeg(menu_overlay: "MenuOverlay", show_error=True):
    from lumachords.video.backends.ffmpeg_video_backends import FfmpegVideoUtils
    from lumachords.overlays.overlay_manager import OverlayManager

    if FfmpegVideoUtils.locate_ffmpeg_binary(None, check_direct_call=True):
        menu_overlay.toggle()
        OverlayManager.show_toast("FFmpeg is located successfully!")
    elif show_error:
        OverlayManager.show_error("FFmpeg could not have been located.")

def linux_distro_id():
    """
    Returns (id, id_like) from /etc/os-release when available.
    """
    os_id = ""
    os_like = ""
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"')
                if k == "ID":
                    os_id = v.lower()
                elif k == "ID_LIKE":
                    os_like = v.lower()
    except Exception:
        pass
    return os_id, os_like

def create_ffmpeg_install_alternatives():
    platform_hint = None
    command_list = None
    platform = sys.platform
    if platform.startswith("win"):
        platform_hint = "Windows detected"
        """
        command_list = [
            (
                ["winget", "--version"],
                "Install with winget",
                'win-winget',
            ),
            (
                ["choco", "-v"],
                "Install with Chocolatey",
                "win-choco",
            ),
            (
                ["scoop", "--version"],
                "Install with scoop",
                "win-scoop",
            ),
            (None, "Direct Download", "win-direct"),
            (None, "Build from Source Code", "win-source"),
        ]
        """
    elif platform == "darwin":
        platform_hint = "MacOS detected"
        """
        command_list = [
            (
                ["brew", "--version"],
                "Install with Homebrew",
                "mac-brew",
            ),
            (
                ["port", "version"],
                "Install with MacPorts",
                "mac-macports",
            ),
            (None, "Direct Download", "mac-direct"),
            (None, "Build from Source Code", "mac-source"),
        ]
        """
    elif platform.startswith("linux"):
        os_id, os_like = linux_distro_id()
        distro_hint = f"{os_id} {os_like}".strip()
        if distro_hint:
            platform_hint = f"Linux detected: {distro_hint}"
        else:
            platform_hint = "Linux detected"
        """
        # Choose likely manager based on distro family
        is_deb = any(x in (os_id + " " + os_like) for x in ["debian", "ubuntu", "mint", "pop"])
        is_rpm = any(x in (os_id + " " + os_like) for x in ["fedora", "rhel", "centos", "rocky", "almalinux", "suse"])
        is_arch = any(x in (os_id + " " + os_like) for x in ["arch", "manjaro", "endeavouros"])
        is_alpine = "alpine" in (os_id + " " + os_like)

        command_list = [
            (
                is_deb, 
                (
                    ["apt", "--version"],
                    "Install with apt",
                    "lin-apt",
                )
            ),
            (
                is_arch,
                (
                    ["pacman", "-V"],
                    "Install with pacman (Arch)",
                    "lin-pacman",
                )
            ),
            (
                is_alpine,
                (
                    ["apk", "--version"],
                    "Install with apk (Alpine)",
                    "lin-apk",
                )
            ),
            (
                is_rpm,
                (
                    ["dnf", "--version"],
                    "Install with dnf (Fedora)",
                    "lin-dnf",
                )
            ),
            (
                is_rpm,
                (
                    ["yum", "--version"],
                    "Install with yum (RHEL / CentOS / AlmaLinux / Rocky Linux)",
                    "lin-yum",
                )
            ),
            (
                is_rpm,
                (
                    ["zypper", "--version"],
                    "Install with zypper (openSUSE)",
                    "lin-zypper",
                )
            ),
            (
                True,
                (None, "Direct Download", "lin-direct")
            ),
            (
                True,
                (None, "Build from Source Code", "lin-source")
            ),
        ]
        fallback = not (is_deb or is_arch or is_alpine or is_rpm)
        command_list = [command for distro_bool, command in command_list if (fallback or distro_bool)]
        """
    command_list = [
        (
            (None, "FFmpeg Installation Instructions (LumaChords)", "")
        ),

    ]
    return platform_hint, command_list



def build_menu_ffmpeg(menu_overlay: "MenuOverlay", menu: pygame_menu.Menu) -> None:
    """Populate the FFmpeg warning/config menu section."""
    def pack_card_button(card, title: str, action) -> None:
        card.pack(
            add_button(
                menu,
                title,
                action,
                margin=(0, 0),
            ),
            align=pygame_menu.locals.ALIGN_CENTER,
        )

    try:
        im_ffmpeg_logo = pygame.image.load(Utils.resource_path("ffmpeg-logo.svg", "resources")).convert_alpha()
        menu.add.surface(im_ffmpeg_logo, align=pygame_menu.locals.ALIGN_CENTER)
    except Exception:
        menu.add.label("FFmpeg", font_color=CARD_TITLE)
    menu.add.label("FFmpeg was not found.", max_char=70, font_color=CARD_TITLE)
    menu.add.label("Choose the suitable path below. You can keep using the app without FFmpeg, but with reduced media support.", max_char=70, font_color=MUTED)
    menu.add.vertical_margin(10)

    continue_card = create_card(menu, "Fastest Path", "Continue now with the OpenCV backend. Video can still work, but performance may be slower and audio export will be unavailable.", 150, min_width=320, horizontal_padding=26)
    pack_card_button(continue_card, "Continue Without FFmpeg", menu_overlay.wrap_action(
            (lambda: continue_without_ffmpeg(menu_overlay)),
            immediate=True, toggle_menu=False,
        ))

    locate_card = create_card(menu, "Use An Existing Installation", "If FFmpeg is already installed, point LumaChords to it or retry the automatic search.", 200, min_width=320, horizontal_padding=26)
    pack_card_button(locate_card, "Locate FFmpeg...", menu_overlay.wrap_action(
            (lambda: open_ffmpeg_file_dialog(menu_overlay)),
            immediate=True, toggle_menu=False,
        ))
    locate_card.pack(menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_CENTER)
    pack_card_button(locate_card, "Retry Search", menu_overlay.wrap_action(
            (lambda: retry_search_ffmpeg(menu_overlay)),
            immediate=True, toggle_menu=False,
        ))

    install_card = create_card(menu, "Install FFmpeg", "Open the installation guide, install FFmpeg, then return here and retry the search.", 240, min_width=320, horizontal_padding=26)
    platform_hint, command_list = create_ffmpeg_install_alternatives()
    if platform_hint is not None:
        install_card.pack(
            menu.add.label(platform_hint, align=pygame_menu.locals.ALIGN_LEFT, font_color=MUTED, font_size=muted_font_size(menu), margin=(0, 0)),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        install_card.pack(menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_CENTER)
    for command_descriptor in command_list:
        test_cmd_args, btn_label, url_anchor = command_descriptor
        btn_label = f"Instructions for: {btn_label}" if url_anchor else f"Visit: {btn_label}"
        btn = add_button(
            menu,
            btn_label,
            menu_overlay.wrap_action(lambda u=f"#{url_anchor}": _open_url(_process_url_or_anchor(u)), immediate=True, toggle_menu=False),
            margin=(0, 0),
        )
        if test_cmd_args and not FfmpegVideoUtils.test_binary_executable(test_cmd_args):
            btn.update_font({'color': (210, 210, 210), 'selected_color': menu.get_theme().widget_font_color})
        install_card.pack(btn, align=pygame_menu.locals.ALIGN_CENTER)
        install_card.pack(menu.add.vertical_margin(6), align=pygame_menu.locals.ALIGN_CENTER)
    pack_card_button(
        install_card,
        "Official FFmpeg Downloads",
        menu_overlay.wrap_action(lambda: _open_url("https://ffmpeg.org/download.html"), immediate=True, toggle_menu=False),
    )

    menu.add.vertical_margin(20)


def print_ffmpeg_terminal_instructions():
    """
    Terminal-friendly FFmpeg guidance with a simple one-key menu.
    """
    TERMINAL_MENU_ITEMS.clear()
    sys.stdout.write("\nThis application requires FFmpeg, a third-party multimedia framework, is installed on your computer.\nWe could not have located the FFmpeg executable files.\nInstall it or add it to PATH.\n")
    sys.stdout.write("\nYou can continue without FFmpeg (it's optional).")
    sys.stdout.write("\nBut, without FFmpeg, video reading/writing may be slower and audio won’t be included.\n")
    platform_hint, command_list = create_ffmpeg_install_alternatives()
    if platform_hint:
        sys.stdout.write(f"{platform_hint}\n")

    # Build unified menu inspired by GUI order (numbers for install links, letters for actions).
    for _, btn_label, url_anchor in command_list:
        url = _process_url_or_anchor(f"#{url_anchor}")
        TERMINAL_MENU_ITEMS.append(("link", f"Instructions for: {btn_label}", url))
    # Add official downloads link, mirroring GUI button
    TERMINAL_MENU_ITEMS.append(("link", "Visit: Official FFmpeg Downloads (ffmpeg.org)", "https://ffmpeg.org/download.html"))

    sys.stdout.write("\nChoose an option (enter number for a link, or letter):\n")
    for idx, (_, label, _) in enumerate(TERMINAL_MENU_ITEMS, start=1):
        sys.stdout.write(f"  [{idx}] {label}\n")

    sys.stdout.write("\n")
    sys.stdout.write("  [c] Continue without FFmpeg\n")
    sys.stdout.write("  [r] Retry search for FFmpeg\n")
    sys.stdout.write("  [q] Quit\n")
    sys.stdout.write("Type and press Enter.\n")
    sys.stdout.flush()


def get_terminal_menu_items() -> list[tuple[str, str, str | None]]:
    """Return the last generated terminal menu items list."""
    return TERMINAL_MENU_ITEMS


def handle_ffmpeg_prompt(app, ch: str) -> None:
    """
    Handle one-key FFmpeg helper menu in headless mode.
    Expects `app` to provide: command_queue, ffmpeg_prompt_active, ffmpeg_prompt_event.
    """
    ch = ch.lower()
    try:
        if ch.isdigit():
            idx = int(ch) - 1
            items = get_terminal_menu_items()
            if 0 <= idx < len(items):
                kind, label, payload = items[idx]
                if kind == "link":
                    url = payload
                    try:
                        import webbrowser
                        if not webbrowser.open(url, new=2):
                            raise Exception("Dummy")
                        sys.stdout.write(f"Opening {label}...\n")
                    except Exception:
                        sys.stdout.write(f"Couldn't open the browser, please visit: {url}\n")
                    sys.stdout.flush()
                    return
        if ch == "c":
            app.settings.video_backend = "opencv"
            ProfileStore.set_continue_without_ffmpeg(True)
            sys.stdout.write('Video backend changed to "OpenCV". Continuing.\n')
            app.ffmpeg_prompt_active = False
            if app.ffmpeg_prompt_event and not app.ffmpeg_prompt_event.done():
                app.ffmpeg_prompt_event.set_result(True)
        if ch == "r":
            found = FfmpegVideoUtils.locate_ffmpeg_binary(None, check_direct_call=True)
            if found:
                sys.stdout.write("FFmpeg located. Continuing.\n")
                app.ffmpeg_prompt_active = False
                if app.ffmpeg_prompt_event and not app.ffmpeg_prompt_event.done():
                    app.ffmpeg_prompt_event.set_result(True)
            else:
                sys.stdout.write("Still not found. Install FFmpeg or choose another option from the menu.\n")
            sys.stdout.flush()
            return
        if ch == "q":
            app.command_queue.put_nowait(UICommand.QUIT)
            if app.ffmpeg_prompt_event and not app.ffmpeg_prompt_event.done():
                app.ffmpeg_prompt_event.set_result(True)
            return
        # Unknown input; re-show menu
        print_ffmpeg_terminal_instructions()
    except Exception as e:
        sys.stdout.write(f"Error handling FFmpeg prompt: {e}\n")
        sys.stdout.flush()
