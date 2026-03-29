# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import stat
import shutil

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

binaries = []
datas = []
hiddenimports = []

binaries += collect_dynamic_libs('verovio')
datas += collect_data_files('verovio', subdir='data')
hiddenimports += ["rtmidi", "mido.backends.rtmidi", "fluidsynth"]
hiddenimports += collect_submodules('yt_dlp')
datas += collect_data_files('yt_dlp')
datas += [('resources/*', 'resources')]
datas += [('THIRD_PARTY_LICENSES.txt', 'resources')]
datas += [('LICENSE', 'resources')]

WINDOWS_ICON = 'resources/icon.ico'
MACOS_ICON = 'resources/icon.icns'

a = Analysis(
    ['lumachords/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Matplotlib imports trigger Tkinter inclusion on GitHub Actions environment.
        # So, these packages are excluded explicitly here.
        "tkinter",
        "_tkinter",
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends._backend_tk",
    ],
    noarchive=False,
    optimize=0,
)

def add_alias_copy_for_fluidsynth(binaries):
    for alias, src, binary_type in list(binaries):
        base = os.path.basename(alias)

        # matches libfluidsynth.3.1.1.dylib, to create alias for an existing "libfluidsynth*.dylib" file coming from other dependencies, e.g. Pygame.
        if base.startswith("libfluidsynth.3.") and base.endswith(".dylib"):
            alias_dest = "libfluidsynth.dylib"
            if not any(existing_alias == alias_dest for existing_alias, _, _ in binaries):
                binaries.append((alias_dest, src, binary_type))
            return


add_alias_copy_for_fluidsynth(a.binaries)
# for x in a.binaries:
#     print("aaaaa", len(x), x)


pyz = PYZ(a.pure)

def build_exe(name, console):
    exe_icon = WINDOWS_ICON if sys.platform == "win32" and os.path.exists(WINDOWS_ICON) else None
    return EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        contents_directory="lib",
        icon=exe_icon,
    )


def copy_license_files():
    # DISTPATH is provided by PyInstaller when evaluating the .spec
    shutil.copyfile('THIRD_PARTY_LICENSES.txt', f'{DISTPATH}/THIRD_PARTY_LICENSES.txt')
    shutil.copyfile('LICENSE', f'{DISTPATH}/LICENSE')

def _remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

def flatten_dist_layout(app_file_name):
    if sys.platform == "darwin":
        # Keep .app + cli wrapper + licenses + lib only at top level.
        _remove_path(os.path.join(DISTPATH, app_file_name))
    else:
        # DISTPATH is provided by PyInstaller when evaluating the .spec
        source_dir = os.path.join(DISTPATH, app_file_name)
        if not os.path.isdir(source_dir):
            return

        shutil.move(source_dir, source_dir + "_tmp")
        source_dir += "_tmp"

        for item_name in os.listdir(source_dir):
            src = os.path.join(source_dir, item_name)
            dst = os.path.join(DISTPATH, item_name)
            if os.path.exists(dst):
                _remove_path(dst)
            shutil.move(src, dst)

        shutil.rmtree(source_dir)


def create_macos_cli(app_file_name):
    # DISTPATH is provided by PyInstaller when evaluating the .spec
    cli_path = os.path.join(DISTPATH, f"{app_file_name}-cli")

    script = f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec "./{app_file_name}.app/Contents/MacOS/{app_file_name}" "--mode=headless" "$@"
"""

    with open(cli_path, "w", newline="\n") as f:
        f.write(script)

    # chmod +x
    st_mode = os.stat(cli_path).st_mode
    os.chmod(cli_path, st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


exe = build_exe("LumaChords", console=False)
if sys.platform == "win32":
    exe_cli = build_exe("LumaChords-cli", console=True)
    coll = COLLECT(
        exe,
        exe_cli,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='LumaChords',
    )
else:
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='LumaChords',
    )
    if sys.platform == "darwin":
        bundle_icon = MACOS_ICON if os.path.exists(MACOS_ICON) else None
        app = BUNDLE(
            coll,
            name='LumaChords.app',
            icon=bundle_icon,
            bundle_identifier=None,
        )
        create_macos_cli('LumaChords')

copy_license_files()
flatten_dist_layout('LumaChords')
