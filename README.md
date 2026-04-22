<p align="center">
  <a href="https://pypi.org/project/lumachords">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/adalkiran/lumachords/HEAD/docs/images/banner_dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/adalkiran/lumachords/HEAD/docs/images/banner_light.png">
      <img alt="LumaChords" src="https://raw.githubusercontent.com/adalkiran/lumachords/HEAD/docs/images/banner_light.png">
    </picture>
  </a>
</p>
<p align="center">
    <em>LumaChords, computer vision powered, for turning piano tutorial videos into MIDI, MEI and musical notation videos</em>
</p>
<p align="center">
    <a href="https://www.linkedin.com/in/alper-dalkiran/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&style=flat-square" alt="LinkedIn"></a>
    <a href="https://x.com/aalperdalkiran"><img src="https://img.shields.io/badge/X-1DA1F2?style=for-the-badge&logo=X&logoColor=white&style=flat-square" alt="X"></a>
    <a href="https://hits.dwyl.com/adalkiran/lumachords.svg?style=flat-square"><img src="https://hits.dwyl.com/adalkiran/lumachords.svg?style=flat-square" alt="HitCount"></a>
    <br />
    <a href="https://pypi.org/project/lumachords"><img src="https://img.shields.io/pypi/v/lumachords?color=blue&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/lumachords"><img src="https://img.shields.io/badge/python-3.12-blue" alt="Supported Python versions"></a>
    <a href="https://img.shields.io/badge/License-Apache%202.0-blue.svg"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue"><img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue" alt="Platform"></a>
</p>

---

Convert piano tutorial videos into MIDI, MEI files or videos with beautiful sheet music overlays! LumaChords uses computer vision to detect piano keys and falling notes in Synthesia-style tutorial videos, then generates musical notation synchronized with video playback.

Great for piano learners, enthusiasts, hobbyists, and music educators who want to extract note information from tutorial or performance videos and view it as sheet music.

It is also a strong end-to-end sample project for computer vision enthusiasts, with a multi-stage pipeline that includes specialized strategies for challenging cases, plus both a game-like GUI built with Pygame and OpenGL and a headless terminal interface.

LumaChords was developed primarily for personal use, educational purposes, and experimentation, combining software engineering, computer vision, and musical interests rather than commercial goals. For the same reason, it has been open-sourced.

LumaChords was designed and developed by a human with AI-assistance, is is not a vibecoded project. The initial commit history is intentionally clean, since the earlier draft repository had many (~250) experimental commits.

<p align="center">
  <img src="https://raw.githubusercontent.com/adalkiran/lumachords/HEAD/docs/images/lumachords-screen-recording.webp">
</p>

## 💭 **WHY THIS PROJECT?**

As a piano learner and enthusiast, I often explore different sources for my favorite songs I can play, including sheet music, piano tutorial videos, and MIDI files, both free and paid. However, these sources often have significant differences in musical interpretation, such as simplified chords, arpeggios, and other arrangements, and many of them do not sound close enough to the original song.

Sometimes I found piano tutorial videos that matched what I was looking for, but they often had no corresponding sheet music available, whether free or paid.

There are open-source projects that cover certain parts of LumaChords’ overall goal, but they may focus on only a narrow subset of tutorial video styles or rely heavily on manual input and setup, such as defining keybed and note regions, specifying video-specific colors, or manual calibration by the user. In contrast, LumaChords is designed to detect and extract the necessary information on its own, without requiring additional manual input or calibration.

So, as an engineer, I decided to combine my technical and musical skills together, then this project was born.

At the technical side, a deep learning segmentation model could have been used, but I ***intentionally*** put myself into a challenge with the following requirements and constraints:

* The end-to-end application must have both a GUI and a headless terminal user interface,
* The GUI must provide two modes: a basic mode that minimizes technical detail and lets the user focus on the outcome, and an advanced mode that exposes the detection process from multiple perspectives,  
* Have multiple video input backends: FFmpeg and OpenCV. OpenCV is used when FFmpeg is not located on the system as a fallback,
* Have multiple note detection strategies: one for videos with textured backgrounds, and another for videos with sparse or near-monotonic backgrounds,
* Have multiple hands detection strategies: one for videos with visible human hands, and another for videos with colored keys,
* Use only rule-based classical computer vision techniques, ideally no machine learning methods at all (only K-Means Clustering is used in FFT Analysis),
* Avoid any GPU requirement for computation, except for video file processing and GUI rendering,
* Just rely on Numpy's vectorized calculation operations and CPU SIMD capabilities wherever possible,
* Provide (as much as) smooth, low-latency and responsive GUI using OpenGL.

## 📘 **DOCUMENTATION**

For technical details of the multi-stage pipeline of this project, you can check out [docs/METHODOLOGY.md](docs/METHODOLOGY.md) file.

## 📦 **INSTALLATION and BUILDING**

This project contains mutliple styles of installation methods available.

### Installing FFmpeg dependency (optional but recommended)

FFmpeg is a third-party multimedia framework that LumaChords optionally uses to process multimedia files. FFmpeg is not included and distributed with LumaChords. LumaChords does not distribute FFmpeg binaries. Users (you) are responsible for installing FFmpeg independently according to their requirements.

LumaChords can operate using included OpenCV video backend, but having FFmpeg is recommended. You can follow instructions at [Installing FFmpeg](https://adalkiran.github.io/lumachords/installing-ffmpeg.html) document for more information.

### From PyPI - Installation

If you want to install LumaChords as a Python package;

***(Recommended)***

You can install it in a virtual environment as follows:

* Ensure ```uv``` tool is installed. If not, check out Astral's documentation at [Installing uv](https://docs.astral.sh/uv/getting-started/installation/),

* Execute the following:

```sh
$ uv tool install lumachords --python 3.12
$ lumachords # to run the application
```

***(Not recommended)***

Or, you can install it directly calling:

```sh
$ pip install lumachords
$ lumachords # to run the application
```

### From Source - Running via ```uv```

* Ensure ```uv``` tool is installed. If not, check out Astral's documentation at [Installing uv](https://docs.astral.sh/uv/getting-started/installation/),

* Clone this repo and enter into this directory,

On Linux/MacOS:

```sh
$ cd lumachords
$ uv venv
$ source .venv/bin/activate
$ uv sync
$ uv run lumachords
```

On Microsoft Windows:

```sh
$ cd lumachords
$ uv venv
$ .venv\Scripts\activate
$ uv sync
$ uv run lumachords
```

### From Releases

If you prefer using prebuilt executables, you can download them from the project's [GitHub Releases](https://github.com/adalkiran/lumachords/releases) page.

* Open the latest release and check the **Assets** section.
* Download the archive that matches your operating system and CPU architecture.
* Extract the downloaded archive to a directory of your choice.
* Run the included executable file from the extracted directory.

The published executables are currently not code-signed. Depending on your operating system and security configuration, they may be blocked, require additional confirmation, or may not run without manual allowance. If that happens, use one of the other available installation methods.

Release artifacts are platform-specific, and some platforms may not be available for every version. For current platform-related limitations, see [KNOWN-ISSUES.md](KNOWN-ISSUES.md).

### From Source - Building executables via *PyInstaller*

* Clone this repo and enter into this directory,
* Run the following script and check out the "dist" directory in the project directory. It will generate executable files only for the OS/platform on your machine.

```sh
$ cd lumachords
$ ./scripts/build.sh
```

### From Source - Development Mode

* For VSCode and its counterparts, this repository contains debug mode launching definitions in .vscode/launch.json file.

* Ensure ```uv``` tool is installed. If not, check out Astral's documentation at [Installing uv](https://docs.astral.sh/uv/getting-started/installation/),

* Clone this repo and enter into this directory,

On Linux/MacOS:

```sh
$ cd lumachords
$ uv venv
$ source .venv/bin/activate
$ uv sync
```

On Microsoft Windows:

```sh
$ cd lumachords
$ uv venv
$ .venv\Scripts\activate
$ uv sync
```

* Start debugging in your IDE.

### From Source - Experiment/Notebook Mode

* This repository contains [experiment.ipynb](./experiment.ipynb) and [continuity_experiment.ipynb](./continuity_experiment.ipynb) Python notebooks containing experiments on one frame image or series of frame images at Python notebook environment. While development of this project, the Python Notebook extension of VSCode is used.
* The algorithms in the whole pipeline were developed and tested using these notebooks.
* You can find [experiment_samples/](./experiment_samples/) and [tests/data/](./tests/data/) directories containing sample frames to be used in experiments, in categorized form.
* The ```continuity``` subfolders of them contain consecutive frame series to experiment on time sequence.

## 🧱 **ASSUMPTIONS and LIMITATIONS**

This project does not claim to provide fully successful or fully accurate results, and no such guarantee is given. While development, LumaChords was tested on various styles and formats of piano tutorial videos. Its multi-stage pipeline includes specialized strategies to handle many of these differences, but it is normal to output weird outputs or no output for some video inputs. For example, LumaChords won’t be able to detect notes correctly in videos that already contain sheet music overlays or fixed text at the top of the video.

## 🚧 **KNOWN ISSUES**

For details of current known problems, limitations, and workarounds, see [KNOWN-ISSUES.md](KNOWN-ISSUES.md).

The current headlines are:

- Detection Failures and Unexpected Outputs
- Staccato detection is missing
- Auto Timing is experimental
- MacOS Desktop: Latency while the first run
- Linux Desktop: Exit Hang
- Linux and MacOS: File Dialog Focus
- Windows: Save Dialog Filter Limitation
- Windows ARM64: Release Availability

## ⭐ **CONTRIBUTING and SUPPORTING the PROJECT**

You are welcome to [create issues](https://github.com/adalkiran/lumachords/issues/new) to report any bugs or problems you encounter.

If you liked and found my project helpful and valuable, I would greatly appreciate it if you could give the repo a star ⭐ on GitHub. Your support and feedback not only help the project improve and grow but also contribute to reaching a wider audience within the community. Additionally, it motivates me to create even more innovative projects in the future.

## 📖 **REFERENCES**

I want to thank to contributors of the awesome libraries and tools which LumaChords depends on:

### Direct Dependencies

| Dependency | License | PyPI | Source | Website / Notes |
| --- | --- | --- | --- | --- |
| crossfiledialog | LGPL-3 | [PyPI](https://pypi.org/project/crossfiledialog/) | [Source](https://github.com/maikelwever/crossfiledialog/) | - |
| ffmpeg-python | Apache-2.0 | [PyPI](https://pypi.org/project/ffmpeg-python/) | [Source](https://github.com/kkroening/ffmpeg-python) | - |
| matplotlib | PSF | [PyPI](https://pypi.org/project/matplotlib/) | [Source](https://github.com/matplotlib/matplotlib) | [Website](https://matplotlib.org/) |
| mido | MIT | [PyPI](https://pypi.org/project/mido/) | [Source](https://github.com/mido/mido) | - |
| numpy | BSD-3-Clause | [PyPI](https://pypi.org/project/numpy/) | [Source](https://github.com/numpy/numpy) | - |
| opencv-python-headless | MIT | [PyPI](https://pypi.org/project/opencv-python-headless/) | [Source](https://github.com/opencv/opencv-python) | - |
| pygame | LGPL-2.1 | [PyPI](https://pypi.org/project/pygame/) | [Source](https://github.com/pygame/pygame) | [Website](https://www.pygame.org/news) |
| pygame-menu | MIT | [PyPI](https://pypi.org/project/pygame-menu/) | [Source](https://github.com/ppizarror/pygame-menu) | [Website](https://pygame-menu.readthedocs.io/en/latest/) |
| pyopengl | BSD | [PyPI](https://pypi.org/project/PyOpenGL/) | [Source](https://github.com/mcfletch/pyopengl) | - |
| pyopengl-accelerate | BSD | [PyPI](https://pypi.org/project/PyOpenGL-accelerate/) | [Source](https://github.com/mcfletch/pyopengl) | - |
| pyfluidsynth | LGPL-2.1 | [PyPI](https://pypi.org/project/pyfluidsynth/) | [Source](https://github.com/nwhitehead/pyfluidsynth) | - |
| platformdirs | MIT | [PyPI](https://pypi.org/project/platformdirs/) | [Source](https://github.com/tox-dev/platformdirs) | - |
| python-rtmidi | MIT | [PyPI](https://pypi.org/project/python-rtmidi/) | [Source](https://github.com/SpotlightKid/python-rtmidi) | - |
| pyvips | MIT | [PyPI](https://pypi.org/project/pyvips/) | [Source](https://github.com/libvips/pyvips) | [3rd Party Info](https://github.com/kleisauke/libvips-packaging/blob/main/THIRD-PARTY-NOTICES.md) |
| tqdm | MPL-2.0, MIT | [PyPI](https://pypi.org/project/tqdm/) | [Source](https://github.com/tqdm/tqdm) | [Website](https://tqdm.github.io/) |
| yt-dlp | Unlicense | [PyPI](https://pypi.org/project/yt-dlp/) | [Source](https://github.com/yt-dlp/yt-dlp) | - |
| verovio | LGPL-3.0 | [PyPI](https://pypi.org/project/verovio/) | [Source](https://github.com/rism-digital/verovio) | [Website](https://www.verovio.org) |
| pywin32 | PSF | [PyPI](https://pypi.org/project/pywin32/) | [Source](https://github.com/mhammond/pywin32) | - |

### Development Dependencies

These dependencies are used for development and building only. They are not included in distributed artifacts.

| Dependency | License | PyPI | Source | Website |
| --- | --- | --- | --- | --- |
| pyinstaller | GPL-2.0 | [PyPI](https://pypi.org/project/pyinstaller/) | [Source](https://github.com/pyinstaller/pyinstaller) | [Website](https://pyinstaller.org/en/stable/) |
| pytest | MIT | [PyPI](https://pypi.org/project/pytest/) | [Source](https://github.com/pytest-dev/pytest) | [Website](https://docs.pytest.org/en/latest/) |
| uv | Apache-2.0, MIT | - | [Source](https://github.com/astral-sh/uv) | [Website](https://docs.astral.sh/uv/) |

### External Tools and Resources

| Name | License | Source | Website / Notes |
| --- | --- | --- | --- |
| FFmpeg | LGPL-2.1+/GPL-2.0+ | [Source](https://github.com/FFmpeg/FFmpeg) | [Website](https://ffmpeg.org/).<br>LGPL builds are included indirectly as an OpenCV sub-dependency. GPL builds are not distributed by this project, but can still be used if installed separately on the user's machine and available via command-line pipes. |
| UprightPianoKW Soundfont | CC0 1.0 Universal | [Source](https://freepats.zenvoid.org/Piano/UprightPianoKW/UprightPianoKW-small-bright-SF2-20190703.7z) | [Website](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)<br>This soundfont file is used when no system default MIDI output device found. |

### Experiment Sample Frame Images

The input sample images are captured/extracted from third-party YouTube videos for limited research/testing use purposes only. See [experiment_samples/EXPERIMENT_SOURCES.txt](experiment_samples/EXPERIMENT_SOURCES.txt) and [tests/data/TEST_SOURCES.txt](tests/data/TEST_SOURCES.txt) for ownership and limited-use notes.

## 📜 **LICENSE**

LumaChords is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text, also see [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) for third-party licenses.
