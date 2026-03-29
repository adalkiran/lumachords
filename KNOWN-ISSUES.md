# Known Issues

### Detection Failures and Unexpected Outputs

This project is experimental and does not guarantee fully accurate results for every input video. LumaChords was tested on multiple piano tutorial styles and includes specialized handling for difficult cases, but some videos may still produce incorrect output or fail entirely. In particular, note detection is not expected to work reliably when the video already contains sheet music overlays or fixed text near the top of the frame.

### Staccato detection is missing

Currently staccato articulation is not represented with staccato sign, instead of this, these are engraved with a short-time note and a following rest.

### Auto Timing is experimental

Currently the "Auto Timing (experimental)" menu option appears in the starting menu with OFF position as default. Because it mostly lacks the estimation of BPM, then the time signature. It's left as is, but it should be reconsidered further.

## Platform-Specific Issues

### MacOS Desktop: Latency while the first run

- After the first fresh installation of the project, OpenCV (cv2) and other .so and .dylib dependency files are validated and scanned by MacOS if the application is on MacOS. This may take some time.
- This issue doesn't occur while the next runs.
- This issue doesn't occur if you run the executable created via PyInstaller.

### Linux Desktop: Exit Hang

On Linux desktop environments such as Ubuntu, the application may appear to close but the process can remain running in the terminal.

- The terminal may print `Finished.` and `Quitting...`, and the GUI window may disappear, while the process does not exit.
- Pressing `Ctrl+C` after shutdown may not terminate it, although `Ctrl+C` during video processing behaves as expected.
- This issue was not observed in Ubuntu Docker container runs in headless mode.
- The underlying cause appears to be `pygame.quit()` hanging, specifically through `pygame.mixer.quit()` when a Linux audio driver is active.
- Wrapping the shutdown path with `await asyncio.wait_for(...)` did not resolve the issue.
- Reproduced with `uv run main.py --demo`.

### Linux and MacOS: File Dialog Focus

On Linux and MacOS desktop environments, file dialogs may not receive focus automatically. When this happens, the main application window can appear unresponsive until the dialog is brought to the foreground manually.

### Windows: Save Dialog Filter Limitation

On Windows, save dialogs currently do not show a file type filter. This limitation comes from the `crossfiledialog` library, which does not provide that feature for cross-platform compatibility.

### Windows ARM64: Release Availability

Windows ARM64 builds are currently not provided in project releases because `opencv-python-headless` and `opencv-python` do not publish prebuilt wheels for that platform. See [opencv-python issue #1092](https://github.com/opencv/opencv-python/issues/1092).
