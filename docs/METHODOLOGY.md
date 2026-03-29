# Methodology

This document describes the algorithmic pipeline used by LumaChords to convert Synthesia-style piano tutorial videos into note events, musical notation, MIDI, MEI, and video output with notation overlay.

The project is ***intentionally*** built around classical computer vision and rule-based processing. It does not rely on deep learning models. The only machine learning component in the detection path is a small K-Means clustering step used during keybed detection.

## Design Goals and Constraints

The pipeline was designed around the following requirements and constraints:

* The end-to-end application must have both a GUI and a headless terminal user interface,
* The GUI must provide two modes: a basic mode that minimizes technical detail and lets the user focus on the outcome, and an advanced mode that exposes the detection process from multiple perspectives,  
* Have multiple video input backends: FFmpeg and OpenCV. OpenCV is used when FFmpeg is not located on the system as a fallback,
* Have multiple note detection strategies: one for videos with textured backgrounds, and another for videos with sparse or near-monotonic backgrounds,
* Have multiple hands detection strategies: one for videos with visible human hands, and another for videos with colored keys
* Use only rule-based classical computer vision techniques, ideally no machine learning methods at all (only K-Means Clustering is used in FFT Analysis),
* Avoid any GPU requirement for computation, except for video file processing and GUI rendering,
* Just rely on Numpy's vectorized calculation operations and CPU SIMD capabilities wherever possible,
* Provide (as much as) smooth, low-latency and responsive GUI using OpenGL.

## Dictionary

Some of commonly used terms in the documentation and the code.

- `pts`: The frame index in the processing frame stream. Stands for "Presentation Time Stamp"
- `pts_time`: The time derived from the actual processing FPS as seconds
- `BGR`: Due to OpenCV uses BGR (blue-green-red) as default instead of RGB (red-green-blue), BGR form is choosed as the common form of colored image representation.  
- `LAB`: A colorspace that contains components: Lightness, A*, B*. The "L" channel (Luminosity/Lightness/Brightness) of this space is used in this project.
- `HSV`: A colorspace that contains components: Hue, Saturation, Value. Hue represents values in color spectrum, each value corresponds to different color range. Hue channel is used to detect colored keys in the videos. Saturation and Value channels are used to filter out unnecessary pixels from images.
- `Luma`: In computer vision, mostly grayscale form of images are used. But because of the nature of used video frames in this project, the "brightness" is more important feature of images. So instead of using "grayscale", it is chosen to use L (Luminosity/Lightness/Brightness) channel of LAB form of images. The project name "LumaChords" comes from this.
- `keybed`: The supported piano tutorial video styles consist of two vertical parts: Keybed at the bottom, remaining top part is note rain area. Keybed term is used for location of the visible keyboard on the frames and also contains validated locations of each white and black keys with corresponding pitches, note names, octaves, MIDI numbers and X coordinates of edges
- `note rain`: The falling boxes each representing a corresponding piano key/pitch/note name/MIDI number, times of note on/off events
- `im_crop_bgr/im_crop_luma`: The note-rain region above the keyboard with BGR and LUMA forms.
- `im_crop_keybed_bgr/im_crop_keybed_luma`: An extended keybed crop used for hand detection with BGR and LUMA forms. 
The note-rain crop is converted to Luminosity (Luma) from LAB space and lightly blurred. This representation is used for background analysis and note detection.
- `Fourier/FFT Analysis`: Uses Fast Fourier Transform to perform frequency analysis and clustering image rows in this project. For more information, see [Wikipedia](https://en.wikipedia.org/wiki/Fast_Fourier_transform).
- `SPARSE background type`: Piano tutorial videos mostly have dark-ish/black-ish backgrounds. The backgrounds of these type of videos doesn't contain an explicit texture.
- `TEXTURED background type`: Some of piano tutorial videos have more complex backgrounds with inconsistent colors like landscape pictures.


## End-to-End Pipeline

At runtime, the system processes the input video in two major phases:

1. Read the input video as frames in a fixed frame rate: 10 fps,
2. Detect and validate the piano keybed from one sufficiently clear frame, try until finding the appropriate frame,
3. Process consecutive frames to detect falling-note boxes, track them over time with a "poor man's object tracker", and convert them into musical events (note on/off MIDI events)
4. While the detection of falling-note boxes phase, the accumulated note events are transformed into:
    - Real-time note playback (using Fluidsynth or MIDI output port)
    - MEI XML format to generate staff notation
    - Staff notation rendered as image (to blend as overlay on the video frame)
5. At any time of processing or after finishing, the collected musical events data can be exported as MIDI file, MEI file, and video file with notation staff overlay.

## 1. Video Input and Frame Preparation

LumaChords reads frames through either an FFmpeg-based backend or an OpenCV fallback backend. Frames are sampled at a configured processing FPS and may be height-limited for efficiency. The resulting frames are processed as an iterable, and each iteration feeds the pipeline.

The pipeline is stateful. Keybed detection is performed first, then reused for all later frames.

## 2. Keybed Detection

Keybed detection is the foundation of the entire system. It determines:

- Where the visible keyboard is located vertically in the frame
- The horizontal edges of individual keys
- The pitches/MIDI numbers associated with those key slots
- Approximate white-key and black-key widths, later reused by note detection heuristics

### 2.1 Grayscale Preparation

The input frame is converted to grayscale and enhanced with CLAHE. This improves local contrast before frequency analysis.

### 2.2 Row-wise FFT Analysis

Some of projects that have similar goal use template-matching. But it didn't work well for the sample input videos used during experiments, mostly lighting conditions and keybed size differences. Instead of this, I decided to use frequency analysis/Fourier Transform to get more robust results.

The key insight is that piano keyboards create a strong horizontal texture pattern because of repeated vertical key boundaries. To capture that, LumaChords computes a row-wise FFT over a Scharr-x-transformed image.

For each image row, the detector derives:

- Dominant spatial frequency
- Peakiness ratio of the dominant frequency relative to the row spectrum

These values are normalized and used as per-row features. The normalization and converting to ratio operations are performed to generate independent and individually comparable features.

### 2.3 K-Means Row Clustering

Rows are clustered into two groups with K-Means. The cluster with:

- Lower dominant frequency
- Higher peakiness

is treated as the keyboard-like region candidate. This gives a rough row mask for the keybed.

### 2.4 Keybed Segment Selection

The resulting rough mask may contain a correct keybed region as separated sections, because of the noise in the image. So the keybed region may contain multiple thin opposite labels.

Because of this, the rough mask is post-prpocessed and converted into contiguous row segments. Segments are scored using:

- Vertical span
- Average row spectral strength

Then, the post-processing result have possible keybed regions. Additional validation is performed to check and reject inappropriate candidates, including:

- Keybed region too high in the image
- Selected rows having too little keyboard-like frequency content
- Segment too tall for frame height

The best valid segment becomes the keybed bounds. But the resulting and validated region is still "a candidate".

### 2.5 White-Key Boundary Extraction

After the keybed is located, the detector crops the keyboard image part and analyzes two subregions:

- Upper keyboard portion for black-key-related boundaries (1/3 of height at the top). This portion is normalized.
- Lower keyboard portion for clearer white-key edge structure (2/3 of height at the bottom). This portion is not normalized.

Column-wise vertical edge energy is accumulated, smoothed, normalized, and thresholded. Contiguous high-energy runs are interpreted as candidate key boundaries.

The detector then:

- Removes boundary outliers
- Estimates the default white-key width using a robust median-based method
- Splits merged white keys when a gap is likely hiding multiple keys (the detector might not detect some separator edges from the image)
- Reconstructs a corrected white-key grid

### 2.6 Black-Key Recovery and Note Naming

Black-key candidates are associated with neighboring white-key intervals. Before black-key validation, the detector validates and corrects the full-white-key edges (for e.g., between white E-white F keys and between white B-white C keys, there is no black key, they are called in this project as "*full-white-key edges*").

After full-white-key correction, there might be missing (not detected/marked) full-white-key edges. So, the detector performs a scoring on the all of full-white-key edges to name the notes at the full-white-key markers (which are E-F, which are B-C notes).

 The detector uses the expected piano octave pattern to infer note labels and correct missing or inconsistent full-white-key markers. Think of the piano key arrangement:

```
 (full-white-key) white C | black C# | white D | black D# | white E |(full-white-key) white F | black F# | white G | black G# | white A | black A# | white B | ***(full-white-key)*** white C | ...
```

The pattern is for a keybed with the first full-white-key is C:

```
(full-white-key edge of C) + 3 white keys + (full-white-key edge of F) + 4 white keys + (full-white-key edge of C) ...
```

The pattern is for a keybed with the first full-white-key is F:

```
(full-white-key edge of F) + 4 white keys + (full-white-key edge of C) + 3 white keys + (full-white-key edge of F) ...
```

From that reconstructed layout, it derives:

- Ordered white and black key slots
- Note names
- Octave numbers
- MIDI numbers

### 2.7 Keybed Validation

The recovered key order is validated against the expected chromatic sequence. Width outliers and note-order mismatches are used to reject bad detections.

Only after this phase succeeds does the system move to note tracking.

## 3. Note-Rain Region Preparation

Once the keybed is known and validated, each later frame is split into two working regions vertically:

- The note-rain region above the keyboard (im_crop_bgr/im_crop_luma)
- An extended keybed crop used for hand detection (im_crop_keybed_bgr/im_crop_keybed_luma)

The note-rain crop is converted to Luma from LAB space and lightly blurred. This representation is used for background analysis and note detection.

## 4. Background-Type Classification

Tutorial videos do not all use the same visual style. Some have a relatively clean, sparse background around the falling notes, while others contain textured smoke, gradients, or overlays.

LumaChords classifies the note-rain crop into one of two modes:

- `SPARSE`: Mostly dark-ish and black-ish background, consistent
- `TEXTURED`: Background is more complex or inconsistent like a landscape picture

This decision determines which detection strategy is used for note boxes.

For sparse backgrounds, the dominant background luma is estimated and used to suppress background pixels before detection.

## 5. Hand and Colored-Key Range Detection

Piano is an instrument that is mostly played with using both hands. So that, music sheets for pianos mostly have 2 staves (treble-clef and bass-clef). In a MIDI events data there's no information about this separation: which notes are played with the left hand and which with the right hand.

Because of this requirement, the system analyzes the keyboard region to estimate the horizontal span of visible hands or colored key ranges. This information is used later for staff assignment and notation splitting.

The hand information exists in the piano tutorial videos in two types:

- The content creator films the video with their real piano keys and hands visible at the bottom of the video.
- The video is generated completely by computer, so the keyboard section is computer generated and the pressed keys are emphasized by coloring the pressed keys.

### 5.1 Skin-Based Hand Detection

For videos containing human-hands, LumaChords performs a YCrCb skin-color segmentation with component analysis. The result is a cleaned hand mask over the keyboard area.

### 5.2 Colored-Key Detection

LumaChords also supports computer-generated tutorial videos where no real hands are visible and pressed keys are indicated by colored highlights on the keyboard. In this case, the keyboard crop is converted into HSV space, and only sufficiently saturated, non-dark pixels are kept so the colored regions stand out from the normal keybed.

The detector then analyzes hue values inside those regions to find dominant hue ranges and score candidate colored-key spans. Nearby hue groups are merged, and large hue differences are used to separate up to two hand-like color groups. A lightweight color tracker keeps these hue identities stable across frames, even when only one side is visible temporarily. The final colored spans are then mapped to keyboard note ranges and used as a proxy feature for left-hand and right-hand separation.

### 5.3 Range Projection to Keyboard

The hand mask is reduced column-wise into contiguous x-ranges. Each boundary is mapped onto the nearest key slot, producing per-hand note ranges such as:

- left hand: `A2` to `E3`
- right hand: `F4` to `F5`

These ranges help assign notes to treble/bass staves more musically than a fixed split alone.

## 6. Note-Rain Detection Strategies

The detector uses two different strategies depending on background type.

### 6.1 Strategy A: Textured Background

This is the more general path for visually noisy videos.

### Edge Extraction

The system computes Scharr gradients on the note-rain luma image in both axes:

- vertical edges for note box left/right edges
- horizontal edges for note box top/bottom edges

Thresholds are chosen using a `k * sigma` rule over gradient magnitude. Morphological filtering keeps only sufficiently long, thin components with appropriate orientation.

### Line Grouping and Pairing

Both vertical and horizontal start/end edges of a rectangle can be distinguishable thanks to positivity/negativity of the Scharr gradients.

Detected vertical start/end lines are paired into note box candidates using constraints on:

- Horizontal width
- Vertical overlap
- Continuity
- Proximity
- Pverlap preference against stronger competing pairs

Horizontal lines are then used to:

- Snap top and bottom edges
- Split vertically merged blobs
- Reject blocked or implausible candidates

The result is a set of note-box candidates consistent with expected piano-key widths.

### 6.2 Strategy B: Sparse Background

When the background is nearly binary after thresholding, LumaChords uses a simpler blob-first approach.

### Binary Cleanup

The sparse-background path:

- Thresholds the note-rain crop around the dominant background value
- Removes thin long line noise
- Performs morphology to preserve note blocks

### Component Analysis and Refinement

Connected components become note box candidates. These are refined with local Scharr-based boundary searches to tighten their box edges.

Further filters reject candidates that:

- Are too small
- Fit too many keyboard slots
- Have poor brightness density
- Are largely covered by stronger overlapping boxes

This strategy is faster and more direct when the note shapes are already visually separated from the background.

## 7. Mapping Note Boxes to Keys

Each note box is assigned to the most likely piano key slot using the keybed geometry detected earlier.

The mapping is based on horizontal overlap between x-ranges of boxes and key slots.

Ambiguous fits are allowed only under limited conditions, such as black/white adjacency. This prevents wide false detections from being interpreted as multiple simultaneous keys.

## 8. Temporal Tracking and Event Extraction

Per-frame detection alone is not enough. LumaChords must preserve identity across frames and estimate when a falling note actually reaches the play point.

### 8.1 Tracking Band and Play Line

Two horizontal reference regions are derived from the detected keyboard position:

- A tracking band above the keybed, where note boxes are matched frame to frame
- A play line closer to the keyboard, where note activation is declared

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                         ┃
┃╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ tracking_y0 ┃
┃                                         ┃
┃                                         ┃
┃───────────────────────────────── play_y ┃
┃╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ tracking_y1 ┃
┃                                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

These offsets are relative to the detected keybed height and are configurable.

The note on/off events are declared while movements of valid boxes incoming or outgoing through the play_y line. The moving boxes are validated using tracking_y0 and tracking_y1, so some of moving boxes could be perceived as "noise" and ingored during the process. 

### 8.2 Box Matching Across Frames

Current-frame boxes are matched to previous-frame boxes with an order-preserving tracker. Matching considers:

- Key index (index of the matching keybed slot)
- Box center position
- Vertical and horizontal displacement

The tracker preserves left-to-right ordering to reduce identity swaps in dense passages.

### 8.3 Velocity Estimation

For matched boxes, the tracker estimates vertical velocity in pixels per frame. A robust median is computed from recent matches, with outlier suppression. Once enough evidence accumulates, a velocity consensus is established, otherwise the found velocity values are used as temporary values.

This consensus serves two purposes:

- It stabilizes temporal tracking
- It estimates the lag between the visible play line and the actual keyboard edge

LumaChords detects the note on/off events at the play_y position, but the notes are played further when the boxes reach out to the top of keybed region. The detection is not done at the the top of keybed region because this region is the most noisy part of the video: most of videos contain lighting and smoke animations there.

### 8.4 Occlusion Recovery

If a note temporarily disappears because of overlap or partial occlusion, the tracker can extrapolate its next position forward rather than immediately dropping it. This is important for dense note rain and short occlusions.

Actually there's no occlusion in reality, but the note rain detector fails on areas with smoke animations, so in these type of cases, LumaChords estimates the next position of the box previously detected using the calculated velocity.

### 8.5 Note On/Off Generation

Tracked boxes are turned into note events when they cross the play line:

- Entering the play scope generates note-on events
- Leaving the play scope generates note-off events

Each event carries:

- Persistent box ID
- Mapped MIDI number
- Sub-frame time offset relative to the current frame

The time offset is derived from the estimated box velocity, which improves timing resolution beyond raw frame boundaries.

## 9. MIDI Event Accumulation

The `MidiTracker` converts raw per-frame events into stable musical note state.

It maintains:

- Currently active MIDI notes (notes that in `note-on` state)
- Note-on timestamps
- Completed `(midi_num, on_time, off_time)` pairs
- Recent snapshots for playback and GUI rewind logic
- Hand-range history over time

This stage also removes invalid duplicate events, such as repeated note-on for an already active note.

## 10. Timing and Meter Estimation (experimental)

LumaChords can automatically estimate tempo and time signature from the extracted note events. But this feature is optional and experimental now, so it may output wrong results.

## 11. Staff Assignment and Notation Construction

The notation layer transforms note event pairs into score structure.

### 11.1 Staff Assignment

Each note is assigned to a staff using:

- Recent hand ranges when available
- A configurable fallback split note otherwise

This allows the left hand/right hand staff split to follow actual hand placement rather than relying only on a fixed MIDI threshold.

### 11.2 Time to Tick Conversion

Note pair times in seconds are converted to absolute musical ticks using the current BPM, PPQ, and time signature.

Optional cropping removes leading silence from the notation timeline.

### 11.3 Cleanup and Quantization-Like Structuring

The notation builder then:

- Resolves overlapping notes within a staff
- Converts absolute ticks into bar/beat/division/tick (bbdt) positions
- Inserts rests to keep measures complete
- Splits notes that span bar boundaries with preserving ties across split segments
- Determines clefs per bar
- Groups beamable durations within accent-friendly windows

The result is not a transcription-grade music-sheet engraving engine, but it produces a practical and readable staff notation representation from the recovered note stream.

## 12. Rendering and Export

The final notation state is reused across several outputs.

### MIDI

Completed note event pairs are exported as a standard MIDI file with note-on/note-off events.

### MEI

The score structure is serialized into MEI, including measures, staves, layers, notes and rests, chords, ties, clef and meter definitions, and estimated tempo metadata

### Overlay Video

For video export, the notation image is re-rendered frame by frame and composited as an overlay synchronized to the detected note timeline.

### Real-Time Playback

During detection and live preview, note events can also be sent to a realtime MIDI output device (like a DAW) or FluidSynth synthesizer.

## 13. Pipeline Structure

A few architectural decisions are effective to the project:

- Keybed detection is done once, because key geometry is stable and expensive to rediscover every frame
- Note detection is adaptive, because different tutorial videos behave very differently in the note-rain region

## 14. Main Assumptions

The current pipeline works best when the video resembles a typical top-down or slightly angled tutorial layout where:

- The keyboard is clearly visible with all of its details in at least one frame
- The keybed appears the lower portion of the frame
- The keybed does not move significantly during the video
- Falling notes are mostly vertical and aligned to keys
- Note motion is downward with a stable velocity

## 15. Summary

LumaChords combines:

- FFT-based keybed location detection
- Rule-based key edge reconstruction
- Adaptive note-rain segmentation
- Temporal box tracking (with a "poor man's object tracker") with velocity estimation
- Note events to notation conversion
- Lightweight (and experimental) tempo and meter inference

to produce a full video-to-notation pipeline without using a deep learning transcription model.

That combination is what makes the project both practical for many Synthesia-style videos and technically interesting as an end-to-end classical computer vision system.
