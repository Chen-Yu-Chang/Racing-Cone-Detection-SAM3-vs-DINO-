# DINO vs SAM3 cone auto-labeling test

Goal: take one `.mcap` file, pull out a handful of real camera frames, run
both Grounding DINO and SAM3 on them with the same cone-color text prompts,
and compare which one labels cones better - as a first data point for
CMR perception's training-data-synthesis (auto-labeling) effort.

**Status: done.** This was run end-to-end on ~15 real frames from one mcap
file on Apple Silicon. See [Discovery](#discovery--findings) below for the
results and [How to run](#how-to-run) for the exact working commands.

## Why this runs on your own machine, not in a cloud session

Cloud dev sandboxes commonly have no GPU and network egress blocked to
Hugging Face, Colab, and Meta's model hosting - so model weights can't even
be downloaded, let alone run fast enough to matter. Apple Silicon Macs work
fine for a smoke test like this via PyTorch's MPS backend; a CMR lab GPU box
would be faster still if you want to scale this up later.

## What's in this folder

| File | Purpose |
|---|---|
| `extract_frames.py` | Pulls frames out of the `.mcap` file's camera topic |
| `run_grounding_dino.py` | Runs Grounding DINO (HuggingFace `transformers`) on the frames |
| `run_sam3.py` | Runs SAM3 (via the `sam3-mac` shim) on the frames |
| `compare_side_by_side.py` | Stitches original / DINO / SAM3 into one labeled panel per frame |
| `requirements.txt` | Python dependencies (see the pins/notes below - they matter) |

## Setup

These are the exact steps that worked, including the dependency issues we
actually hit and how we got past each one. SAM3 is a very new (Nov 2025),
Linux/CUDA-oriented research release, so most of the friction is
macOS-specific packaging problems, not anything wrong with the approach.

1. **Create a clean virtual environment.** Don't skip this - installing
   into system Python is what caused the first failure (a `pip install -e .`
   / `distutils` / `build_editable` error from an old system pip trying to
   build a modern package).
   ```
   python3 -m venv venv
   source venv/bin/activate
   python3 -m pip install --upgrade pip
   ```

2. **Install `setuptools<81` before anything else.** Recent `setuptools`
   versions dropped `pkg_resources`, which some of SAM3's dependencies still
   import directly - without this pin you'll hit
   `ModuleNotFoundError: No module named 'pkg_resources'` partway through
   the install.
   ```
   pip install "setuptools<81"
   ```

3. **Install requirements.txt.** Note the numpy/opencv pin already baked
   into `requirements.txt`: `opencv-python` 5.x requires `numpy>=2`, but
   `sam3` requires `numpy<2,>=1.26`. Pip's resolver will happily install a
   combination that satisfies neither if you're not careful (we saw the
   resolver warn about this in both directions before pinning it down).
   The working combination is `numpy<2,>=1.26.4` + `opencv-python<5`, both
   already reflected in `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```
   If you still see missing-module errors partway through (we hit this
   several times in sequence - `cv2`, `einops`, `triton`, `pycocotools`,
   etc., each appearing only after the previous one was fixed), just
   `pip install <missing package>` and re-run; `requirements.txt` already
   has the full final list so a fresh install shouldn't need this, but
   worth knowing why if it happens.

4. **Install `sam3-mac`, and install it from GitHub, not PyPI.** Its own
   README says `pip install sam3-mac`, but as of this writing it isn't
   actually published to PyPI (`pip install sam3-mac` fails with "no
   versions found") - install it from source instead:
   ```
   pip install "git+https://github.com/benreichman/sam3-mac.git"
   ```
   This exists specifically because SAM3's official code hard-requires
   `triton`, which has no macOS build (see facebookresearch/sam3 issues
   #179 and #154). `sam3-mac` is a community shim that monkey-patches
   around the triton import and auto-selects MPS/CPU. Its only limitation
   is that video tracking doesn't work - irrelevant here, since we're only
   doing per-frame text-prompted segmentation, not tracking across frames.
   `run_sam3.py` already imports `sam3_mac` (not `sam3` directly) for this
   reason.

5. **Accept SAM3's license and set an HF token.** `facebook/sam3` is a
   gated HuggingFace repo - without this step you'll get a 401 /
   `GatedRepoError` the first time `run_sam3.py` tries to load it, even
   though `grounding-dino-tiny` (used by `run_grounding_dino.py`) needs no
   auth at all.
   - Go to https://huggingface.co/facebook/sam3 while logged in and accept
     the license/agreement.
   - Create a token at https://huggingface.co/settings/tokens (read access
     is enough).
   - Export it before running:
     ```
     export HF_TOKEN=hf_your_token_here
     ```

At this point `pip list` should show `torch`, `transformers`, `sam3_mac`,
`opencv-python`, `mcap`, `rosbags`, and the rest of `requirements.txt`
installed cleanly with no resolver warnings.

## How to run

Run these in order, from inside this folder, with the venv activated and
`HF_TOKEN` exported (per setup step 5). Only the first command needs your
actual `.mcap` file path and camera topic - everything else uses the same
frames.

1. **Find your camera topic:**
   ```
   python extract_frames.py --mcap /path/to/your/session.mcap --list-topics
   ```
   Look for something like `/camera/image_raw` or `/camera/compressed` in
   the output. Worth checking with the rest of the team first in case
   there's already an internal script for this - "get frames out of our
   mcap files" is a common enough need that someone's probably solved it.

2. **Extract ~15 frames** (roughly 1/sec - tune `--every-n` to your
   camera's actual frame rate):
   ```
   python extract_frames.py --mcap /path/to/session.mcap --topic /your/camera/topic --every-n 30 --max-frames 15 --out frames/
   ```

3. **Run Grounding DINO:**
   ```
   python run_grounding_dino.py --frames frames/ --out out_dino/
   ```
   First run downloads ~170MB of weights (the "tiny" checkpoint). Swap
   `--model IDEA-Research/grounding-dino-base` for the larger, more
   accurate checkpoint if you want to compare that too.

4. **Run SAM3:**
   ```
   python run_sam3.py --frames frames/ --out out_sam3/
   ```
   If you get zero detections, try `--confidence-threshold 0.1` to see
   whether SAM3 is finding low-confidence matches that the default
   threshold (0.5) is just filtering out - the script prints every raw
   score per concept either way, so you can see this directly in the
   terminal output without re-running.

5. **Build the side-by-side comparison images:**
   ```
   python compare_side_by_side.py --frames frames/ --dino out_dino/ --sam3 out_sam3/ --out comparison/
   ```

6. **Open the `comparison/` folder** and flip through the images - each one
   is original | Grounding DINO | SAM3 side by side.

## Discovery / findings

This is based on the 15 comparison panels produced from real CMR cone
footage (a dusk/dawn parking-lot-style test area, two rows of cones -
orange near the foreground, then blue, then yellow/black - receding toward
the horizon under artificial lighting).

**Grounding DINO** draws bold, solid red boxes with a visible
`label score` text tag on each one. It's reliably confident on near- and
mid-field cones, but consistently boxes fewer cones per frame overall -
typically around 4 to 6 - and its detections thin out sharply past
mid-distance. Cones near the horizon are very often left completely
unboxed. Where it does detect a cone, the box is tight and the label is
easy to read at a glance.

**SAM3** (via the `sam3-mac` shim) fills each detected cone with a
translucent color tint that matches the concept it matched against (blue
tint for "blue cone", yellow for "yellow cone", orange for "orange cone"),
plus a thin outline and a small score label. Across the same 15 frames it
picked up a noticeably wider range of distances - including many small,
far-away cones near the horizon that DINO's panel left completely unboxed.
It also gets per-instance color classification "for free," since each
match comes from a specific color-word prompt rather than a single generic
"cone" class. The tradeoff is presentation: the thin tinted-fill style is
much less visually prominent than DINO's bold red boxes, especially on
small or low-contrast cones. This actually caused an early false alarm
during setup - it briefly looked like SAM3 "wasn't labeling anything," when
what had actually happened was a drawing bug in `overlay_masks()` (an
`ImageDraw.Draw` context bound to an image object that had already been
reassigned by the mask-compositing loop, so boxes/labels were being drawn
onto a discarded image). Once that was fixed to composite all mask fills
first and only then draw boxes/labels onto the final image, SAM3's
detections rendered correctly and turned out to be doing real work all
along.

**Net read from this smoke test:** SAM3 appears to find substantially more
of the small/distant cones that Grounding DINO misses, at some cost to how
immediately legible its output is on a quick visual scan. DINO's smaller
set of detections is easier to trust at a glance because of the bold
labeling, but it's leaving real cones - especially far-field ones, which
matter a lot for a driverless car's planning horizon - undetected. This
points toward SAM3 being the stronger candidate for auto-labeling coverage,
though it's worth deliberately checking mask tightness (does the
segmentation hug the actual cone silhouette, or bleed into neighbors/
background?) and false-positive rate before committing to it, since this
test was optimized for "did it find the cone" rather than a full precision/
recall pass.

### What to actually judge if you look through the panels yourself

- **False positives**: did either model label something that isn't a cone?
  This is the most expensive kind of error - it silently poisons training
  data if nobody catches it.
- **Missed cones**: cones neither model caught, especially far-field ones,
  mark where you'll still need human-labeled data regardless of which
  model you pick.
- **Color accuracy**: does "blue cone" vs "yellow cone" actually match
  reality, or does either model mix them up under this lighting?
- **Box/mask tightness**: for SAM3 specifically, does the mask hug the
  cone's real silhouette or bleed into the background/neighboring cones?
- **Where they disagree**: frames where DINO and SAM3 disagree are exactly
  the ones worth a closer look - that disagreement is a built-in signal for
  which frames need human review before going into a training set.

## Part 2: Auto-labeling pipeline (SAM3 -> YOLO26s)

Part 1 compared Grounding DINO and SAM3 on real cone frames. This part uses SAM3 as an automatic labeler, trains YOLO26s on its output, and adds tools for the frames where SAM3 fails (night, glare, motion blur).

### Where this fits in the team architecture

```
Camera frames (mcap)
      |
      v
 Dataset building  -- SAM3 auto-labels + previous-frame check + enhancement
      |                + frame-quality report (what a human must review)
      v
 Model training    -- YOLO26s, standard supervised training
      |
      v
 Model evaluation  -- precision / recall / mAP, per condition (night vs day)
      |
      +--> refine the dataset (hand-label the hard frames, add data) --> back to dataset building
```

"Loss feedback" here means ordinary supervised training. Ultralytics computes the loss between YOLO's predictions and the SAM3-derived labels every batch and updates the weights. There is no separate feedback code.

### Files

| File | Purpose |
|---|---|
| `autolabel_sam3_temporal.py` | **Main labeling script.** SAM3 on consecutive frames, with enhanced second look, previous-frame tracking, color memory, frame-quality report. Writes a YOLO dataset. |
| `temporal_utils.py` | The logic it uses: enhancement, optical flow, tracker with color voting, frame-quality flags, review ranking, block train/val split. |
| `rerank_review.py` | Rebuilds the review list from an existing `frame_quality.csv`. No SAM3 re-run needed. |
| `train_yolo26s.py` | Trains YOLO26s on the dataset (thin wrapper around Ultralytics). |
| `test_temporal.py` | 21 self-checks. Need no SAM3 and no GPU (a fake detector stands in for SAM3). |
| `extract_frames.py` | Cuts frames from the mcap. Use `--every-n 2` or `3` for this pipeline. |
| `run_grounding_dino.py`, `run_sam3.py`, `compare_side_by_side.py` | Part 1 comparison and benchmark timing. Finished, not part of the training pipeline. |
| `autolabel_sam3_to_yolo.py` | **Retired.** Single-frame labeler, replaced by `autolabel_sam3_temporal.py` (which behaves like it with `--temporal-concepts none --no-enhance`, apart from the split and score flags). |

### How to run

1. **Extract consecutive frames.** The previous-frame check needs frames close in time (the first test used 1 frame per second, which is too sparse).
   ```
   python extract_frames.py --mcap <file.mcap> --topic <camera topic> --every-n 3 --max-frames 150 --out frames_seq/
   ```
2. **Self-test (optional, fast).**
   ```
   python test_temporal.py
   ```
3. **Label.**
   ```
   python autolabel_sam3_temporal.py --frames frames_seq/ --out dataset_seq/ --preview preview_seq/ --export-review review_seq/
   ```
4. **Review.** Open `dataset_seq/review_first.csv` and the preview images. Fix or remove bad frames (see "Reviewing the labels").
5. **Train.**
   ```
   python train_yolo26s.py --data dataset_seq/data.yaml --epochs 30
   ```
6. **Re-rank without re-labeling** (to change thresholds or the ranking):
   ```
   python rerank_review.py --dataset dataset_seq/ --min-priority 2 --export-review review_seq/
   ```

Process each recording on its own, with its own `--out`. Frames are treated as one continuous video in filename order, so night and day recordings must not share a folder. Day and night datasets can be merged afterward; a merge script is not written yet.

### How the labeling works

Per frame, in order:

1. **Motion since the previous frame.** Dense optical flow (Farneback). If the typical scene motion exceeds `--reset-px` (default 60 px), the view jumped, so all tracks are forgotten.
2. **Pass 1:** SAM3 on the original frame for every color (blue, yellow, orange).
3. **Pass 2 (enhanced second look):** only if a tracked cone was lost, or no confident detection exists for that color, SAM3 is run again on an enhanced copy (local contrast, blue boost, sharpening). Boxes from both passes are merged. The enhanced copy is used for detection only. The image saved in the dataset is always the untouched original.
4. **Tracker.** All colors share one tracker, so one physical cone is one track:
   - A detection with score >= `--hi-score` (0.5) is accepted and starts or extends a track.
   - A faint detection (`--lo-score` 0.25 to `--hi-score`) is accepted only if it lines up with a confirmed track (seen in at least `--min-hits` frames).
   - A confirmed cone that nothing detected is carried forward by the measured motion for at most `--max-gap` frames (default 3). These boxes are marked **propagated**.
5. **Color memory.** Each confident detection votes for its color. Faint detections and propagated boxes take the track's voted color, so a smeared blue cone called "yellow" at 0.3 stays blue (marked `relabeled`). A confident detection that disagrees with the vote is kept but marked `color_conflict`. Color is not guessed from left/right position, since that flips in hairpins.
6. **Frame quality.** Brightness, blown-out %, near-black %, and a normalized sharpness number are measured, then flagged (see below).

Defaults: `--temporal-concepts all`, `--enhance-concepts "blue cone,yellow cone"`. For daylight footage use `--no-enhance` (the blue boost also boosts blue sky).

The train/val split is by blocks of consecutive frames (`--block-size`, default 20). Shuffling single frames would put near-identical neighbors on both sides and inflate validation scores.

### Reviewing the labels

SAM3's output is used as ground truth, so every guess must be visible.

**`label_report.csv`** has one row per box: frame, concept, source (`detected` / `propagated`), score, `enhanced` (found only on the enhanced copy), `low_conf`, `relabeled`, `color_conflict`, `color_votes` (like `blue:3.1|yellow:0.6`), track id, box coordinates, and the frame's flags, brightness and priority.

**`frame_quality.csv`** has one row per frame, including frames with no boxes:

| Flag | Meaning |
|---|---|
| `glare` | at least 1% of pixels blown out AND at least 20% nearly black (a lamp or headlight in the dark). A bright white sky alone does not count. |
| `low_light` | median gray level below 25. Informational: on a night recording nearly every frame has it. |
| `blurry` | sharpness below half the sequence's median sharpness |

**`review_first.csv`** lists frames to look at, worst first:

| Priority | Meaning |
|---|---|
| 3 | A flagged frame with **no boxes**. The label file is empty, so YOLO would be taught "no cone here". |
| 2 | At least half of its boxes are guesses, or a color disagreement/relabel happened |
| 1 | Some guessed (propagated) boxes |
| 0 | Everything else. `low_light` and `low_conf` alone never raise a frame. |

**Preview images** (`--preview`): magenta box / `P` = propagated, `E` = found on the enhanced copy, `R` = color relabeled, `!` = color conflict.

`--export-review DIR` copies the frames that need review, with their draft YOLO labels, into one folder that can be opened in CVAT, Label Studio or Roboflow. `--drop-propagated` keeps propagated boxes out of the label files (they stay in the report).

The glare, blur and low-light thresholds (`--glare-sat-pct`, `--glare-dark-pct`, `--low-light-median`, `--blur-ratio`) are starting points measured on one night frame. Tune them on your own footage.

### Training and how to read the output

```
python train_yolo26s.py --data dataset_seq/data.yaml --epochs 30
```

Results land in `runs_cone_autolabel/yolo26s_sam3_autolabel/` (`results.csv`, `results.png`, `weights/best.pt`). `.pt` is a PyTorch checkpoint; `best.pt` is the best validation epoch.

**Losses** (printed values are already multiplied by their gains):

| Loss | What it measures |
|---|---|
| `box_loss` | box position/size error, 7.5 x (1 - CIoU) |
| `cls_loss` | class (color) error, binary cross-entropy against IoU-aligned targets, gain 0.5 |
| `l1_loss` | L1 distance error on box edges, gain 1.5. YOLO26 has no DFL, so there is no `dfl_loss`. |

The raw box loss is about 0.2 (CIoU about 0.8); the printed 1-2 is the 7.5x gain, not a negative CIoU.

**Metrics:** precision (of drawn boxes, how many are right), recall (of real cones, how many were found), mAP50 (overlap at least 50%), mAP50-95 (stricter overlap, rewards tight boxes). The `(B)` in plot titles means box detection.

**Resolution.** Training uses `imgsz=640` by default. The camera frames are 1920x1200, so far cones are only a few pixels wide after resizing. 1280 costs about 4x the compute of 640. The horizon-band crop and the 1280 experiment are not done yet.

### Results so far

**Small test (15 frames, 20 epochs, single-frame labeler).** This only proves the plumbing works end to end:

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| all | 170 | 0.896 | 0.758 | 0.82 | 0.544 |
| blue | 97 | 0.884 | 0.856 | 0.91 | 0.562 |
| yellow | 59 | 0.884 | 0.78 | 0.894 | 0.516 |
| orange | 14 | 0.965 | 0.643 | 0.645 | 0.558 |

Orange is weakest: few instances and confusion with yellow. Inference was about 41 ms per image on the Mac.

**30-epoch run** [TODO: confirm which dataset these plots came from]: precision about 0.89, recall about 0.70, mAP50 about 0.81, mAP50-95 about 0.59. Losses fell steadily, validation loss did not rise, and the metrics flattened around epochs 20-25. Precision is much higher than recall (cautious model: few false alarms, about 30% of cones missed), and mAP50-95 is far below mAP50 (boxes are not tight, especially on far cones).

**Night recording analysis (378 consecutive frames).** The frame-quality report showed:
- Frames 0-215 are healthy: 10-12 cones per frame, no guessed boxes.
- Frames 255-275 lose almost everything in glare and blur: the box count falls from about 19 to 0, and 266-275 have no labels at all. Frames 255-257 are 100% propagated.
- Frames 336-362 have about 0-1 boxes per frame against about 12 elsewhere (to be checked by eye: empty track or missed cones).
- 22 frames have zero boxes while flagged (priority 3), and these are in the training and validation sets.
- 478 of 3,400 training boxes (14%) are propagated guesses. 108 of 908 validation boxes are guesses.
- The validation blocks (40-59, 140-159, 220-239, 320-339) include the hardest stretches, so validation is harsher and less reliable than training.

### What was learned

- **Enhancement helps dim blue cones but does not fix glare or motion blur.** Tested on a real night frame: brightening mostly amplified noise, because the cones there are motion-blurred smears with washed-out color and about 3% of pixels are fully blown out. Clipped pixels cannot be recovered in software.
- **Real fixes:** shorter exposure with more gain (or HDR) at the camera; hand-checked labels on the hard frames so YOLO can learn what blurred cones look like (SAM3's text prompt cannot); training augmentations (motion blur, darkening, noise, fake flare); and per-condition evaluation.
- **LiDAR was deliberately not used**, because the goal is a camera-only (independent) perception path.

### Limitations

- All labels are SAM3's. The scores measure agreement with SAM3, not with the real cones: a cone SAM3 missed is "background" in the labels, and a model that also misses it is scored as correct.
- Propagated boxes are estimates. Review them before trusting a model trained on them.
- The self-tests use a fake color-based detector. They check the tracking, enhancement, color vote, report and file plumbing, not SAM3's quality on real footage.
- Thresholds (score cutoffs, `--max-gap`, flag limits) are not yet tuned on real data.
