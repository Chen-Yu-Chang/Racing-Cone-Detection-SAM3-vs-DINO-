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

## A note on scale

This was deliberately a ~15-frame smoke test, not a full benchmark. Now
that there's a qualitative signal (SAM3 finds more far-field cones; DINO is
more legible at a glance but less complete), the next step is bringing this
back to the team before scaling up - running either model across a full
dataset is a much bigger compute/time commitment worth deciding on
together, and a proper precision/recall comparison against some
human-labeled ground truth would be needed before picking one for real
auto-labeling use.
