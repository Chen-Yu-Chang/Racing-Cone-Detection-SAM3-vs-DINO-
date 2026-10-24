"""
run_sam3.py

Runs SAM3 (Meta, Nov 2025) on every image in a folder using text "concept"
prompts for cone colors, and saves annotated copies with the segmentation
masks overlaid + confidence scores.

Also benchmarks per-frame inference time and writes the results to
<out>/benchmark.json so compare_side_by_side.py can show them next to
Grounding DINO's numbers.

SAM3 is very new - the exact API below matches the official repo's README
as of when this was written. If a call signature has changed, check
https://github.com/facebookresearch/sam3 directly (their example notebooks
are the source of truth, more so than this script).

--- On Apple Silicon Macs ---
SAM3's official code hard-requires `triton`, which has no macOS build (this
is a known, common issue - see facebookresearch/sam3 issues #179 and #154,
and the HF discussion "Cannot run on Apple Silicon (M4) due to Triton").
This script uses the community `sam3-mac` compatibility shim instead of
importing sam3 directly, which patches around that at runtime and picks
MPS/CPU automatically. Its one limitation is that video tracking doesn't
work - that's fine here, we're only doing per-frame text-prompted image
segmentation, not tracking across frames.

    pip install "git+https://github.com/benreichman/sam3-mac.git"

If you're on Linux with an NVIDIA GPU instead, `triton` installs fine
normally and you likely don't need sam3-mac at all - swap the import block
below back to plain `from sam3.model_builder import build_sam3_image_model`.

Usage:
    python run_sam3.py --frames frames/ --out out_sam3/
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sam3_mac
sam3_mac.install()  # auto-picks mps/cpu, patches around the triton import

# One concept at a time keeps this simple and matches the README example.
# Loop over your actual cone color classes here.
CONCEPTS = ["blue cone", "yellow cone", "orange cone"]

MASK_COLORS = {
    "blue cone": (60, 120, 255),
    "yellow cone": (255, 220, 60),
    "orange cone": (255, 140, 40),
}


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def sync_device(device):
    """GPU ops (mps/cuda) are launched asynchronously - without this, the
    timer stops before the actual compute finishes and every number looks
    unrealistically fast."""
    try:
        import torch
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def to_cpu_numpy(x):
    """Handles values whether they're torch tensors (possibly still on the
    mps/cuda device from inference) or already plain numpy/python values."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def overlay_masks(image: Image.Image, masks, boxes, scores, concept: str):
    color = MASK_COLORS.get(concept, (255, 0, 0))

    masks = [to_cpu_numpy(m) for m in masks]
    boxes = [to_cpu_numpy(b) for b in boxes]
    scores = [float(to_cpu_numpy(s)) for s in scores]

    # Pass 1: composite every mask's translucent fill onto the image.
    # (each alpha_composite call returns a NEW image, so img is reassigned
    # each time - that's fine here since nothing else is bound to the old one)
    img = image.convert("RGBA")
    for mask in masks:
        mask_np = mask.astype(bool)
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        colored = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        colored[mask_np] = (*color, 110)  # semi-transparent fill
        img = Image.alpha_composite(img, Image.fromarray(colored, "RGBA"))

    # Pass 2: NOW create the draw context, bound to the final composited
    # image, and draw all boxes/labels directly onto it in place. Doing this
    # in a separate pass (after all the reassignment above is done) is what
    # was missing before - draw.rectangle/draw.text mutate img in place, so
    # this only works correctly if img isn't being reassigned anymore.
    draw = ImageDraw.Draw(img)
    for box, score in zip(boxes, scores):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(0, y0 - 12)), f"{concept} {score:.2f}", fill=color)

    return img.convert("RGB")


def main(frames_dir: str, out_dir: str, concepts: list, confidence_threshold: float):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading SAM3 model (confidence_threshold={confidence_threshold})...")

    t_load_start = time.perf_counter()
    model, processor = sam3_mac.build_image_model(confidence_threshold=confidence_threshold)
    sync_device(device)
    model_load_time = time.perf_counter() - t_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No .jpg/.png images found in {frames_dir}")

    per_frame_log = []

    for fp in frame_paths:
        image = Image.open(fp).convert("RGB")
        annotated = image.copy()
        total = 0

        t_frame_start = time.perf_counter()

        # NOTE: set_image() is called ONCE per frame here, on the clean,
        # un-annotated image, and the resulting embedding is reused across
        # all concepts below. The original version called set_image() once
        # PER CONCEPT, on `annotated` - which by the 2nd/3rd concept already
        # had earlier overlays drawn onto it. That was both ~3x slower than
        # necessary (re-embedding the whole image every concept) and meant
        # later concepts were being detected against a partially-annotated
        # image instead of the real one. Fixed here so the benchmark timing
        # reflects real per-frame cost and every concept sees the same,
        # unmodified input.
        inference_state = processor.set_image(image)

        for concept in concepts:
            output = processor.set_text_prompt(state=inference_state, prompt=concept)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            total += len(masks)
            # print every raw score, even ones we'd otherwise ignore, so you
            # can see whether SAM3 is finding low-confidence candidates that
            # the threshold is just filtering out, vs finding nothing at all
            score_list = [round(float(to_cpu_numpy(s)), 3) for s in scores]
            print(f"    '{concept}': {len(masks)} matches, scores={score_list}")
            if len(masks) > 0:
                annotated = overlay_masks(annotated, masks, boxes, scores, concept)

        sync_device(device)
        frame_time = time.perf_counter() - t_frame_start

        per_frame_log.append({
            "frame": fp.name,
            "inference_time_sec": round(frame_time, 4),
            "num_detections": total,
        })

        out_path = Path(out_dir) / fp.name
        annotated.save(out_path)
        print(f"{fp.name}: {total} total instances across {len(concepts)} concepts -> {out_path}  ({frame_time:.3f}s)")

    times = [r["inference_time_sec"] for r in per_frame_log]
    summary = {
        "model": "sam3 (via sam3-mac)",
        "device": device,
        "num_frames": len(times),
        "model_load_time_sec": round(model_load_time, 3),
        "total_inference_time_sec": round(sum(times), 3) if times else None,
        "mean_inference_time_sec": round(statistics.mean(times), 4) if times else None,
        "fps": round(len(times) / sum(times), 3) if times and sum(times) else None,
        "concepts_per_frame": len(concepts),
        "per_frame": per_frame_log,
    }
    (Path(out_dir) / "benchmark.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. Annotated frames written to {out_dir}")
    print(
        f"Benchmark: {summary['num_frames']} frames x {len(concepts)} concepts on {device}, "
        f"mean {summary['mean_inference_time_sec']}s/frame, "
        f"{summary['fps']} fps  (full detail in {out_dir}/benchmark.json)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of extracted frames")
    ap.add_argument("--out", default="out_sam3")
    ap.add_argument(
        "--confidence-threshold", type=float, default=0.5,
        help="lower this (try 0.1) if you're getting zero detections, to see "
             "whether SAM3 is finding low-confidence matches that are just "
             "being filtered out",
    )
    ap.add_argument(
        "--concepts", default=",".join(CONCEPTS),
        help="comma-separated text prompts, e.g. 'cone' or 'traffic cone' "
             "if the color-specific ones ('blue cone') aren't matching well",
    )
    args = ap.parse_args()
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    main(args.frames, args.out, concepts, args.confidence_threshold)
