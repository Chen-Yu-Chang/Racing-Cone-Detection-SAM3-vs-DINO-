"""
autolabel_sam3_to_yolo.py

Stage 2 of the pipeline (DATA -> SAM -> LABELS on the whiteboard): runs
SAM3 on a folder of frames, same as run_sam3.py, but instead of drawing
overlays for visual comparison, writes a standard Ultralytics YOLO-format
object detection dataset directly - the auto-labeled data that trains
YOLO26s.

This is a "prove the mechanics work end to end" script, meant to run on the
same small frame set you already have (~15 frames). SAM3's raw output
becomes ground truth here with zero human review - that's fine for
checking the pipeline plumbing (auto-label -> YOLO format -> a training
run actually starts and the loss goes down), but do NOT read anything into
the resulting model's accuracy numbers yet. At real scale you'd want
spot-checking, or at minimum a human-reviewed validation split, before
trusting SAM3's labels as ground truth outright - that's the same
false-positive-poisoning risk called out in the DINO-vs-SAM3 README.

Output layout (standard Ultralytics YOLO dataset):
    dataset/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt      # one line per box: "class_id xc yc w h", normalized 0-1
      labels/val/*.txt
      data.yaml

Usage:
    python autolabel_sam3_to_yolo.py --frames frames/ --out dataset/ --val-split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import sam3_mac
sam3_mac.install()

CONCEPTS = ["blue cone", "yellow cone", "orange cone"]
# The order here defines the class ids written into the label files - it
# must match the `names` list written into data.yaml below, in the same
# order, or your labels and class names will silently mismatch.


def to_cpu_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def box_to_yolo_line(class_id, box, img_w, img_h):
    x0, y0, x1, y1 = box
    xc = (x0 + x1) / 2 / img_w
    yc = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    xc, yc, w, h = (max(0.0, min(1.0, v)) for v in (xc, yc, w, h))
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main(frames_dir, out_dir, concepts, confidence_threshold, min_score, val_split, seed):
    print(f"Loading SAM3 model (confidence_threshold={confidence_threshold})...")
    model, processor = sam3_mac.build_image_model(confidence_threshold=confidence_threshold)

    frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No .jpg/.png images found in {frames_dir}")

    rng = random.Random(seed)
    shuffled = frame_paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])

    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for fp in frame_paths:
        split = "val" if fp in val_set else "train"
        image = Image.open(fp).convert("RGB")
        img_w, img_h = image.size

        # one embedding per frame, reused across all concepts (same fix as
        # in run_sam3.py) - cheaper and keeps every concept looking at the
        # same, unmodified image
        inference_state = processor.set_image(image)

        lines = []
        for class_id, concept in enumerate(concepts):
            output = processor.set_text_prompt(state=inference_state, prompt=concept)
            boxes, scores = output["boxes"], output["scores"]
            for box, score in zip(boxes, scores):
                score_f = float(to_cpu_numpy(score))
                if score_f < min_score:
                    continue
                box_np = to_cpu_numpy(box)
                lines.append(box_to_yolo_line(class_id, box_np, img_w, img_h))

        shutil.copy(fp, out / "images" / split / fp.name)
        (out / "labels" / split / (fp.stem + ".txt")).write_text("\n".join(lines))
        total_boxes += len(lines)
        print(f"{fp.name} [{split}]: {len(lines)} boxes")

    data_yaml = out / "data.yaml"
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(concepts))
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n"
    )

    print(f"\nDone. {total_boxes} auto-labeled boxes across {len(frame_paths)} frames "
          f"({len(frame_paths) - len(val_set)} train / {len(val_set)} val).")
    print(f"Dataset written to {out}/  (see {data_yaml} for the Ultralytics data config)")
    print("\nNOTE: these are unreviewed SAM3 outputs used as-is - fine for testing the "
          "pipeline plumbing, not a substitute for a human-checked validation set.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of extracted frames (same as the DINO/SAM3 comparison)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--concepts", default=",".join(CONCEPTS))
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                     help="SAM3's own internal filtering threshold")
    ap.add_argument("--min-score", type=float, default=0.0,
                     help="extra filter applied on top, if you want stricter label quality")
    ap.add_argument("--val-split", type=float, default=0.2,
                     help="fraction of frames held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    main(args.frames, args.out, concepts, args.confidence_threshold, args.min_score, args.val_split, args.seed)
