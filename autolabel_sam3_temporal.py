"""
autolabel_sam3_temporal.py

Version of autolabel_sam3_to_yolo.py for the hard cases: blue cones that are
dim, backlit or blurry and that SAM3 misses on a single frame. It adds two
things on top of the single-frame pipeline:

  1. Enhanced second look. If SAM3 loses a blue cone that we were tracking
     (or finds no blue cones at all), the frame is run through
     enhance_for_detection() (local contrast + blue boost + sharpening) and
     SAM3 is asked again. Only the detector sees the enhanced copy; the
     image saved into the dataset is always the untouched original.

  2. Previous-frame check. Optical flow slides last frame's blue boxes into
     this frame. Then:
       - a faint SAM3 detection (score between --lo-score and --hi-score) is
         accepted if it lines up with a tracked cone;
       - a tracked cone that nothing detected is carried forward as a
         'propagated' box for at most --max-gap frames in a row.

  3. Color memory. All tracked colors share ONE tracker, so a physical cone is
     one track whatever color SAM3 called it in a given frame. Each confident
     detection votes for its color. A faint or smeared detection, and a
     carried-forward box, takes the track's voted color (a blue cone that SAM3
     calls 'yellow' at 0.3 in a glare frame stays blue). A CONFIDENT detection
     that disagrees with the vote is kept as-is but flagged 'color_conflict'.
     (It deliberately does NOT guess color from left/right position - that
     flips in hairpins.)

  4. Frame quality report. Every frame gets brightness / blown-out % /
     blur numbers and flags (glare, low_light, blurry), written to
     frame_quality.csv, plus review_first.csv: the frames a human should look
     at, worst first. A glared frame with ZERO boxes is the dangerous case:
     the label file is empty, so YOLO would learn "no cone here". Use
     --export-review DIR to copy those frames with their draft labels into a
     folder you can open in a labeling tool and fix by hand.

IMPORTANT - what this needs from your data
  * Frames must be CONSECUTIVE and close in time. The 1-frame-per-second
    sampling used for the first test (--every-n 30) is too sparse: the car
    has moved metres between frames, so last frame says little about this
    one. Re-extract with something like
        python extract_frames.py --mcap ... --topic ... --every-n 3 --max-frames 150 --out frames_seq/
    (about 10 fps for a 30 fps camera; every-n 2 is better if you can afford it).
  * Frames are processed in sorted filename order.

IMPORTANT - what to review
  Propagated boxes are guesses, not detections. Every box is written to
  label_report.csv with its source (detected / propagated), score, and
  whether it was found on the enhanced copy. Use --preview to also write
  annotated images (magenta = propagated) and look at them before training.
  --drop-propagated leaves them out of the label files.

Usage:
    python autolabel_sam3_temporal.py --frames frames_seq/ --out dataset_seq/ --preview preview_seq/
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from temporal_utils import (
    Det, Out, TrackFiller, assign_blocks_split, box_to_yolo_line,
    enhance_for_detection, estimate_flow, flag_frames, frame_quality,
    nms_dets, to_cpu_numpy,
)

CONCEPTS = ["blue cone", "yellow cone", "orange cone"]

COLORS = {
    "blue cone": (60, 120, 255),
    "yellow cone": (255, 220, 60),
    "orange cone": (255, 140, 40),
}
PROPAGATED_COLOR = (255, 0, 255)


class Sam3Detector:
    """Thin wrapper so the rest of the script does not care about SAM3's API."""

    def __init__(self, confidence_threshold: float):
        import sam3_mac  # imported here so the pure-logic parts can be tested without it
        sam3_mac.install()
        self.model, self.processor = sam3_mac.build_image_model(
            confidence_threshold=confidence_threshold
        )

    def embed(self, image: Image.Image):
        # One embedding per image, reused for every concept.
        return self.processor.set_image(image)

    def detect(self, state, concept: str, src: str = "orig", cls: int = 0) -> list:
        out = self.processor.set_text_prompt(state=state, prompt=concept)
        dets = []
        for box, score in zip(out["boxes"], out["scores"]):
            b = to_cpu_numpy(box).reshape(-1)[:4]
            dets.append(Det(tuple(float(v) for v in b), float(to_cpu_numpy(score)), src, cls))
        return dets


def parse_concept_list(value: str, concepts: list) -> list:
    value = (value or "").strip().lower()
    if value in ("", "none"):
        return []
    if value == "all":
        return list(concepts)
    return [c.strip() for c in value.split(",") if c.strip()]


def draw_preview(image, rows, concepts):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for r in rows:
        color = PROPAGATED_COLOR if r["source"] == "propagated" else COLORS.get(r["concept"], (255, 0, 0))
        x0, y0, x1, y1 = r["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3 if r["source"] == "propagated" else 2)
        tag = "P" if r["source"] == "propagated" else ("E" if r["enhanced"] else "")
        if r.get("relabeled"):
            tag += "R"          # faint detection took the track's color
        if r.get("conflict"):
            tag += "!"          # confident detection disagrees with the track's color
        draw.text((x0, max(0, y0 - 12)), f"{r['concept'].split()[0]} {r['score']:.2f} {tag}", fill=color)
    return img


def run(args) -> None:
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    temporal_concepts = parse_concept_list(args.temporal_concepts, concepts)
    enhance_concepts = parse_concept_list(args.enhance_concepts, concepts)
    if args.no_enhance:
        enhance_concepts = []

    frame_paths = sorted(Path(args.frames).glob("*.jpg")) + sorted(Path(args.frames).glob("*.png"))
    frame_paths = sorted(frame_paths)
    if not frame_paths:
        raise SystemExit(f"No .jpg/.png images found in {args.frames}")

    print(f"Concepts: {concepts}")
    print(f"Temporal (previous-frame check) for: {temporal_concepts or 'nothing'}")
    print(f"Enhanced second look for: {enhance_concepts or 'nothing'}")

    # Build SAM3 with the LOW threshold. We filter by score ourselves so that
    # faint detections are available to be confirmed by the tracker.
    detector = Sam3Detector(args.lo_score)

    # ONE tracker for all tracked colors, so the color vote can work across them
    tracked_ids = [i for i, c in enumerate(concepts) if c in temporal_concepts]
    tracker = (TrackFiller(min_hits=args.min_hits, max_gap=args.max_gap,
                           hi_score=args.hi_score, lo_score=args.lo_score)
               if tracked_ids else None)

    splits = assign_blocks_split(len(frame_paths), args.val_split, args.block_size)

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview) if args.preview else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    frame_info = []       # one dict per frame, filled in as we go, flagged at the end
    prev_gray = None
    totals = {"detected": 0, "propagated": 0, "enhanced_rescues": 0, "resets": 0}

    for idx, fp in enumerate(frame_paths):
        image = Image.open(fp).convert("RGB")
        img_w, img_h = image.size
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)

        # --- previous-frame motion ------------------------------------------
        flow = None
        if tracker and prev_gray is not None and prev_gray.shape == gray.shape:
            flow = estimate_flow(prev_gray, gray)
            if flow.median_magnitude() > args.reset_px:
                # the view jumped (new run, big gap): old boxes are meaningless
                tracker.reset()
                flow = None
                totals["resets"] += 1
                print(f"  {fp.name}: scene jumped, tracks reset")
        elif tracker and prev_gray is not None and prev_gray.shape != gray.shape:
            tracker.reset()
        if tracker:
            tracker.predict(flow)
        prev_gray = gray
        quality = frame_quality(np.asarray(image))

        # --- pass 1: original frame -----------------------------------------
        state = detector.embed(image)
        dets = {c: detector.detect(state, c, cls=i) for i, c in enumerate(concepts)}

        def tracked_dets():
            return [d for i in tracked_ids for d in dets[concepts[i]]]

        # --- pass 2: enhanced copy, only where something looks missing -------
        need = []
        for c in enhance_concepts:
            cid = concepts.index(c)
            no_confident = not any(d.score >= args.hi_score for d in dets[c])
            if args.always_enhance:
                need.append(c)
            elif tracker and cid in tracked_ids:
                if tracker.unmatched_confirmed(tracked_dets(), cls=cid) or no_confident:
                    need.append(c)
            elif no_confident:
                need.append(c)
        if need:
            state_enh = detector.embed(enhance_for_detection(image))
            for c in need:
                extra = detector.detect(state_enh, c, src="enh", cls=concepts.index(c))
                dets[c] = nms_dets(dets[c] + extra, iou_thr=0.5)

        # --- decide final boxes ---------------------------------------------
        frame_rows = []

        def add_row(o, cls_id):
            frame_rows.append({
                "frame": fp.name, "concept": concepts[cls_id], "class_id": cls_id,
                "source": o.source, "score": round(float(o.score), 3),
                "enhanced": o.enhanced, "low_conf": o.low_conf,
                "relabeled": getattr(o, "relabeled", False),
                "conflict": getattr(o, "conflict", False),
                "votes": "|".join(f"{concepts[k].split()[0]}:{v:.1f}"
                                  for k, v in sorted(getattr(o, "votes", {}).items())),
                "track_id": o.track_id, "box": o.box,
            })

        if tracker:
            for o in tracker.update(tracked_dets(), img_w, img_h):
                add_row(o, o.cls)
        for class_id, c in enumerate(concepts):
            if class_id in tracked_ids:
                continue
            for d in dets[c]:
                if d.score >= args.hi_score:
                    add_row(Out(d.box, d.score, "detected", d.src == "enh", False, -1), class_id)

        lines = []
        for r in frame_rows:
            if r["source"] == "propagated" and args.drop_propagated:
                continue
            lines.append(box_to_yolo_line(r["class_id"], r["box"], img_w, img_h))

        split = splits[idx]
        shutil.copy(fp, out / "images" / split / fp.name)       # ORIGINAL pixels, never the enhanced copy
        (out / "labels" / split / (fp.stem + ".txt")).write_text("\n".join(lines))

        n_det = sum(r["source"] == "detected" for r in frame_rows)
        n_prop = sum(r["source"] == "propagated" for r in frame_rows)
        n_resc = sum(r["source"] == "detected" and r["enhanced"] for r in frame_rows)
        totals["detected"] += n_det
        totals["propagated"] += n_prop
        totals["enhanced_rescues"] += n_resc
        report_rows.extend(frame_rows)
        frame_info.append({
            "frame": fp.name, "split": split, "quality": quality,
            "boxes": len(frame_rows), "propagated": n_prop,
            "low_conf": sum(r["low_conf"] for r in frame_rows),
            "relabeled": sum(r["relabeled"] for r in frame_rows),
            "conflicts": sum(r["conflict"] for r in frame_rows),
        })
        flag = f"  [enhanced pass: {', '.join(need)}]" if need else ""
        print(f"{fp.name} [{split}]: {n_det} detected, {n_prop} propagated, {n_resc} from enhanced{flag}")

        if preview_dir:
            draw_preview(image, frame_rows, concepts).save(preview_dir / fp.name)

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(concepts))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n"
    )

    # ---- frame quality + review list ------------------------------------
    flags_per_frame = flag_frames(
        [fi["quality"] for fi in frame_info],
        glare_sat_pct=args.glare_sat_pct, glare_dark_pct=args.glare_dark_pct,
        low_light_median=args.low_light_median, blur_ratio=args.blur_ratio)
    for fi, fl in zip(frame_info, flags_per_frame):
        fi["flags"] = fl
        reasons = list(fl)
        if fi["propagated"]:
            reasons.append("propagated")
        if fi["conflicts"]:
            reasons.append("color_conflict")
        if fi["relabeled"]:
            reasons.append("relabeled")
        if fi["low_conf"]:
            reasons.append("low_conf")
        if fl and fi["boxes"] == 0:
            reasons.append("hard_frame_no_boxes")
        fi["reasons"] = reasons
        # 3 = glared/dark/blurry frame with no labels at all (silent missing labels)
        # 2 = contains guessed boxes or color disagreements; 1 = hard frame or faint boxes
        if "hard_frame_no_boxes" in reasons:
            fi["priority"] = 3
        elif fi["propagated"] or fi["conflicts"] or fi["relabeled"]:
            fi["priority"] = 2
        elif reasons:
            fi["priority"] = 1
        else:
            fi["priority"] = 0
    by_frame = {fi["frame"]: fi for fi in frame_info}

    q_cols = ["frame", "split", "boxes", "propagated", "low_conf", "relabeled", "color_conflicts",
              "brightness", "median", "sat_pct", "dark_pct", "sharpness",
              "flags", "priority", "review_reason"]

    def q_row(fi):
        q = fi["quality"]
        return [fi["frame"], fi["split"], fi["boxes"], fi["propagated"], fi["low_conf"],
                fi["relabeled"], fi["conflicts"], round(q["brightness"], 1), round(q["median"], 1),
                round(q["sat_pct"], 2), round(q["dark_pct"], 1), round(q["sharpness"], 5),
                "|".join(fi["flags"]), fi["priority"], "|".join(fi["reasons"])]

    with open(out / "frame_quality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(q_cols)
        for fi in frame_info:
            w.writerow(q_row(fi))
    to_review = sorted((fi for fi in frame_info if fi["priority"] > 0),
                       key=lambda fi: (-fi["priority"], fi["frame"]))
    with open(out / "review_first.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(q_cols)
        for fi in to_review:
            w.writerow(q_row(fi))

    if args.export_review:
        rdir = Path(args.export_review)
        (rdir / "images").mkdir(parents=True, exist_ok=True)
        (rdir / "labels").mkdir(parents=True, exist_ok=True)
        for fi in to_review:
            name = Path(fi["frame"])
            shutil.copy(out / "images" / fi["split"] / name.name, rdir / "images" / name.name)
            shutil.copy(out / "labels" / fi["split"] / (name.stem + ".txt"), rdir / "labels" / (name.stem + ".txt"))

    with open(out / "label_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "concept", "source", "score", "enhanced", "low_conf", "relabeled",
                    "color_conflict", "color_votes", "track_id", "x0", "y0", "x1", "y1",
                    "frame_flags", "brightness", "sat_pct", "priority"])
        for r in report_rows:
            x0, y0, x1, y1 = (round(float(v), 1) for v in r["box"])
            fi = by_frame[r["frame"]]
            w.writerow([r["frame"], r["concept"], r["source"], r["score"], r["enhanced"],
                        r["low_conf"], r["relabeled"], r["conflict"], r["votes"], r["track_id"],
                        x0, y0, x1, y1, "|".join(fi["flags"]),
                        round(fi["quality"]["brightness"], 1), round(fi["quality"]["sat_pct"], 2),
                        fi["priority"]])

    print(f"\nDone. {totals['detected']} detected boxes "
          f"({totals['enhanced_rescues']} only found on the enhanced copy), "
          f"{totals['propagated']} propagated boxes, {totals['resets']} track resets.")
    n_hard0 = sum(fi["priority"] == 3 for fi in frame_info)
    print(f"Dataset: {out}/   Box list: {out / 'label_report.csv'}")
    print(f"Frames to look at first: {out / 'review_first.csv'}  "
          f"({len(to_review)} of {len(frame_info)} frames; {n_hard0} are hard frames with NO labels)")
    if args.export_review:
        print(f"Review frames + draft labels copied to: {args.export_review}/")
    if preview_dir:
        print(f"Preview images (magenta = propagated, E = enhanced rescue): {preview_dir}/")
    print("NOTE: propagated boxes are estimates - review them before training on this set.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of CONSECUTIVE frames (sorted by filename)")
    ap.add_argument("--out", default="dataset_seq")
    ap.add_argument("--preview", default=None, help="folder for annotated review images")
    ap.add_argument("--concepts", default=",".join(CONCEPTS))
    ap.add_argument("--temporal-concepts", default="all",
                    help="which colors get the previous-frame check AND share the color vote: "
                         "'all' (default), 'none', or a comma list like 'blue cone,yellow cone'")
    ap.add_argument("--enhance-concepts", default="blue cone,yellow cone",
                    help="which colors get the enhanced second look: 'blue cone,yellow cone' (default), 'all', 'none', or a comma list")
    ap.add_argument("--no-enhance", action="store_true", help="turn the enhanced second look off")
    ap.add_argument("--always-enhance", action="store_true",
                    help="run the enhanced pass on every frame instead of only when a cone looks missing (slower)")
    ap.add_argument("--hi-score", type=float, default=0.5,
                    help="detections at/above this are trusted on their own (same as the old --confidence-threshold)")
    ap.add_argument("--lo-score", type=float, default=0.25,
                    help="weaker detections are kept only if they line up with a tracked cone")
    ap.add_argument("--min-hits", type=int, default=2, help="detections needed before a track may be carried forward")
    ap.add_argument("--max-gap", type=int, default=3, help="most consecutive frames a cone may be carried forward")
    ap.add_argument("--reset-px", type=float, default=60.0,
                    help="if the typical scene motion between two frames exceeds this many pixels, forget all tracks")
    ap.add_argument("--drop-propagated", action="store_true",
                    help="write propagated boxes to the report/preview only, not into the YOLO label files")
    ap.add_argument("--export-review", default=None,
                    help="copy frames that need review (with their draft YOLO labels) into this folder, "
                         "ready to open in a labeling tool")
    ap.add_argument("--glare-sat-pct", type=float, default=1.0, help="glare flag: at least this %% of pixels blown out ...")
    ap.add_argument("--glare-dark-pct", type=float, default=20.0, help="... AND at least this %% nearly black")
    ap.add_argument("--low-light-median", type=float, default=25.0, help="low_light flag: median gray level below this")
    ap.add_argument("--blur-ratio", type=float, default=0.5,
                    help="blurry flag: sharpness below this fraction of the sequence's median sharpness")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--block-size", type=int, default=20,
                    help="frames per block for the train/val split (whole blocks go to val so neighbours don't leak)")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
