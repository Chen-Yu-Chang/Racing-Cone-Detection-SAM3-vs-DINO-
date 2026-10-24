"""
compare_side_by_side.py

Takes the annotated output folders from run_grounding_dino.py and
run_sam3.py and stacks each matching frame side by side (original | DINO |
SAM3), so you can flip through one image per frame instead of two separate
folders. This is what you'd actually look at to judge "which one labeled
better."

If out_dino/benchmark.json and/or out_sam3/benchmark.json exist (both
scripts write these automatically now), their per-frame timing is shown
right in each panel's label, and a benchmark summary is printed at the end.

Usage:
    python compare_side_by_side.py --frames frames/ --dino out_dino/ --sam3 out_sam3/ --out comparison/
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def label_strip(width: int, text: str, height: int = 28) -> Image.Image:
    strip = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(strip)
    draw.text((8, 6), text, fill="white")
    return strip


def load_benchmark(dir_path: str):
    p = Path(dir_path) / "benchmark.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def main(frames_dir: str, dino_dir: str, sam3_dir: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dino_bench = load_benchmark(dino_dir)
    sam3_bench = load_benchmark(sam3_dir)
    dino_times = {r["frame"]: r["inference_time_sec"] for r in dino_bench["per_frame"]} if dino_bench else {}
    sam3_times = {r["frame"]: r["inference_time_sec"] for r in sam3_bench["per_frame"]} if sam3_bench else {}

    frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {frames_dir}")

    for fp in frame_paths:
        original = Image.open(fp).convert("RGB")
        dino_path = Path(dino_dir) / fp.name
        sam3_path = Path(sam3_dir) / fp.name

        panels = [("original", original)]
        if dino_path.exists():
            label = "Grounding DINO"
            if fp.name in dino_times:
                label += f" ({dino_times[fp.name]:.2f}s)"
            panels.append((label, Image.open(dino_path).convert("RGB")))
        if sam3_path.exists():
            label = "SAM3"
            if fp.name in sam3_times:
                label += f" ({sam3_times[fp.name]:.2f}s)"
            panels.append((label, Image.open(sam3_path).convert("RGB")))

        # resize all panels to the same height for a clean row
        target_h = 480
        resized = []
        for name, img in panels:
            w = int(img.width * (target_h / img.height))
            resized.append((name, img.resize((w, target_h))))

        total_w = sum(img.width for _, img in resized)
        strip_h = 28
        canvas = Image.new("RGB", (total_w, target_h + strip_h), (0, 0, 0))

        x = 0
        for name, img in resized:
            canvas.paste(label_strip(img.width, name), (x, 0))
            canvas.paste(img, (x, strip_h))
            x += img.width

        out_path = Path(out_dir) / fp.name
        canvas.save(out_path)
        print(f"wrote {out_path}")

    if dino_bench or sam3_bench:
        print("\nBenchmark summary:")
        if dino_bench:
            print(
                f"  Grounding DINO ({dino_bench.get('device')}): "
                f"{dino_bench['mean_inference_time_sec']}s/frame avg, "
                f"{dino_bench['fps']} fps, "
                f"model load {dino_bench.get('model_load_time_sec')}s"
            )
        if sam3_bench:
            print(
                f"  SAM3 ({sam3_bench.get('device')}): "
                f"{sam3_bench['mean_inference_time_sec']}s/frame avg, "
                f"{sam3_bench['fps']} fps, "
                f"model load {sam3_bench.get('model_load_time_sec')}s"
            )
        if dino_bench and sam3_bench and dino_bench.get("mean_inference_time_sec") and sam3_bench.get("mean_inference_time_sec"):
            ratio = sam3_bench["mean_inference_time_sec"] / dino_bench["mean_inference_time_sec"]
            faster = "DINO" if ratio > 1 else "SAM3"
            print(f"  -> {faster} is ~{max(ratio, 1/ratio):.1f}x faster per frame on this run")

    print(f"\nDone. Open the images in {out_dir} and flip through them - "
          f"that's your DINO-vs-SAM3 comparison.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--dino", required=True)
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--out", default="comparison")
    args = ap.parse_args()
    main(args.frames, args.dino, args.sam3, args.out)
