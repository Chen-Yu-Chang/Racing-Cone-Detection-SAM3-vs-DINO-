"""
run_grounding_dino.py

Runs Grounding DINO (via HuggingFace transformers) on every image in a
folder, using text prompts for cone colors, and saves annotated copies with
boxes + labels + confidence drawn on them.

Also benchmarks per-frame inference time (model load time is reported
separately, since that's a one-time cost, not a per-frame one) and writes
the results to <out>/benchmark.json so compare_side_by_side.py can show
them next to SAM3's numbers.

Usage:
    python run_grounding_dino.py --frames frames/ --out out_dino/

First run will download model weights from Hugging Face (~1.7GB for the
"base" checkpoint, ~170MB for "tiny") - needs normal internet access.
"""

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Grounding DINO wants prompts as lowercase phrases separated by periods.
# Tune these to match whatever your team's actual cone color labels are.
TEXT_PROMPT = "blue cone. yellow cone. orange cone."

BOX_THRESHOLD = 0.30   # lower = more (noisier) detections, higher = stricter
TEXT_THRESHOLD = 0.25


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple Silicon
        return "mps"
    return "cpu"


def sync_device(device):
    """GPU ops (mps/cuda) are launched asynchronously - without this, the
    timer stops before the actual compute finishes and every number looks
    unrealistically fast."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def draw_boxes(image: Image.Image, boxes, labels, scores):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, max(0, y0 - 12)), f"{label} {score:.2f}", fill="red")
    return img


def main(frames_dir: str, out_dir: str, model_id: str):
    device = get_device()
    print(f"Using device: {device}")

    t_load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    model_load_time = time.perf_counter() - t_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No .jpg/.png images found in {frames_dir}")

    per_frame_log = []

    for fp in frame_paths:
        image = Image.open(fp).convert("RGB")

        t_frame_start = time.perf_counter()

        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # transformers has renamed this function's threshold arguments across
        # versions (box_threshold -> threshold, in some releases). Instead of
        # hardcoding a name that might not match what you have installed,
        # check what the installed version actually accepts and use that.
        sig = inspect.signature(processor.post_process_grounded_object_detection)
        kwargs = {"text_threshold": TEXT_THRESHOLD, "target_sizes": [image.size[::-1]]}
        if "box_threshold" in sig.parameters:
            kwargs["box_threshold"] = BOX_THRESHOLD
        elif "threshold" in sig.parameters:
            kwargs["threshold"] = BOX_THRESHOLD

        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **kwargs
        )[0]

        sync_device(device)  # make sure the forward pass has actually finished before stopping the clock
        frame_time = time.perf_counter() - t_frame_start

        per_frame_log.append({
            "frame": fp.name,
            "inference_time_sec": round(frame_time, 4),
            "num_detections": len(results["boxes"]),
        })

        annotated = draw_boxes(image, results["boxes"], results["text_labels"], results["scores"])
        out_path = Path(out_dir) / fp.name
        annotated.save(out_path)
        print(f"{fp.name}: {len(results['boxes'])} detections -> {out_path}  ({frame_time:.3f}s)")

    times = [r["inference_time_sec"] for r in per_frame_log]
    summary = {
        "model": model_id,
        "device": device,
        "num_frames": len(times),
        "model_load_time_sec": round(model_load_time, 3),
        "total_inference_time_sec": round(sum(times), 3) if times else None,
        "mean_inference_time_sec": round(statistics.mean(times), 4) if times else None,
        "fps": round(len(times) / sum(times), 3) if times and sum(times) else None,
        "per_frame": per_frame_log,
    }
    (Path(out_dir) / "benchmark.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. Annotated frames written to {out_dir}")
    print(
        f"Benchmark: {summary['num_frames']} frames on {device}, "
        f"mean {summary['mean_inference_time_sec']}s/frame, "
        f"{summary['fps']} fps  (full detail in {out_dir}/benchmark.json)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of extracted frames")
    ap.add_argument("--out", default="out_dino")
    ap.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
        help="use 'IDEA-Research/grounding-dino-base' for higher accuracy "
             "(slower, bigger download) - worth trying both since you're "
             "optimizing for accuracy, not real-time speed",
    )
    args = ap.parse_args()
    main(args.frames, args.out, args.model)
