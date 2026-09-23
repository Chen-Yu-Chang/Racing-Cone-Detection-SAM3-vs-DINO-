"""
run_grounding_dino.py

Runs Grounding DINO (via HuggingFace transformers) on every image in a
folder, using text prompts for cone colors, and saves annotated copies with
boxes + labels + confidence drawn on them.

Usage:
    python run_grounding_dino.py --frames frames/ --out out_dino/

First run will download model weights from Hugging Face (~1.7GB for the
"base" checkpoint, ~170MB for "tiny") - needs normal internet access.
"""

import argparse
import os
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

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No .jpg/.png images found in {frames_dir}")

    for fp in frame_paths:
        image = Image.open(fp).convert("RGB")
        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # transformers has renamed this function's threshold arguments across
        # versions (box_threshold -> threshold, in some releases). Instead of
        # hardcoding a name that might not match what you have installed,
        # check what the installed version actually accepts and use that.
        import inspect
        sig = inspect.signature(processor.post_process_grounded_object_detection)
        kwargs = {"text_threshold": TEXT_THRESHOLD, "target_sizes": [image.size[::-1]]}
        if "box_threshold" in sig.parameters:
            kwargs["box_threshold"] = BOX_THRESHOLD
        elif "threshold" in sig.parameters:
            kwargs["threshold"] = BOX_THRESHOLD

        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **kwargs
        )[0]

        annotated = draw_boxes(image, results["boxes"], results["text_labels"], results["scores"])
        out_path = Path(out_dir) / fp.name
        annotated.save(out_path)
        print(f"{fp.name}: {len(results['boxes'])} detections -> {out_path}")

    print(f"\nDone. Annotated frames written to {out_dir}")


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
