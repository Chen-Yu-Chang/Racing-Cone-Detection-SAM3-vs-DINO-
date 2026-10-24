"""
train_yolo26s.py

Stage 3: trains YOLO26s on the auto-labeled dataset that
autolabel_sam3_to_yolo.py produced (the SAM3 -> LABELS -> YOLO26s part of
the whiteboard).

On the "loss function feeding back into YOLO26s to increase accuracy"
part: that's not something you need separate code for. Ultralytics'
model.train() already computes the loss (box + classification + DFL loss)
between predictions and your SAM3-derived labels every batch and backprops
through the optimizer automatically, every epoch, for whatever --epochs
you give it - that loop IS this one function call. This script just kicks
it off and tells you where the resulting loss curve lands afterward.

Requires:
    pip install ultralytics

YOLO26 is a very new (2026) release - if the `yolo26s.pt` checkpoint name
below doesn't resolve, check Ultralytics' current docs/model zoo for the
exact name; Ultralytics auto-downloads pretrained weights by name the same
way across YOLOv8/11/12/26, so only the string itself should need updating,
not the surrounding code.

Usage:
    python train_yolo26s.py --data dataset/data.yaml --epochs 20
"""

import argparse

from ultralytics import YOLO


def main(data_yaml: str, model_name: str, epochs: int, imgsz: int, device: str):
    model = YOLO(model_name)  # downloads pretrained YOLO26s weights on first run

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project="runs_cone_autolabel",
        name="yolo26s_sam3_autolabel",
    )

    print("\nDone.")
    print("Per-epoch loss curve (box_loss/cls_loss/dfl_loss) + val metrics: "
          "runs_cone_autolabel/yolo26s_sam3_autolabel/results.csv")
    print("Plotted automatically to: runs_cone_autolabel/yolo26s_sam3_autolabel/results.png")
    print("Best weights: runs_cone_autolabel/yolo26s_sam3_autolabel/weights/best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo26s.pt",
                     help="pretrained checkpoint to fine-tune from - swap for 'yolo26s.yaml' "
                          "to train from-scratch weights instead")
    ap.add_argument("--epochs", type=int, default=20,
                     help="with only ~15 auto-labeled frames this proves the mechanics work, "
                          "it doesn't produce a usable model yet - don't read much into the "
                          "accuracy numbers at this scale")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None,
                     help="'mps', 'cpu', or a cuda device index - omit to let ultralytics auto-pick")
    args = ap.parse_args()
    main(args.data, args.model, args.epochs, args.imgsz, args.device)
