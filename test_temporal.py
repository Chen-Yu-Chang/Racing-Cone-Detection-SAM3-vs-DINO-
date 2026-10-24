"""
test_temporal.py

Self-checks for temporal_utils.py and autolabel_sam3_temporal.py that need
neither SAM3 nor a GPU. Run:

    python test_temporal.py

The end-to-end test swaps SAM3 for a tiny fake detector (finds blue blobs by
color), so it checks the plumbing - flow, tracking, enhanced second look,
label files, report - NOT how well real SAM3 does on your footage.
"""

import csv
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from temporal_utils import (
    Det, TrackFiller, assign_blocks_split, box_iou, enhance_for_detection,
    estimate_flow, flag_frames, frame_quality, nms_dets,
)


def textured_canvas(h, w, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 2.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (50 + 45 * noise).astype(np.uint8)


# ---------------------------------------------------------------------------
def test_flow_follows_motion():
    canvas = textured_canvas(400, 900)
    f0 = canvas[:, 0:640]
    f1 = canvas[:, 6:646]            # scene content moves LEFT by 6 px
    flow = estimate_flow(f0, f1)
    dx, dy = flow.median_in_box((300, 150, 330, 210))
    assert abs(dx - (-6)) < 1.0 and abs(dy) < 1.0, (dx, dy)
    print(f"ok  flow: measured ({dx:.2f}, {dy:.2f}) px, true (-6, 0)")


def test_nms_and_iou():
    a, b = Det((0, 0, 10, 10), 0.9), Det((1, 1, 11, 11), 0.6)
    assert len(nms_dets([a, b], 0.5)) == 1 and nms_dets([b, a], 0.5)[0].score == 0.9
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    print("ok  nms / iou")


def test_tracker_fills_gap_then_gives_up():
    tf = TrackFiller(min_hits=2, max_gap=2, hi_score=0.5, lo_score=0.25)
    box = (100, 100, 120, 140)
    seq = []
    for f in range(7):
        dets = [Det(box, 0.9)] if f in (0, 1, 2, 6) else []
        outs = tf.update(dets, 640, 400)
        seq.append([o.source for o in outs])
    # frames 0-2 detected, 3-4 propagated (max_gap=2), 5 gone, 6 detected again as a NEW track
    assert seq[:3] == [["detected"]] * 3, seq
    assert seq[3] == ["propagated"] and seq[4] == ["propagated"], seq
    assert seq[5] == [] and seq[6] == ["detected"], seq
    print("ok  tracker: gap filled for exactly max_gap frames, then dropped")


def test_unconfirmed_track_not_carried():
    tf = TrackFiller(min_hits=2, max_gap=3)
    assert [o.source for o in tf.update([Det((10, 10, 30, 50), 0.9)], 640, 400)] == ["detected"]
    assert tf.update([], 640, 400) == []      # seen once only -> not trusted to be carried
    print("ok  tracker: a cone seen only once is not carried forward")


def test_faint_detection_needs_a_track():
    box = (100, 100, 120, 140)
    # no history: faint detection must be ignored
    tf = TrackFiller(min_hits=2, hi_score=0.5, lo_score=0.25)
    assert tf.update([Det(box, 0.3)], 640, 400) == []
    # with a confirmed track: same faint detection is accepted and flagged
    tf = TrackFiller(min_hits=2, hi_score=0.5, lo_score=0.25)
    tf.update([Det(box, 0.9)], 640, 400)
    tf.update([Det(box, 0.9)], 640, 400)
    outs = tf.update([Det((102, 100, 122, 140), 0.3)], 640, 400)
    assert len(outs) == 1 and outs[0].source == "detected" and outs[0].low_conf, outs
    # a faint blob far from the track is still rejected
    outs = tf.update([Det((400, 300, 420, 340), 0.3)], 640, 400)
    assert all(o.source == "propagated" for o in outs) and len(outs) == 1, outs
    print("ok  tracker: faint detections accepted only next to a tracked cone")


def test_propagation_follows_flow():
    canvas = textured_canvas(400, 900)
    tf = TrackFiller(min_hits=2, max_gap=3)
    box0 = (300.0, 150.0, 320.0, 190.0)
    for k in range(2):                       # build a confirmed track
        tf.update([Det(box0, 0.9)], 640, 400)
    f0, f1 = canvas[:, 0:640], canvas[:, 5:645]
    tf.predict(estimate_flow(f0, f1))
    outs = tf.update([], 640, 400)           # nothing detected this frame
    assert len(outs) == 1 and outs[0].source == "propagated"
    x0 = outs[0].box[0]
    assert abs(x0 - (box0[0] - 5)) < 1.5, outs[0].box
    print(f"ok  propagation: box moved to x0={x0:.1f}, truth {box0[0] - 5:.1f}")


def test_enhance_helps_dim_blue():
    img = np.full((200, 300, 3), 45, np.uint8)            # dark asphalt (RGB)
    img[:, :, 2] = 48
    cone = (120, 60, 140, 140)
    img[cone[1]:cone[3], cone[0]:cone[2]] = (40, 46, 70)  # dim bluish cone
    img = cv2.GaussianBlur(img, (0, 0), 1.5)              # a bit soft
    pil = Image.fromarray(img)
    enh = np.asarray(enhance_for_detection(pil))
    assert enh.shape == img.shape and enh.dtype == np.uint8

    def blue_excess(a):
        roi = a[cone[1] + 5:cone[3] - 5, cone[0] + 3:cone[2] - 3].astype(float)
        bg = a[10:50, 10:100].astype(float)
        return (roi[..., 2] - roi[..., 0]).mean() - (bg[..., 2] - bg[..., 0]).mean()

    def edge_strength(a):
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(g, cv2.CV_64F).var()

    b0, b1 = blue_excess(img), blue_excess(enh)
    e0, e1 = edge_strength(img), edge_strength(enh)
    assert b1 > b0 * 1.3, (b0, b1)
    assert e1 > e0, (e0, e1)
    print(f"ok  enhance: blue-vs-background {b0:.1f} -> {b1:.1f}, edge strength {e0:.1f} -> {e1:.1f}")


BLUE, YELLOW = 0, 1


def _confirmed_blue_track(box=(100, 100, 120, 140), n=3):
    tf = TrackFiller(min_hits=2, max_gap=3, hi_score=0.5, lo_score=0.25)
    for _ in range(n):
        tf.update([Det(box, 0.9, cls=BLUE)], 640, 400)
    return tf


def test_color_memory_faint_detection_takes_track_color():
    tf = _confirmed_blue_track()
    # glare frame: the detector calls the smeared cone 'yellow' at 0.3
    outs = tf.update([Det((101, 100, 121, 140), 0.3, cls=YELLOW)], 640, 400)
    assert len(outs) == 1, outs
    o = outs[0]
    assert o.cls == BLUE and o.relabeled and o.low_conf and not o.conflict, o
    print("ok  color memory: faint 'yellow' on a blue track becomes blue (flagged relabeled)")


def test_color_memory_confident_disagreement_is_flagged_not_overruled():
    tf = _confirmed_blue_track()
    outs = tf.update([Det((101, 100, 121, 140), 0.8, cls=YELLOW)], 640, 400)
    assert len(outs) == 1 and outs[0].cls == YELLOW and outs[0].conflict and not outs[0].relabeled, outs
    print("ok  color memory: a confident disagreement is kept but flagged color_conflict")


def test_color_memory_propagated_box_keeps_voted_color():
    tf = TrackFiller(min_hits=2, max_gap=3)
    for _ in range(3):
        tf.update([Det((50, 50, 70, 100), 0.9, cls=YELLOW)], 640, 400)
    outs = tf.update([], 640, 400)
    assert len(outs) == 1 and outs[0].source == "propagated" and outs[0].cls == YELLOW, outs
    print("ok  color memory: a carried-forward box keeps the track's color")


def test_color_memory_one_box_per_cone():
    tf = TrackFiller()
    box = (100, 100, 120, 140)
    outs = tf.update([Det(box, 0.8, cls=BLUE), Det(box, 0.6, cls=YELLOW)], 640, 400)
    assert len(outs) == 1 and outs[0].cls == BLUE, outs      # more confident color wins
    print("ok  color memory: two colors on the same cone -> one box, the more confident color")


def test_color_votes_accumulate_and_flip():
    tf = TrackFiller(min_hits=2)
    box = (100, 100, 120, 140)
    tf.update([Det(box, 0.6, cls=BLUE)], 640, 400)
    tf.update([Det(box, 0.6, cls=BLUE)], 640, 400)
    for _ in range(3):                                         # now clearly yellow for 3 frames
        tf.update([Det(box, 0.9, cls=YELLOW)], 640, 400)
    outs = tf.update([Det(box, 0.3, cls=BLUE)], 640, 400)
    assert outs[0].cls == YELLOW and outs[0].relabeled, outs  # 2.7 yellow vs 1.2 blue
    print("ok  color memory: votes accumulate, the majority color wins")


def test_frame_quality_flags():
    rng = np.random.default_rng(1)
    def textured(seed, blur=0.0):
        g = textured_canvas(400, 640, seed)
        rgb = np.stack([g] * 3, -1)
        return cv2.GaussianBlur(rgb, (0, 0), blur) if blur else rgb
    glare = np.zeros((400, 640, 3), np.uint8) + 8                # dark scene ...
    cv2.line(glare, (100, 50), (500, 350), (255, 255, 255), 14)  # ... with a blown-out streak
    sharp_frames = [textured(i) for i in range(6)]
    seq = sharp_frames + [textured(9, blur=4.0), glare]
    m = [frame_quality(f) for f in seq]
    flags = flag_frames(m)
    assert all(f == [] for f in flags[:6]), flags[:6]            # ordinary frames: no flags
    assert "blurry" in flags[6], (flags[6], [x["sharpness"] for x in m])
    assert "glare" in flags[7] and "low_light" in flags[7], flags[7]
    print(f"ok  frame quality: sharp frames clean, blurred -> {flags[6]}, glare frame -> {flags[7]}")


def test_white_sky_is_not_glare():
    day = textured_canvas(400, 640, 5)
    day = np.stack([day + 60] * 3, -1).astype(np.uint8)
    day[:120] = 255                                              # overcast, blown-out sky
    assert "glare" not in flag_frames([frame_quality(day)] * 5)[0]
    print("ok  frame quality: a bright white sky alone is not called glare")


def test_block_split():
    s = assign_blocks_split(100, 0.2, 10)
    assert s.count("val") == 20 and s.count("train") == 80
    # whole blocks: every val frame's block-mates are val too
    for i in range(0, 100, 10):
        assert len(set(s[i:i + 10])) == 1
    assert "val" in assign_blocks_split(8, 0.2, 20)
    print("ok  split: whole blocks, never single frames")


# ---------------------------------------------------------------------------
# end-to-end with a fake SAM3
# ---------------------------------------------------------------------------

class FakeProcessor:
    """Finds blue blobs by color. Score grows with how blue the blob is."""

    def __init__(self, threshold, flip_frames=()):
        self.threshold = threshold
        self.flip_frames = set(flip_frames)   # frames where the blue blob is reported as 'yellow'
        self.n_images = 0

    def set_image(self, image):
        self.n_images += 1                    # (only valid with --no-enhance: one embed per frame)
        return {"img": np.asarray(image.convert("RGB")).astype(np.int16),
                "frame": self.n_images - 1}

    def set_text_prompt(self, state, prompt):
        boxes, scores = [], []
        flipped = state["frame"] in self.flip_frames
        if prompt == ("yellow cone" if flipped else "blue cone"):
            a = state["img"]
            excess = a[..., 2] - a[..., 0]
            mask = (excess > 10).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < 40:
                    continue
                s = float(np.clip(excess[lab == i].mean() / 50.0, 0, 1))
                if s >= self.threshold:
                    boxes.append(np.array([x, y, x + w, y + h], float))
                    scores.append(np.float32(s))
        return {"masks": [None] * len(boxes), "boxes": boxes, "scores": scores}


def make_fake_sam3(flip_frames=()):
    m = types.ModuleType("sam3_mac")
    m.install = lambda: None
    m.build_image_model = lambda confidence_threshold=0.5: (
        None, FakeProcessor(confidence_threshold, flip_frames))
    return m


def make_sequence(folder: Path, n=12, hidden=(4, 5), dim=(8,), glare_first=False):
    """Camera pans: scene moves left 4 px/frame. One blue cone, attached to the scene.
    Frames in `hidden`: cone has the same color as asphalt (nothing can see it).
    Frames in `dim`: cone is faint (score ~0.3)."""
    canvas = textured_canvas(400, 900, seed=3)
    truth = {}
    for i in range(n):
        gray = canvas[:, 4 * i:4 * i + 640]
        rgb = np.stack([gray] * 3, axis=-1).astype(np.int16)
        x0, y0, x1, y1 = 300 - 4 * i, 180, 300 - 4 * i + 24, 180 + 48
        if i in hidden:
            pass
        else:
            excess = 15 if i in dim else 85
            rgb[y0:y1, x0:x1, 2] += excess // 2 + 5
            rgb[y0:y1, x0:x1, 0] -= excess // 2 + 5
        img = np.clip(rgb, 0, 255).astype(np.uint8)
        if glare_first and i == 0:               # dark frame, blown-out streak, no visible cone
            img = np.full_like(img, 8)
            cv2.line(img, (100, 50), (500, 350), (255, 255, 255), 14)
        Image.fromarray(img).save(folder / f"frame_{i:04d}.png")
        truth[f"frame_{i:04d}.png"] = (x0, y0, x1, y1)
    return truth


def run_pipeline(extra_args, flip=(), glare_first=False, review=False):
    sys.modules["sam3_mac"] = make_fake_sam3(flip)
    import autolabel_sam3_temporal as mod
    tmp = Path(tempfile.mkdtemp())
    frames = tmp / "frames"
    frames.mkdir()
    truth = make_sequence(frames, glare_first=glare_first)
    extra = ["--export-review", str(tmp / "rv")] if review else []
    args = mod.build_parser().parse_args(
        ["--frames", str(frames), "--out", str(tmp / "ds"), "--preview", str(tmp / "pv")] + extra + extra_args)
    mod.run(args)
    rows = list(csv.DictReader(open(tmp / "ds" / "label_report.csv")))
    return tmp, truth, rows


def test_end_to_end_propagation():
    tmp, truth, rows = run_pipeline(["--no-enhance", "--block-size", "3"])
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    for i in (4, 5):                                   # cone invisible: must be carried forward
        name = f"frame_{i:04d}.png"
        assert len(by_frame[name]) == 1 and by_frame[name][0]["source"] == "propagated", by_frame.get(name)
        r = by_frame[name][0]
        got = (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
        assert box_iou(got, truth[name]) > 0.6, (got, truth[name])
    name = "frame_0008.png"                            # faint cone: accepted because a track supports it
    assert by_frame[name][0]["source"] == "detected" and by_frame[name][0]["low_conf"] == "True", by_frame[name]
    # labels written, images are the ORIGINAL pixels, class id 0 = blue
    labels = list((tmp / "ds" / "labels").rglob("*.txt"))
    assert len(labels) == 12
    line = (tmp / "ds" / "labels" / "train" / "frame_0000.txt").read_text().split()
    assert line[0] == "0" and len(line) == 5 and all(0 <= float(v) <= 1 for v in line[1:])
    src = np.asarray(Image.open(tmp / "frames" / "frame_0000.png"))
    dst = np.asarray(Image.open(tmp / "ds" / "images" / "train" / "frame_0000.png"))
    assert (src == dst).all()
    assert (tmp / "ds" / "data.yaml").exists() and (tmp / "pv" / "frame_0004.png").exists()
    print("ok  end-to-end: invisible cone carried across 2 frames, faint cone confirmed, originals untouched")


def test_end_to_end_drop_propagated():
    tmp, truth, rows = run_pipeline(["--no-enhance", "--drop-propagated", "--block-size", "3"])
    for split in ("train", "val"):
        p = tmp / "ds" / "labels" / split / "frame_0004.txt"
        if p.exists():
            assert p.read_text().strip() == ""
    assert any(r["source"] == "propagated" for r in rows)   # still listed in the report
    print("ok  end-to-end: --drop-propagated keeps them out of the labels but in the report")


def test_end_to_end_color_memory_and_review_files():
    # frame 8: the faint cone is reported as 'yellow' by the detector
    tmp, truth, rows = run_pipeline(["--no-enhance", "--block-size", "3"], flip=(8,), review=True)
    r8 = [r for r in rows if r["frame"] == "frame_0008.png"]
    assert len(r8) == 1 and r8[0]["concept"] == "blue cone", r8
    assert r8[0]["relabeled"] == "True" and r8[0]["low_conf"] == "True", r8
    assert r8[0]["color_votes"].startswith("blue:"), r8[0]
    # the YOLO label for frame 8 must say class 0 (blue), not 1 (yellow)
    lab = next((tmp / "ds" / "labels").rglob("frame_0008.txt")).read_text().split()
    assert lab[0] == "0", lab
    # review_first.csv lists exactly the frames with guessed boxes / relabeled colors
    rf = list(csv.DictReader(open(tmp / "ds" / "review_first.csv")))
    assert {r["frame"] for r in rf} == {"frame_0004.png", "frame_0005.png", "frame_0008.png"}, rf
    assert all(int(r["priority"]) == 2 for r in rf), rf
    # frame_quality.csv has every frame; the export folder has image + draft label for each review frame
    fq = list(csv.DictReader(open(tmp / "ds" / "frame_quality.csv")))
    assert len(fq) == 12
    assert {p.name for p in (tmp / "rv" / "images").iterdir()} == {r["frame"] for r in rf}
    assert {p.stem for p in (tmp / "rv" / "labels").iterdir()} == {Path(r["frame"]).stem for r in rf}
    print("ok  end-to-end: relabeled color lands in the YOLO label, review_first.csv + export folder correct")


def test_end_to_end_hard_frame_without_labels_is_top_priority():
    tmp, truth, rows = run_pipeline(["--no-enhance", "--block-size", "3"], glare_first=True)
    fq = {r["frame"]: r for r in csv.DictReader(open(tmp / "ds" / "frame_quality.csv"))}
    f0 = fq["frame_0000.png"]
    assert f0["boxes"] == "0" and f0["priority"] == "3", f0
    assert "glare" in f0["flags"] and "hard_frame_no_boxes" in f0["review_reason"], f0
    first = next(csv.DictReader(open(tmp / "ds" / "review_first.csv")))
    assert first["frame"] == "frame_0000.png", first            # worst frame is listed first
    print("ok  end-to-end: a glared frame with zero labels is ranked #1 in review_first.csv")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nall {len(tests)} checks passed")
