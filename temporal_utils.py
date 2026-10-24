"""
temporal_utils.py

Helpers for autolabel_sam3_temporal.py. Two ideas live here:

1. enhance_for_detection()  - make dim / low-contrast blue cones easier for
   SAM3 to see (local contrast + blue boost + sharpening). It is only ever
   applied to the copy of the frame that is shown to the detector. The
   frame that is saved into the training set is always the untouched
   original, so YOLO is trained on the same kind of pixels it will see on
   the car.

2. TrackFiller + FlowField - "look at the previous frame to estimate this
   one". Optical flow tells us how the scene moved between two frames. We
   use it to slide last frame's cone boxes into the current frame. If SAM3
   missed a cone that we were confidently tracking, the slid box fills the
   gap (for a few frames only), and weak SAM3 detections are accepted when
   they line up with a tracked cone.

Nothing in this file needs a GPU or SAM3, so it can be tested on its own
(see test_temporal.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# small conversion helpers
# ---------------------------------------------------------------------------

def to_cpu_numpy(x):
    """Torch tensor (maybe on mps/cuda) or plain value -> numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def box_to_yolo_line(class_id, box, img_w, img_h):
    """(x0,y0,x1,y1) in pixels -> 'class xc yc w h' normalized to 0-1."""
    x0, y0, x1, y1 = box
    xc = (x0 + x1) / 2 / img_w
    yc = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    xc, yc, w, h = (max(0.0, min(1.0, v)) for v in (xc, yc, w, h))
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------------------
# 1. image enhancement for the detector
# ---------------------------------------------------------------------------

def enhance_for_detection(
    img: Image.Image,
    clahe_clip: float = 3.0,
    clahe_grid: int = 8,
    blue_boost: float = 1.5,
    blue_hue_range=(95, 135),   # OpenCV hue is 0-179; pure blue is ~120
    sharpen_amount: float = 1.0,
    sharpen_sigma: float = 1.2,
) -> Image.Image:
    """Return a copy of `img` with dim blue objects made easier to detect.

    Three steps, in this order:
      a) CLAHE on the luminance channel: lifts local contrast in dark areas
         (a dark blue cone on dark asphalt at dusk) without blowing out the sky.
      b) Saturation/brightness boost on blue-ish pixels only.
      c) Unsharp mask: sharpens edges, which also helps with slight blur.

    Side effect to be aware of: the blue sky is also blue-ish, so it gets
    boosted too. That is harmless for a text-prompted detector but means the
    output is for the detector's eyes only - never save it as training data.
    """
    bgr = pil_to_bgr(img)

    # a) local contrast on luminance only (keeps colors from shifting)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    l = clahe.apply(l)
    bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # b) boost blue-ish pixels
    if blue_boost != 1.0:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = blue_hue_range
        mask = ((h >= lo) & (h <= hi) & (s > 40)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)  # soft edges, no halos
        gain = blue_boost - 1.0
        hsv[..., 1] = np.clip(s * (1.0 + gain * mask), 0, 255)
        hsv[..., 2] = np.clip(v * (1.0 + 0.5 * gain * mask), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # c) unsharp mask
    if sharpen_amount > 0:
        blur = cv2.GaussianBlur(bgr, (0, 0), sharpen_sigma)
        bgr = cv2.addWeighted(bgr, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

    return bgr_to_pil(bgr)


# ---------------------------------------------------------------------------
# box math + detections
# ---------------------------------------------------------------------------

@dataclass
class Det:
    box: tuple          # (x0, y0, x1, y1) pixels
    score: float
    src: str = "orig"   # "orig" = found on the original frame, "enh" = on the enhanced copy
    cls: int = 0        # which color the detector called it (index into the concept list)


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def center_dist_ratio(a, b) -> float:
    """Distance between box centers, divided by the biggest box side.
    Far-away cones are only ~10 px wide, where IoU is brutally sensitive to a
    2 px shift, so matching also falls back on this."""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return math.hypot(acx - bcx, acy - bcy) / scale


def nms_dets(dets: list, iou_thr: float = 0.5) -> list:
    """Keep the highest-scoring box among overlapping ones."""
    kept = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(box_iou(d.box, k.box) < iou_thr for k in kept):
            kept.append(d)
    return kept


# ---------------------------------------------------------------------------
# 2. optical flow: "how did the scene move since the previous frame?"
# ---------------------------------------------------------------------------

class FlowField:
    """Dense optical flow, computed on a shrunken copy of the frame to keep
    it fast. `scale` remembers how much it was shrunk."""

    def __init__(self, flow: np.ndarray, scale: float):
        self.flow = flow
        self.scale = scale

    def median_in_box(self, box, pad: float = 0.5):
        """Robust (median) motion of the region around `box`, in full-res
        pixels. `pad` widens the region by that fraction of the box size on
        every side, because a 10 px cone has too little texture on its own
        for reliable flow - the asphalt around it moves with it."""
        h, w = self.flow.shape[:2]
        x0, y0, x1, y1 = (v * self.scale for v in box)
        bw, bh = x1 - x0, y1 - y0
        xi0 = int(max(0, math.floor(x0 - bw * pad)))
        xi1 = int(min(w, math.ceil(x1 + bw * pad)))
        yi0 = int(max(0, math.floor(y0 - bh * pad)))
        yi1 = int(min(h, math.ceil(y1 + bh * pad)))
        if xi1 <= xi0 or yi1 <= yi0:
            return 0.0, 0.0
        patch = self.flow[yi0:yi1, xi0:xi1].reshape(-1, 2)
        dx, dy = np.median(patch, axis=0)
        return float(dx) / self.scale, float(dy) / self.scale

    def median_magnitude(self) -> float:
        mag = np.hypot(self.flow[..., 0], self.flow[..., 1])
        return float(np.median(mag)) / self.scale


def estimate_flow(prev_gray: np.ndarray, cur_gray: np.ndarray, scale: float = 0.5) -> FlowField:
    """Farneback dense optical flow from prev -> cur (both uint8 grayscale)."""
    if scale != 1.0:
        prev_gray = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cur_gray = cv2.resize(cur_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    return FlowField(flow, scale)


# ---------------------------------------------------------------------------
# tracking: carry confident detections forward, fill short gaps
# ---------------------------------------------------------------------------

@dataclass
class Out:
    box: tuple
    score: float
    source: str          # "detected" or "propagated"
    enhanced: bool       # detection only found on the enhanced copy
    low_conf: bool       # accepted below hi_score because a track supported it
    track_id: int
    cls: int = 0             # final color of this box (may differ from what the detector said)
    relabeled: bool = False  # a faint detection's color was overruled by the track's color vote
    conflict: bool = False   # a confident detection disagrees with the track's color vote
    votes: dict = field(default_factory=dict)   # {class id: summed confidence} behind the vote


@dataclass
class _Track:
    id: int
    box: np.ndarray
    score: float
    hits: int = 1
    misses: int = 0
    votes: dict = field(default_factory=dict)   # {class id: summed confidence of confident detections}

    @property
    def cls(self) -> int:
        """The color this cone has been called most (weighted by confidence)."""
        return max(self.votes, key=self.votes.get) if self.votes else 0


class TrackFiller:
    """Tracks cones of ALL tracked colors together (one physical cone = one
    track, whatever color the detector called it in a given frame). Per frame,
    call in this order:

        predict(flow)               slide old boxes into the current frame
        unmatched_confirmed(dets)   (optional) which tracked cones did the
                                    detector just fail to find?
        update(dets, w, h)          decide the final boxes for this frame

    Rules, in plain words:
      * A detection with score >= hi_score is always accepted, and starts a
        new track. Its color is added to the track's color VOTE.
      * A weaker detection (lo_score <= score < hi_score) is accepted ONLY if
        it lines up with a *confirmed* track (one seen >= min_hits times).
        "The last frames saw a cone here, and this faint blob agrees."
        Its color is taken from the track's vote, not from the faint detection
        (a smeared blue cone that SAM3 calls 'yellow' at 0.3 stays blue).
      * A confirmed track that gets no detection is carried forward as a
        'propagated' box (in the track's voted color) for at most max_gap
        frames in a row, then dropped.
      * A track that has not been confirmed yet is dropped as soon as it is missed.
      * A CONFIDENT detection whose color disagrees with a confirmed track is
        kept as the detector said it, but flagged `conflict` for human review.
    """

    def __init__(self, min_hits=2, max_gap=3, match_iou=0.2, match_center=0.6,
                 hi_score=0.5, lo_score=0.25):
        self.min_hits = min_hits
        self.max_gap = max_gap
        self.match_iou = match_iou
        self.match_center = match_center
        self.hi_score = hi_score
        self.lo_score = lo_score
        self.tracks: list[_Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []

    def _candidates(self, dets):
        # One box per physical cone: if two colors were both reported on the
        # same object, keep the more confident one.
        return nms_dets([d for d in dets if d.score >= self.lo_score], iou_thr=0.6)

    # -- step 1 -------------------------------------------------------------
    def predict(self, flow):
        if flow is None:
            return
        for t in self.tracks:
            dx, dy = flow.median_in_box(tuple(t.box))
            t.box = t.box + np.array([dx, dy, dx, dy])

    # -- matching -----------------------------------------------------------
    def _associate(self, cands):
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(cands):
                iou = box_iou(t.box, d.box)
                ratio = center_dist_ratio(t.box, d.box)
                if iou >= self.match_iou or ratio <= self.match_center:
                    pairs.append((iou + (1.0 - min(ratio, 1.0)) * 0.25, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d, matches = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        un_d = [i for i in range(len(cands)) if i not in used_d]
        un_t = [i for i in range(len(self.tracks)) if i not in used_t]
        return matches, un_d, un_t

    # -- step 2 (optional) ---------------------------------------------------
    def unmatched_confirmed(self, dets, cls=None):
        """Confirmed tracks that no detection lines up with. With `cls`, only
        tracks whose voted color is that class."""
        _, _, un_t = self._associate(self._candidates(dets))
        return [self.tracks[i] for i in un_t
                if self.tracks[i].hits >= self.min_hits
                and (cls is None or self.tracks[i].cls == cls)]

    # -- step 3 -------------------------------------------------------------
    def update(self, dets, img_w, img_h):
        cands = self._candidates(dets)
        matches, un_d, un_t = self._associate(cands)
        outputs: list[Out] = []
        matched_tracks = set()

        for ti, di in matches:
            t, d = self.tracks[ti], cands[di]
            confirmed = t.hits >= self.min_hits
            if d.score >= self.hi_score:
                cls, relabeled = d.cls, False
                conflict = confirmed and t.cls != d.cls      # judged BEFORE this vote
                t.votes[d.cls] = t.votes.get(d.cls, 0.0) + d.score
            elif confirmed:
                cls, conflict = t.cls, False                  # faint: trust the track's color
                relabeled = d.cls != t.cls
            else:
                continue   # a faint detection next to a not-yet-confirmed track. Ignore it.
            t.box = np.array(d.box, dtype=float)
            t.hits += 1
            t.misses = 0
            t.score = d.score
            matched_tracks.add(ti)
            outputs.append(Out(tuple(d.box), d.score, "detected", d.src == "enh",
                               d.score < self.hi_score, t.id, cls, relabeled, conflict,
                               dict(t.votes)))

        new_tracks = []
        for di in un_d:
            d = cands[di]
            if d.score >= self.hi_score:
                t = _Track(self._next_id, np.array(d.box, dtype=float), d.score,
                           votes={d.cls: d.score})
                self._next_id += 1
                new_tracks.append(t)
                outputs.append(Out(tuple(d.box), d.score, "detected", d.src == "enh",
                                   False, t.id, d.cls, False, False, dict(t.votes)))

        survivors = []
        for ti, t in enumerate(self.tracks):
            if ti in matched_tracks:
                survivors.append(t)
                continue
            t.misses += 1
            if t.hits < self.min_hits or t.misses > self.max_gap:
                continue
            clipped = self._clip(t.box, img_w, img_h)
            if clipped is None:
                continue
            survivors.append(t)
            outputs.append(Out(clipped, t.score * (0.9 ** t.misses), "propagated",
                               False, False, t.id, t.cls, False, False, dict(t.votes)))

        self.tracks = survivors + new_tracks
        return outputs

    @staticmethod
    def _clip(box, img_w, img_h):
        x0, y0, x1, y1 = (float(v) for v in box)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (0 <= cx < img_w and 0 <= cy < img_h) or area <= 0:
            return None  # cone drifted out of view
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(float(img_w), x1), min(float(img_h), y1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            return None
        if (cx1 - cx0) * (cy1 - cy0) < 0.5 * area:
            return None  # mostly outside the frame
        return (cx0, cy0, cx1, cy1)


# ---------------------------------------------------------------------------
# train/val split that does not leak neighbouring frames
# ---------------------------------------------------------------------------

def assign_blocks_split(n: int, val_split: float, block_size: int) -> list:
    """Split n time-ordered frames into train/val by contiguous BLOCKS.

    With dense video frames, shuffling single frames puts near-identical
    neighbours on both sides of the split, which makes validation scores look
    better than they really are. Whole blocks keep the two sets honestly apart.
    """
    if n < 2 or val_split <= 0:
        return ["train"] * n
    block_size = max(1, min(block_size, n // 5)) if n >= 5 else 1
    num_blocks = math.ceil(n / block_size)
    n_val = min(max(1, round(num_blocks * val_split)), num_blocks - 1)
    val_blocks = {int((k + 0.5) * num_blocks / n_val) for k in range(n_val)}
    return ["val" if (i // block_size) in val_blocks else "train" for i in range(n)]


# ---------------------------------------------------------------------------
# frame quality: which frames should a human look at?
# ---------------------------------------------------------------------------

def frame_quality(rgb: np.ndarray) -> dict:
    """Cheap numbers describing how hard a frame is for a detector.

    brightness  mean gray level (0-255)
    median      median gray level; a night frame has a very low median
    sat_pct     % of pixels with some channel >= 250 (blown out: glare, lamps)
    dark_pct    % of pixels with every channel < 20 (nearly black)
    sharpness   variance of the Laplacian divided by the image's own variance.
                Dividing makes it comparable between bright and dark frames;
                only compare it between frames of the SAME sequence.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mx = rgb.max(axis=2)
    lap = float(cv2.Laplacian(small, cv2.CV_64F).var())
    var = float(small.astype(np.float64).var())
    return {
        "brightness": float(gray.mean()),
        "median": float(np.median(gray)),
        "sat_pct": float((mx >= 250).mean() * 100.0),
        "dark_pct": float((mx < 20).mean() * 100.0),
        "sharpness": lap / (var + 1e-6),
    }


def flag_frames(metrics: list, glare_sat_pct=1.0, glare_dark_pct=20.0,
                low_light_median=25.0, blur_ratio=0.5) -> list:
    """Turn each frame's numbers into a list of flags (strings).

      glare      >= glare_sat_pct % blown-out pixels AND >= glare_dark_pct %
                 nearly-black pixels: a high-contrast scene (headlight or lamp
                 in the dark). A bright white sky alone does not trip it.
      low_light   median gray level below low_light_median.
      blurry     sharpness below blur_ratio x the median sharpness of the
                 whole sequence (needs >= 5 frames to be meaningful).

    These thresholds are starting points measured on one night frame; tune
    them on your own footage (the numbers are all in frame_quality.csv).
    """
    sharp = [m["sharpness"] for m in metrics]
    med_sharp = float(np.median(sharp)) if len(sharp) >= 5 else None
    flags = []
    for m in metrics:
        f = []
        if m["sat_pct"] >= glare_sat_pct and m["dark_pct"] >= glare_dark_pct:
            f.append("glare")
        if m["median"] < low_light_median:
            f.append("low_light")
        if med_sharp is not None and m["sharpness"] < blur_ratio * med_sharp:
            f.append("blurry")
        flags.append(f)
    return flags
