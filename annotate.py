#!/usr/bin/env python3
"""Build labels.json for the F05 dense 4K tracking assignment.

Detection uses a YOLO11-L model fine-tuned on VisDrone (aerial cars,
two-wheelers, and pedestrians). Every frame is detected, so the boxes are
not a sparse keyframe interpolation. A homography-compensated tracker then
gives one physical object one instance_id until it leaves the image.
Detector gaps of up to 8 frames are filled by linear interpolation.
Longer gaps stay empty (the object is treated as fully hidden) and the
id resumes only if it reappears near the camera-compensated position.

Usage:
    python3 annotate.py detect     # write detections.npz
    python3 annotate.py track      # write labels.json from detections.npz
    python3 annotate.py qa         # draw sample frames
    python3 annotate.py all
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parent
FRAMES = Path("/Users/divyanshu/Desktop/f05_frames")
WEIGHTS = ROOT / "models" / "yolo11l-visdrone.pt"
DET_PATH = ROOT / "detections.npz"
LABELS_PATH = ROOT / "labels.json"
QA_DIR = ROOT / "qa"

IMG_W, IMG_H = 3840, 2160
N_FRAMES = 1199
# VisDrone: 0 pedestrian, 2 bicycle, 3 car, 4 van, 5 truck,
# 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor.
# Class 1 ("people") is a crowd box and is intentionally not used.
VEHICLE_SUB = {2, 3, 4, 5, 6, 7, 8, 9}
TWO_WHEEL = {2, 6, 7, 9}
FOUR_WHEEL = {3, 4, 5, 8}
PERSON_SUB = 0
DETECT_CLASSES = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def frame_path(sequence: int) -> Path:
    return FRAMES / f"frame_{sequence:05d}.jpg"


def xywh_to_xyxy(dets: np.ndarray) -> np.ndarray:
    out = dets[:, :4].copy()
    out[:, 2] = dets[:, 0] + dets[:, 2]
    out[:, 3] = dets[:, 1] + dets[:, 3]
    return out


def nms_rows(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(dets) == 0:
        return dets
    xyxy = torch.tensor(xywh_to_xyxy(dets), dtype=torch.float32)
    scores = torch.tensor(dets[:, 4], dtype=torch.float32)
    keep = nms(xyxy, scores, iou_thr).numpy()
    return dets[keep]


def box_area(box) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def inter_area(a, b) -> float:
    ax1, ay1, aw, ah = [float(v) for v in a[:4]]
    bx1, by1, bw, bh = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a, b) -> float:
    inter = inter_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xywh boxes. a is (N, 4), b is (M, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def center(box):
    return float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0


def dist(a, b) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def inside_fraction(box) -> float:
    area = box_area(box)
    if area <= 1:
        return 0.0
    x, y, w, h = [float(v) for v in box[:4]]
    ix1, iy1 = max(0.0, x), max(0.0, y)
    ix2, iy2 = min(float(IMG_W), x + w), min(float(IMG_H), y + h)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / area


def touches_border(box, margin: float = 2.0) -> bool:
    x, y, w, h = [float(v) for v in box[:4]]
    return x <= margin or y <= margin or x + w >= IMG_W - margin or y + h >= IMG_H - margin


def clip_box(box):
    x, y, w, h = [float(v) for v in box[:4]]
    x2, y2 = x + w, y + h
    trunc = touches_border((x, y, w, h), 1.5)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(IMG_W), x2), min(float(IMG_H), y2)
    return np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32), trunc


def warp_boxes(boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp xywh boxes by a homography that maps the previous frame onto this one."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    out = np.asarray(boxes, dtype=np.float32).copy()
    if np.allclose(H, np.eye(3)):
        return out
    n = len(out)
    x, y, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    ones = np.ones(n, dtype=np.float64)
    corners = np.stack(
        [
            np.stack([x, y, ones], axis=1),
            np.stack([x + w, y, ones], axis=1),
            np.stack([x + w, y + h, ones], axis=1),
            np.stack([x, y + h, ones], axis=1),
        ],
        axis=0,
    )
    wp = (H @ corners.reshape(-1, 3).T)
    wp = wp[:2] / np.clip(wp[2], 1e-6, None)
    wp = wp.T.reshape(4, n, 2)
    x1 = wp[:, :, 0].min(axis=0)
    y1 = wp[:, :, 1].min(axis=0)
    x2 = wp[:, :, 0].max(axis=0)
    y2 = wp[:, :, 1].max(axis=0)
    return np.stack(
        [x1, y1, np.maximum(1.0, x2 - x1), np.maximum(1.0, y2 - y1)], axis=1
    ).astype(np.float32)


def warp_box(box, H: np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in box[:4]]
    corners = np.array(
        [[x, y, 1.0], [x + w, y, 1.0], [x + w, y + h, 1.0], [x, y + h, 1.0]],
        dtype=np.float64,
    ).T
    wp = H @ corners
    wp = wp[:2] / np.clip(wp[2], 1e-6, None)
    x1, y1 = float(wp[0].min()), float(wp[1].min())
    x2, y2 = float(wp[0].max()), float(wp[1].max())
    return np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)


def compute_homography(prev_bgr, curr_bgr, mask_boxes: np.ndarray) -> np.ndarray:
    """Homography mapping previous-frame pixels onto the current frame."""
    height, width = prev_bgr.shape[:2]
    scale = 960.0 / width
    size = (960, int(round(height * scale)))
    g0 = cv2.cvtColor(cv2.resize(prev_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(curr_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=700, qualityLevel=0.01, minDistance=7, blockSize=7)
    eye = np.eye(3, dtype=np.float64)
    if pts is None or len(pts) < 40:
        return eye
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        g0,
        g1,
        pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    ok = status.reshape(-1).astype(bool)
    src = pts.reshape(-1, 2)[ok]
    dst = nxt.reshape(-1, 2)[ok]
    if mask_boxes is not None and len(src) and len(mask_boxes):
        keep = np.ones(len(src), dtype=bool)
        mb = mask_boxes.copy()
        mb[:, 0] *= scale
        mb[:, 1] *= scale
        mb[:, 2] *= scale
        mb[:, 3] *= scale
        # Expand masks a little so corners of moving vehicles are excluded.
        mb[:, 0] -= 4
        mb[:, 1] -= 4
        mb[:, 2] += 8
        mb[:, 3] += 8
        for x, y, w, h in mb:
            inside = (src[:, 0] >= x) & (src[:, 0] <= x + w) & (src[:, 1] >= y) & (src[:, 1] <= y + h)
            keep &= ~inside
        if int(keep.sum()) >= 40:
            src, dst = src[keep], dst[keep]
    if len(src) < 30:
        return eye
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inliers is None or int(inliers.sum()) < 30:
        return eye
    if abs(H[0, 0] - 1) > 0.06 or abs(H[1, 1] - 1) > 0.06 or abs(H[0, 1]) > 0.06 or abs(H[1, 0]) > 0.06:
        return eye
    if abs(H[0, 2]) > 60 or abs(H[1, 2]) > 60:
        return eye
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(S) @ H @ S


def filter_occupants(dets: np.ndarray) -> np.ndarray:
    """Drop people who read as part of a vehicle, not as their own figure.

    Occupants of cars, vans, trucks, and buses are removed.
    A rider is kept only when their box clearly sticks out above the
    two-wheeler; otherwise the person and vehicle are one blob and only
    the vehicle is annotated.
    """
    if len(dets) == 0:
        return dets
    persons = dets[:, 5] == PERSON_SUB
    if not persons.any():
        return dets
    keep = np.ones(len(dets), dtype=bool)
    vehicles = np.where(~persons)[0]
    for i in np.where(persons)[0]:
        p = dets[i]
        pa = box_area(p)
        if pa < 1:
            keep[i] = False
            continue
        for j in vehicles:
            v = dets[j]
            cover = inter_area(p, v) / pa
            sub = int(v[5])
            if sub in FOUR_WHEEL and cover > 0.50:
                keep[i] = False
                break
            if sub in TWO_WHEEL and cover > 0.60:
                # Clearly separate: the person's head/torso extends above the vehicle.
                if p[1] < v[1] - 0.22 * v[3]:
                    continue
                keep[i] = False
                break
    return dets[keep]


def prepare_dets(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
    if len(xyxy) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    wh = np.stack([xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
    dets = np.stack([xyxy[:, 0], xyxy[:, 1], wh[:, 0], wh[:, 1], conf, cls], axis=1).astype(np.float32)
    # Drop noise and the crowd class if it ever slips through.
    ok = (dets[:, 2] >= 4) & (dets[:, 3] >= 6) & (dets[:, 5] != 1) & (dets[:, 5] != 10)
    ok &= (dets[:, 2] / np.maximum(dets[:, 3], 1) < 8) & (dets[:, 3] / np.maximum(dets[:, 2], 1) < 8)
    dets = dets[ok]
    persons = dets[dets[:, 5] == PERSON_SUB]
    vehicles = dets[dets[:, 5] != PERSON_SUB]
    persons = nms_rows(persons, 0.40)
    vehicles = nms_rows(vehicles, 0.45)
    if len(persons) and len(vehicles):
        dets = np.concatenate([vehicles, persons], axis=0)
    elif len(vehicles):
        dets = vehicles
    else:
        dets = persons
    return filter_occupants(dets)


def detect() -> None:
    from ultralytics import YOLO

    if not WEIGHTS.exists():
        raise SystemExit(f"Missing weights: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    # Warm the MPS graph on frame 1 before timing the full clip.
    warm = cv2.imread(str(frame_path(1)))
    model.predict(
        warm,
        imgsz=1920,
        conf=0.15,
        iou=0.5,
        max_det=1200,
        device="mps",
        classes=DETECT_CLASSES,
        verbose=False,
    )
    print("warmup done", flush=True)

    all_dets = []
    offsets = [0]
    homographies = np.zeros((N_FRAMES, 3, 3), dtype=np.float64)
    homographies[0] = np.eye(3)
    prev = None
    h_fallback = 0
    t0 = time.time()
    start_seq = 1
    if DET_PATH.exists():
        prev_blob = np.load(DET_PATH, allow_pickle=False)
        if int(prev_blob["n_done"]) >= N_FRAMES:
            print("detections already complete", flush=True)
            return
        start_seq = int(prev_blob["n_done"]) + 1
        all_dets = [prev_blob["data"]] if len(prev_blob["data"]) else []
        offsets = prev_blob["offsets"][:start_seq].tolist()
        homographies[: start_seq - 1] = prev_blob["H"][: start_seq - 1]
        if start_seq > 1:
            prev = cv2.imread(str(frame_path(start_seq - 1)))
        print(f"resuming at frame {start_seq}", flush=True)

    for seq in range(start_seq, N_FRAMES + 1):
        im = cv2.imread(str(frame_path(seq)))
        if im is None:
            raise SystemExit(f"Could not read {frame_path(seq)}")
        result = model.predict(
            im,
            imgsz=1920,
            conf=0.15,
            iou=0.5,
            max_det=1200,
            device="mps",
            classes=DETECT_CLASSES,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        else:
            dets = prepare_dets(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
        if prev is not None:
            H = compute_homography(prev, im, dets[:, :4] if len(dets) else np.zeros((0, 4), np.float32))
            if np.allclose(H, np.eye(3)):
                h_fallback += 1
            homographies[seq - 1] = H
        prev = im
        all_dets.append(dets)
        offsets.append(offsets[-1] + len(dets))
        if seq % 50 == 0:
            torch.mps.empty_cache()
        if seq % 25 == 0 or seq == N_FRAMES:
            elapsed = time.time() - t0
            done = seq - start_seq + 1
            rate = done / max(elapsed, 1e-6)
            left = (N_FRAMES - seq) / max(rate, 1e-6)
            n = len(dets)
            print(
                f"frame {seq}/{N_FRAMES} boxes {n}  {rate:.2f} fps  eta {left/60:.1f} min  H-fallback {h_fallback}",
                flush=True,
            )
        if seq % 100 == 0 or seq == N_FRAMES:
            data = np.concatenate(all_dets, axis=0) if all_dets else np.zeros((0, 6), np.float32)
            np.savez_compressed(
                DET_PATH,
                data=data.astype(np.float32),
                offsets=np.array(offsets, dtype=np.int32),
                H=homographies,
                n_done=np.int32(seq),
            )
    print(f"wrote {DET_PATH}", flush=True)


class Track:
    __slots__ = (
        "box",
        "subcls",
        "conf",
        "max_conf",
        "start",
        "hits",
        "miss",
        "vx",
        "vy",
        "confirmed",
        "obs",
        "truncated_ever",
        "super",
    )

    def __init__(self, box, subcls, conf, frame):
        self.box = np.asarray(box[:4], dtype=np.float32).copy()
        self.subcls = int(subcls)
        self.conf = float(conf)
        self.max_conf = float(conf)
        self.start = int(frame)
        self.hits = 1
        self.miss = 0
        self.vx = 0.0
        self.vy = 0.0
        self.confirmed = float(conf) >= 0.55
        self.obs: dict[int, np.ndarray] = {int(frame): self.box.copy()}
        self.truncated_ever = bool(touches_border(self.box))
        self.super = 1 if self.subcls == PERSON_SUB else 0


def associate_fast(preds, supers, dets, det_super, track_idx, det_idx):
    """Hungarian match of track indices to detection indices. Returns pairs."""
    if len(track_idx) == 0 or len(det_idx) == 0:
        return []
    P = preds[track_idx]
    D = dets[det_idx, :4]
    overlap = iou_matrix(P, D)
    pc = np.stack([P[:, 0] + P[:, 2] * 0.5, P[:, 1] + P[:, 3] * 0.5], axis=1)
    dc = np.stack([D[:, 0] + D[:, 2] * 0.5, D[:, 1] + D[:, 3] * 0.5], axis=1)
    delta = pc[:, None, :] - dc[None, :, :]
    distance = np.sqrt((delta * delta).sum(axis=-1))
    gate = np.maximum(18.0, 0.50 * np.maximum(P[:, 2], P[:, 3]))
    area_ratio = (D[:, 2] * D[:, 3])[None, :] / np.maximum((P[:, 2] * P[:, 3])[:, None], 1.0)
    same = supers[track_idx, None] == det_super[det_idx][None, :]
    size_ok = (area_ratio >= 0.34) & (area_ratio <= 2.9)
    close = distance <= gate[:, None]
    ok = same & size_ok & ((overlap >= 0.12) | ((overlap >= 0.02) & close & (area_ratio >= 0.45) & (area_ratio <= 2.3)))
    # A very distant box can share a sliver of IoU in a crowd. Require the
    # centers to be in the same neighbourhood.
    ok &= distance <= np.maximum(gate[:, None] * 2.2, 28.0)
    cost = np.where(ok, (1.0 - overlap) + 0.10 * (distance / gate[:, None]), 1e4)
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(rows, cols):
        if cost[r, c] < 10:
            pairs.append((int(track_idx[r]), int(det_idx[c])))
    return pairs


def select_frame_dets(dets: np.ndarray) -> np.ndarray:
    """Score gates. The low tail can extend a track but a higher score is
    required, later, to mint a new instance_id."""
    if len(dets) == 0:
        return dets
    conf = dets[:, 4]
    sub = dets[:, 5].astype(int)
    keep = np.zeros(len(dets), dtype=bool)
    keep |= (sub == PERSON_SUB) & (conf >= 0.18)
    keep |= np.isin(sub, list(FOUR_WHEEL)) & (conf >= 0.20)
    keep |= np.isin(sub, list(TWO_WHEEL)) & (conf >= 0.22)
    return dets[keep]


def spawn_ok(det) -> bool:
    sub = int(det[5])
    conf = float(det[4])
    border = touches_border(det)
    if sub == PERSON_SUB:
        return conf >= (0.22 if border else 0.27)
    if sub in TWO_WHEEL:
        return conf >= (0.25 if border else 0.30)
    return conf >= (0.22 if border else 0.26)


def track() -> None:
    blob = np.load(DET_PATH)
    if int(blob["n_done"]) < N_FRAMES:
        raise SystemExit(f"detections incomplete ({int(blob['n_done'])}/{N_FRAMES}); run detect first")
    data = blob["data"]
    offsets = blob["offsets"]
    Hs = blob["H"]

    per_frame = []
    for seq in range(1, N_FRAMES + 1):
        a, b = int(offsets[seq - 1]), int(offsets[seq])
        per_frame.append(select_frame_dets(data[a:b]))

    active: list[Track] = []
    finished: list[Track] = []
    t0 = time.time()

    for seq in range(1, N_FRAMES + 1):
        dets = per_frame[seq - 1]
        n_active = len(active)
        if n_active:
            boxes = np.stack([t.box for t in active])
            H = Hs[seq - 1] if seq > 1 else np.eye(3)
            warped = warp_boxes(boxes, H)
            vel = np.array([[t.vx, t.vy] for t in active], dtype=np.float32)
            preds = warped.copy()
            preds[:, 0] += vel[:, 0]
            preds[:, 1] += vel[:, 1]
            bad = ~np.isfinite(preds).all(axis=1) | (preds[:, 2] > 1500) | (preds[:, 3] > 1500)
            if bad.any():
                preds[bad] = warped[bad]
            supers = np.array([t.super for t in active], dtype=np.int32)
        else:
            warped = np.zeros((0, 4), np.float32)
            preds = warped
            supers = np.zeros(0, np.int32)

        if len(dets):
            det_super = np.where(dets[:, 5].astype(int) == PERSON_SUB, 1, 0).astype(np.int32)
            high = dets[:, 4] >= 0.34
        else:
            det_super = np.zeros(0, np.int32)
            high = np.zeros(0, dtype=bool)

        all_idx = np.arange(n_active)
        pairs = associate_fast(preds, supers, dets, det_super, all_idx, np.flatnonzero(high))
        matched = {ti: dj for ti, dj in pairs}
        used = np.zeros(len(dets), dtype=bool)
        for dj in matched.values():
            used[dj] = True
        unmatched = np.array([i for i in all_idx if i not in matched], dtype=np.int32)
        low_idx = np.flatnonzero(~high & ~used) if len(dets) else np.zeros(0, np.int32)
        pairs2 = associate_fast(preds, supers, dets, det_super, unmatched, low_idx)
        for ti, dj in pairs2:
            matched[ti] = dj
            used[dj] = True

        still: list[Track] = []
        for i, t in enumerate(active):
            dj = matched.get(i)
            if dj is None:
                t.miss += 1
                t.vx *= 0.55
                t.vy *= 0.55
                pred = preds[i]
                frac = inside_fraction(pred)
                border = touches_border(pred) or frac < 0.97
                # Left the image: retire the id. A later re-entry is a new object.
                if frac < 0.28:
                    if t.confirmed and len(t.obs) >= 2:
                        finished.append(t)
                    continue
                limit = 8 if border else 48
                if t.miss > limit or (not t.confirmed and t.miss >= 2):
                    if t.confirmed and len(t.obs) >= 2 and t.miss > limit:
                        finished.append(t)
                    continue
                t.box = pred
                still.append(t)
                continue
            det = dets[dj]
            new_box = det[:4].astype(np.float32)
            t.vx = 0.45 * t.vx + 0.55 * float(new_box[0] - warped[i, 0])
            t.vy = 0.45 * t.vy + 0.55 * float(new_box[1] - warped[i, 1])
            t.box = new_box
            t.obs[seq] = new_box.copy()
            t.miss = 0
            t.hits += 1
            t.conf = float(det[4])
            t.max_conf = max(t.max_conf, t.conf)
            t.truncated_ever = t.truncated_ever or touches_border(new_box)
            if t.hits >= 3 or t.conf >= 0.55:
                t.confirmed = True
            still.append(t)
        active = still

        for j, det in enumerate(dets):
            if used[j] or not spawn_ok(det):
                continue
            active.append(Track(det[:4], det[5], det[4], seq))

        if seq % 100 == 0 or seq == N_FRAMES:
            elapsed = time.time() - t0
            print(
                f"track frame {seq}/{N_FRAMES}  active {len(active)}  finished {len(finished)}  "
                f"{elapsed:.1f}s",
                flush=True,
            )

    for t in active:
        if t.confirmed and len(t.obs) >= 2:
            finished.append(t)

    print(f"raw tracks {len(finished)}", flush=True)
    finished = split_implausible(finished, Hs)
    finished = [t for t in finished if keep_track(t)]
    for t in finished:
        t.obs = smooth_track(t.obs)
    finished.sort(key=lambda t: (t.start, float(t.obs[min(t.obs)][1]), float(t.obs[min(t.obs)][0])))

    # Short gaps are detector flicker: interpolate them. Longer gaps are
    # full occlusion or a missed object. Leave those frames empty and keep
    # the same id for when the object is visible again. An interpolated box
    # that would be almost completely covered is treated as fully hidden.
    series: list[dict[int, np.ndarray]] = []
    supers = []
    for t in finished:
        filled = dict(t.obs)
        frames = sorted(t.obs)
        for a, b in zip(frames, frames[1:]):
            gap = b - a - 1
            if gap <= 0 or gap > 8:
                continue
            ba, bb = t.obs[a], t.obs[b]
            for k in range(1, gap + 1):
                alpha = k / (gap + 1)
                filled[a + k] = ((1 - alpha) * ba + alpha * bb).astype(np.float32)
        series.append(filled)
        supers.append(t.super)

    per_frame_items: list[list[tuple]] = [[] for _ in range(N_FRAMES + 1)]
    for idx, (filled, supercls) in enumerate(zip(series, supers), start=1):
        for seq, box in filled.items():
            clipped, trunc = clip_box(box)
            if clipped[2] < 3 or clipped[3] < 4:
                continue
            if inside_fraction(clipped) < 0.15:
                continue
            per_frame_items[seq].append((idx, supercls, clipped, trunc, seq in t_obs_keys(finished[idx - 1])))

    # Drop interpolated boxes that are nearly fully covered: the object is
    # hidden, and the spec says to write no box on those frames.
    for seq in range(1, N_FRAMES + 1):
        items = per_frame_items[seq]
        if len(items) < 2:
            continue
        boxes = np.stack([it[2] for it in items])
        cover = cover_matrix(boxes)
        keep = []
        for i, item in enumerate(items):
            interpolated = not item[4]
            if interpolated and cover[i] >= 0.65:
                continue
            keep.append(item)
        per_frame_items[seq] = keep

    frames_out = []
    for seq in range(1, N_FRAMES + 1):
        items = per_frame_items[seq]
        detections = []
        if items:
            boxes = np.stack([it[2] for it in items])
            cover = cover_matrix(boxes)
            for i, (iid, supercls, box, trunc, _observed) in enumerate(items):
                c = float(cover[i])
                if c >= 0.55:
                    occ = 2
                elif c >= 0.18:
                    occ = 1
                else:
                    occ = 0
                x, y, w, h = [round(float(v), 1) for v in box]
                detections.append(
                    {
                        "box": [x, y, w, h],
                        "class_id": int(supercls),
                        "instance_id": int(iid),
                        "occlusion": int(occ),
                        "truncated": bool(trunc),
                    }
                )
        detections.sort(key=lambda d: d["instance_id"])
        frames_out.append({"sequence": seq, "detections": detections})

    payload = {
        "clip": "F05_dense_4k",
        "image_size": [IMG_W, IMG_H],
        "fps": 60,
        "frame_count": N_FRAMES,
        "annotator": "divyanshu",
        "frames": frames_out,
    }
    with LABELS_PATH.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    validate(payload)
    n_ids = len(finished)
    n_boxes = sum(len(fr["detections"]) for fr in frames_out)
    n_people = sum(1 for t in finished if t.super == 1)
    print(
        f"wrote {LABELS_PATH}  tracks {n_ids} ({n_people} persons)  boxes {n_boxes}  "
        f"mean/frame {n_boxes / N_FRAMES:.1f}",
        flush=True,
    )


def t_obs_keys(track: Track):
    return track.obs


def cover_matrix(boxes: np.ndarray) -> np.ndarray:
    """Fraction of each box covered by a nearer box (larger bottom edge)."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, np.float32)
    ax1, ay1 = boxes[:, 0], boxes[:, 1]
    ax2, ay2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1.0)
    ix1 = np.maximum(ax1[:, None], ax1[None, :])
    iy1 = np.maximum(ay1[:, None], ay1[None, :])
    ix2 = np.minimum(ax2[:, None], ax2[None, :])
    iy2 = np.minimum(ay2[:, None], ay2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    bottoms = ay2
    # A box only occludes something whose bottom is above its own.
    nearer = bottoms[None, :] > (bottoms[:, None] + 4.0)
    np.fill_diagonal(nearer, False)
    covered = np.where(nearer, inter, 0.0).max(axis=1) / area
    return covered.astype(np.float32)


def _smooth_run(frames: list[int], obs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    if len(frames) < 4:
        return {f: obs[f] for f in frames}
    arr = np.stack([obs[f] for f in frames]).astype(np.float32)
    cx = median_filter(arr[:, 0] + arr[:, 2] * 0.5, size=3, mode="nearest")
    cy = median_filter(arr[:, 1] + arr[:, 3] * 0.5, size=3, mode="nearest")
    w = median_filter(arr[:, 2], size=5, mode="nearest")
    h = median_filter(arr[:, 3], size=5, mode="nearest")
    return {
        f: np.array([cx[i] - w[i] * 0.5, cy[i] - h[i] * 0.5, w[i], h[i]], dtype=np.float32)
        for i, f in enumerate(frames)
    }


def smooth_track(obs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Median-smooth each contiguous run. A gap must not blend two positions."""
    frames = sorted(obs)
    if len(frames) < 4:
        return obs
    out: dict[int, np.ndarray] = {}
    run = [frames[0]]
    for prev, frame in zip(frames, frames[1:]):
        if frame - prev > 1:
            out.update(_smooth_run(run, obs))
            run = [frame]
        else:
            run.append(frame)
    out.update(_smooth_run(run, obs))
    return out


def keep_track(t: Track) -> bool:
    n = len(t.obs)
    if n < 2 or t.hits < 2:
        return False
    if n >= 5:
        return True
    if t.max_conf >= 0.55:
        return True
    if t.truncated_ever and t.max_conf >= 0.38 and n >= 2:
        return True
    if t.max_conf >= 0.36 and n >= 3:
        return True
    return False


def split_implausible(tracks: list[Track], Hs: np.ndarray) -> list[Track]:
    """Split when camera-compensated motion is too large to be the same object.

    A wrong merge is worse than a split, so the gate is tight.
    """
    out = []
    for t in tracks:
        frames = sorted(t.obs)
        cuts = [frames[0]]
        for a, b in zip(frames, frames[1:]):
            box = t.obs[a]
            for f in range(a + 1, b + 1):
                box = warp_box(box, Hs[f - 1])
            gap = b - a
            size = max(float(t.obs[a][2]), float(t.obs[a][3]), 8.0)
            per = 14.0 + 0.35 * size
            # Short gaps follow normal motion. A long gap only stays one id
            # when the object reappears near its camera-compensated position.
            # Anything farther is a new id: a wrong merge is worse than a split.
            if gap <= 8:
                limit = per * gap
            else:
                limit = 1.05 * size + 2.0 * gap
            if dist(box, t.obs[b]) > limit or not np.isfinite(box).all():
                cuts.append(b)
        if len(cuts) == 1:
            out.append(t)
            continue
        bounds = cuts + [frames[-1] + 1]
        for c0, c1 in zip(bounds, bounds[1:]):
            obs = {f: t.obs[f] for f in frames if c0 <= f < c1}
            if len(obs) < 2:
                continue
            nt = Track(obs[min(obs)], t.subcls, t.max_conf, min(obs))
            nt.obs = obs
            nt.hits = len(obs)
            nt.confirmed = True
            nt.max_conf = t.max_conf
            nt.truncated_ever = any(touches_border(b) for b in obs.values())
            nt.start = min(obs)
            out.append(nt)
    return out


def validate(payload: dict) -> None:
    frames = payload["frames"]
    assert payload["clip"] == "F05_dense_4k"
    assert payload["image_size"] == [3840, 2160]
    assert payload["fps"] == 60
    assert payload["frame_count"] == 1199
    assert len(frames) == 1199
    sequences = [fr["sequence"] for fr in frames]
    assert sequences == list(range(1, 1200)), "sequences must be 1..1199 exactly once"
    seen_global = set()
    lifetimes: dict[int, list[int]] = {}
    for fr in frames:
        ids = []
        for d in fr["detections"]:
            x, y, w, h = d["box"]
            assert d["class_id"] in (0, 1)
            assert d["occlusion"] in (0, 1, 2)
            assert isinstance(d["truncated"], bool)
            assert w > 0 and h > 0
            assert x >= -0.1 and y >= -0.1
            assert x + w <= IMG_W + 1.5 and y + h <= IMG_H + 1.5
            ids.append(d["instance_id"])
            seen_global.add(d["instance_id"])
            lifetimes.setdefault(d["instance_id"], []).append(fr["sequence"])
        assert len(ids) == len(set(ids)), f"duplicate instance_id in frame {fr['sequence']}"
    if seen_global:
        assert min(seen_global) == 1
        assert seen_global == set(range(1, max(seen_global) + 1))
    print(f"valid labels  ids 1..{max(seen_global) if seen_global else 0}", flush=True)


def qa() -> None:
    with LABELS_PATH.open() as f:
        payload = json.load(f)
    by_seq = {fr["sequence"]: fr["detections"] for fr in payload["frames"]}
    QA_DIR.mkdir(exist_ok=True)
    samples = [1, 30, 120, 300, 600, 601, 900, 1199]
    for seq in samples:
        im = cv2.imread(str(frame_path(seq)))
        for d in by_seq[seq]:
            x, y, w, h = d["box"]
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            color = (0, 220, 0) if d["class_id"] == 0 else (0, 255, 255)
            if d["occlusion"]:
                color = (0, 140, 255)
            if d["truncated"]:
                color = (255, 80, 0)
            cv2.rectangle(im, p1, p2, color, 2)
            cv2.putText(
                im,
                str(d["instance_id"]),
                (p1[0], max(12, p1[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        out = QA_DIR / f"qa_{seq:05d}.jpg"
        cv2.imwrite(str(out), cv2.resize(im, (1440, 810)), [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        print("wrote", out, "n", len(by_seq[seq]), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("detect", "all"):
        detect()
    if cmd in ("track", "all"):
        track()
    if cmd in ("qa", "all"):
        qa()
    if cmd not in ("detect", "track", "qa", "all"):
        raise SystemExit("usage: annotate.py [detect|track|qa|all]")


if __name__ == "__main__":
    main()
