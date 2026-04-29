"""
Drone / UAV Video Analyzer — v3 Dual-Model Edition
=====================================================
Primary model:     YOLOv8n-VisDrone (aerial-trained, 10 classes)
                   pedestrian · people · bicycle · car · van · truck
                   tricycle · awning-tricycle · bus · motor
                   → ByteTrack tracking, trail overlay, loitering detection

Supplement model:  YOLOv8n-COCO (25 extra classes)
                   wildlife · aircraft · watercraft · suspicious objects
                   → predict-only (no tracking IDs), thin boxes

Hazard layer:      Per-frame HSV fire / smoke / vegetation / water analysis
Behavior:          Loitering, fast-mover, crowd alerts
HUD:               Live stats panel, scene type, active model badge
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from ultralytics import YOLO


# ── VisDrone class → semantic category ────────────────────────────────────────

VISDRONE_CATEGORY_MAP: dict[str, str] = {
    "pedestrian":      "People",
    "people":          "People",
    "bicycle":         "Vehicles",
    "car":             "Vehicles",
    "van":             "Vehicles",
    "truck":           "Vehicles",
    "tricycle":        "Vehicles",
    "awning-tricycle": "Vehicles",
    "bus":             "Vehicles",
    "motor":           "Vehicles",
}

# ── COCO supplement: classes not covered by VisDrone ──────────────────────────
# Used as full primary map when no VisDrone model is available.

COCO_CATEGORY_MAP: dict[str, str] = {
    # People & vehicles (fallback when VisDrone absent)
    "person":         "People",
    "car":            "Vehicles",
    "truck":          "Vehicles",
    "bus":            "Vehicles",
    "motorcycle":     "Vehicles",
    "bicycle":        "Vehicles",
    "train":          "Vehicles",
    # Aviation & maritime (VisDrone doesn't detect these)
    "airplane":       "Aircraft",
    "boat":           "Watercraft",
    # Wildlife — perimeter / agricultural monitoring
    "cat":            "Wildlife",
    "dog":            "Wildlife",
    "bird":           "Wildlife",
    "horse":          "Wildlife",
    "cow":            "Wildlife",
    "sheep":          "Wildlife",
    "elephant":       "Wildlife",
    "bear":           "Wildlife",
    "zebra":          "Wildlife",
    "giraffe":        "Wildlife",
    # Infrastructure
    "traffic light":  "Infrastructure",
    "stop sign":      "Infrastructure",
    "fire hydrant":   "Infrastructure",
    "parking meter":  "Infrastructure",
    # Abandoned / suspicious objects
    "backpack":       "Suspicious Object",
    "handbag":        "Suspicious Object",
    "suitcase":       "Suspicious Object",
    # Aerial objects
    "kite":           "Object",
    "frisbee":        "Object",
    "sports ball":    "Object",
}

# When VisDrone is active, only run COCO on these classes (no duplication)
_COCO_SUPPLEMENT = frozenset({
    "airplane", "boat",
    "cat", "dog", "bird", "horse", "cow", "sheep",
    "elephant", "bear", "zebra", "giraffe",
    "backpack", "handbag", "suitcase",
    "traffic light", "stop sign", "fire hydrant", "parking meter",
    "kite", "frisbee", "sports ball",
})

_COCO_ALL = frozenset(COCO_CATEGORY_MAP.keys())

# ── BGR colours per semantic category ────────────────────────────────────────

BOX_COLORS: dict[str, tuple] = {
    "People":             (60,  60, 240),   # Red
    "Vehicles":           (50, 220,  50),   # Green
    "Aircraft":           (240, 80,  50),   # Blue
    "Watercraft":         (220, 200,   0),  # Cyan
    "Wildlife":           (0,  165, 255),   # Orange
    "Infrastructure":     (0,  210, 210),   # Yellow-cyan
    "Suspicious Object":  (20,  20, 220),   # Dark red
    "Object":             (160, 160, 160),  # Grey
    "_default":           (180, 180,   0),
}

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_FRAMES     = 1500   # safety cap (~50 s @ 30 fps)
TRAIL_LEN      = 50    # frames of trail kept per track
LOITER_FRAMES  = 30    # consecutive processed-frames threshold for loitering
LOITER_RADIUS  = 45    # px — centroid must stay within this radius
FAST_SPEED_THR = 25.0  # px/frame


class DroneAnalyzer:
    def __init__(
        self,
        confidence: float = 0.20,
        iou: float = 0.45,
        frame_skip: int = 2,
        model=None,
        visdrone_model=None,
    ):
        self.model            = model if model is not None else YOLO("yolov8n.pt")
        self._visdrone_model  = visdrone_model
        self.confidence       = confidence
        self.iou              = iou
        self.frame_skip       = max(1, frame_skip)

    @property
    def visdrone_model(self):
        return self._visdrone_model

    @visdrone_model.setter
    def visdrone_model(self, m):
        self._visdrone_model = m

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _bytes_to_tmp(data: bytes, suffix: str = ".mp4") -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        return tmp.name

    @staticmethod
    def _track_speed(points: list) -> float:
        if len(points) < 2:
            return 0.0
        return float(np.mean([
            ((points[i][0] - points[i-1][0]) ** 2 +
             (points[i][1] - points[i-1][1]) ** 2) ** 0.5
            for i in range(1, len(points))
        ]))

    # ── Per-frame scene analysis (HSV) ────────────────────────────────────────

    @staticmethod
    def _analyze_scene(frame: np.ndarray) -> dict:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        total = float(h.size)

        fire_mask  = (((h <= 18) | (h >= 165)) & (s >= 140) & (v >= 100)).astype(np.uint8)
        smoke_mask = ((s <= 40) & (v >= 80) & (v <= 210)).astype(np.uint8)
        veg_mask   = ((h >= 35) & (h <= 85) & (s >= 40) & (v >= 40)).astype(np.uint8)
        water_mask = ((h >= 95) & (h <= 130) & (s >= 50) & (v >= 40)).astype(np.uint8)

        fire_pct  = np.sum(fire_mask)  / total * 100
        smoke_pct = np.sum(smoke_mask) / total * 100
        veg_pct   = np.sum(veg_mask)   / total * 100
        water_pct = np.sum(water_mask) / total * 100

        fire_det  = fire_pct > 2.0
        smoke_det = smoke_pct > 15.0 and fire_pct > 0.5

        if fire_det:          scene = "FIRE DETECTED"
        elif water_pct > 40:  scene = "Water / Flood Area"
        elif veg_pct > 55:    scene = "Forest / Vegetation"
        elif veg_pct > 25:    scene = "Mixed Terrain"
        else:                 scene = "Urban / Industrial"

        return dict(
            fire_pct=fire_pct, smoke_pct=smoke_pct,
            veg_pct=veg_pct,   water_pct=water_pct,
            fire_detected=fire_det, smoke_detected=smoke_det,
            scene_type=scene,
            fire_mask=fire_mask, veg_mask=veg_mask, water_mask=water_mask,
        )

    # ── Loitering check ───────────────────────────────────────────────────────

    @staticmethod
    def _is_loitering(points: list) -> bool:
        if len(points) < LOITER_FRAMES:
            return False
        recent = points[-LOITER_FRAMES:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        return (max(xs) - min(xs)) < LOITER_RADIUS and (max(ys) - min(ys)) < LOITER_RADIUS

    # ── Scene zone overlay ────────────────────────────────────────────────────

    @staticmethod
    def _draw_scene_overlay(img: np.ndarray, si: dict) -> np.ndarray:
        result = img.copy()
        for mask, tint, alpha in [
            (si["veg_mask"],   (20, 100, 20),  0.20),
            (si["water_mask"], (120, 50,  10), 0.20),
        ]:
            if np.any(mask):
                ov = np.zeros_like(result)
                ov[mask.astype(bool)] = tint
                cv2.addWeighted(result, 1.0, ov, alpha, 0, result)
        if si["fire_detected"] and np.any(si["fire_mask"]):
            ov = np.zeros_like(result)
            ov[si["fire_mask"].astype(bool)] = (10, 100, 255)
            cv2.addWeighted(result, 0.55, ov, 0.45, 0, result)
        return result

    # ── HUD stats panel ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(
        img: np.ndarray,
        si: dict,
        frame_counts: dict,
        loitering_ids: set,
        vd_active: bool,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        pw, ph = 272, min(215, h - 20)
        px, py = w - pw - 8, 8

        ov = img.copy()
        cv2.rectangle(ov, (px, py), (px + pw, py + ph), (8, 15, 28), -1)
        cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 212, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        def _t(text: str, row: int, color=(190, 220, 255), scale=0.45, bold=False):
            y = py + 18 + row * 22
            if y < py + ph - 4:
                cv2.putText(img, text, (px + 8, y), font, scale,
                            color, 2 if bold else 1, cv2.LINE_AA)

        badge = "VisDrone + COCO" if vd_active else "COCO"
        _t(f"GEOINTEL  [{badge}]", 0, (0, 212, 255), 0.45, True)
        _t(f"Scene: {si['scene_type'][:26]}", 1)
        _t(f"Veg {si['veg_pct']:.0f}%   Water {si['water_pct']:.0f}%", 2, (100, 220, 100))

        row = 3
        for cat, cnt in sorted(frame_counts.items(), key=lambda x: -x[1])[:4]:
            _t(f"{cat}: {cnt}", row, BOX_COLORS.get(cat, (190, 190, 190)))
            row += 1

        if si["fire_detected"]:
            _t(f"FIRE  {si['fire_pct']:.1f}%", row, (20, 80, 255), 0.47, True)
            row += 1
        if si["smoke_detected"]:
            _t("SMOKE DETECTED", row, (140, 140, 220), 0.44, True)
            row += 1
        if loitering_ids:
            _t(f"LOITERING  x{len(loitering_ids)}", row, (20, 20, 240), 0.44, True)

        return img

    # ── Primary (tracked) boxes — VisDrone or COCO-as-primary ─────────────────

    def _draw_tracked(
        self,
        img: np.ndarray,
        result,
        track_history: dict,
        loitering_ids: set,
        model_names: dict,
        cat_map: dict,
    ) -> np.ndarray:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return img
        for box in boxes:
            cls_name = model_names[int(box.cls[0])]
            if cls_name not in cat_map:
                continue
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cat   = cat_map[cls_name]
            color = BOX_COLORS.get(cat, BOX_COLORS["_default"])
            tid   = int(box.id[0]) if box.id is not None else None
            is_l  = tid in loitering_ids if tid is not None else False
            dcol  = (20, 20, 220) if is_l else color

            cv2.rectangle(img, (x1, y1), (x2, y2), dcol, 3 if is_l else 2)

            if is_l:
                cv2.putText(img, "! LOITERING", (x1, max(y1 - 22, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 255), 2, cv2.LINE_AA)

            tid_s  = f" #{tid}" if tid is not None else ""
            label  = f"{cls_name}{tid_s}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            lx, ly = x1, max(y1 - th - 8, 0)
            cv2.rectangle(img, (lx, ly), (lx + tw + 6, ly + th + 6), dcol, -1)
            cv2.putText(img, label, (lx + 3, ly + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

            # Trail
            if tid is not None and tid in track_history:
                pts = track_history[tid]["points"][-TRAIL_LEN:]
                for i in range(1, len(pts)):
                    a  = i / max(len(pts) - 1, 1)
                    tc = tuple(int(c * (0.25 + 0.75 * a)) for c in dcol)
                    cv2.line(img, pts[i - 1], pts[i], tc, 2, cv2.LINE_AA)
        return img

    # ── Supplement (COCO, no trails) ─────────────────────────────────────────

    def _draw_supplement(
        self,
        img: np.ndarray,
        result,
        coco_filter: frozenset,
    ) -> np.ndarray:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return img
        for box in boxes:
            cls_name = self.model.names[int(box.cls[0])]
            if cls_name not in coco_filter:
                continue
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cat   = COCO_CATEGORY_MAP.get(cls_name, cls_name.title())
            color = BOX_COLORS.get(cat, BOX_COLORS["_default"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label  = f"[{cls_name}]  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            lx, ly = x1, max(y1 - th - 6, 0)
            cv2.rectangle(img, (lx, ly), (lx + tw + 4, ly + th + 4), color, -1)
            cv2.putText(img, label, (lx + 2, ly + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    # ── Fire banner ───────────────────────────────────────────────────────────

    @staticmethod
    def _draw_fire_banner(img: np.ndarray, si: dict) -> np.ndarray:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 36), (w, h), (0, 0, 160), -1)
        cv2.putText(
            img,
            f"  FIRE DETECTED — {si['fire_pct']:.1f}% of frame"
            f"{'  |  SMOKE PRESENT' if si['smoke_detected'] else ''}",
            (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2, cv2.LINE_AA,
        )
        return img

    # ── Main analysis loop ────────────────────────────────────────────────────

    def analyze(self, video_input, progress_cb=None):
        def _cb(pct: float, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        if isinstance(video_input, bytes):
            video_path = self._bytes_to_tmp(video_input, ".mp4")
            _cleanup   = True
        else:
            video_path = video_input
            _cleanup   = False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video. Ensure it is a valid MP4/AVI/MOV.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration     = total_frames / fps
        frames_to    = min(total_frames, MAX_FRAMES)

        _base = os.path.join(tempfile.gettempdir(), f"geointel_drone_{int(time.time())}")
        _h264_codecs = [("avc1", f"{_base}.mp4"), ("H264", f"{_base}.mp4"),
                        ("X264", f"{_base}.mp4")]
        _fallback_codecs = [("mp4v", f"{_base}.mp4"), ("XVID", f"{_base}.avi")]
        writer, out_path, _need_reencode = None, f"{_base}.mp4", False
        for _fcc, _op in _h264_codecs:
            _w = cv2.VideoWriter(_op, cv2.VideoWriter_fourcc(*_fcc), fps, (W, H))
            if _w.isOpened():
                writer, out_path = _w, _op
                break
        if writer is None:
            for _fcc, _op in _fallback_codecs:
                _w = cv2.VideoWriter(_op, cv2.VideoWriter_fourcc(*_fcc), fps, (W, H))
                if _w.isOpened():
                    writer, out_path, _need_reencode = _w, _op, True
                    break

        vd_active    = self._visdrone_model is not None
        coco_filter  = _COCO_SUPPLEMENT if vd_active else _COCO_ALL
        primary_mdl  = self._visdrone_model if vd_active else self.model
        primary_cmap = VISDRONE_CATEGORY_MAP if vd_active else COCO_CATEGORY_MAP
        # VisDrone needs lower conf — aerial objects are small and harder to detect
        primary_conf = max(0.10, self.confidence - 0.10) if vd_active \
                       else max(0.15, self.confidence - 0.05)

        detected_objects: dict       = {}
        frame_timeline:   list       = []
        track_history:    dict       = {}
        sample_frames:    list       = []
        last_ann:         np.ndarray | None = None
        loitering_ids:    set        = set()
        alerts:           list       = []
        scene_counts:     dict       = {}
        fire_frames = smoke_frames   = 0
        frame_idx   = 0

        _cb(0.05, f"Starting dual-model drone analysis  "
                  f"[{'VisDrone + COCO' if vd_active else 'COCO'}]…")

        while cap.isOpened() and frame_idx < frames_to:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                frame_stats:  dict = {"time": round(frame_idx / fps, 2)}
                frame_counts: dict = {}

                # ── Scene analysis ─────────────────────────────────────────
                si = self._analyze_scene(frame)
                if si["fire_detected"]:  fire_frames  += 1
                if si["smoke_detected"]: smoke_frames += 1
                scene_counts[si["scene_type"]] = scene_counts.get(si["scene_type"], 0) + 1

                # ── Primary model with ByteTrack ───────────────────────────
                try:
                    pres = primary_mdl.track(
                        frame, conf=primary_conf, iou=self.iou,
                        persist=True, verbose=False, tracker="bytetrack.yaml",
                    )
                except Exception:
                    pres = primary_mdl.track(
                        frame, conf=primary_conf, iou=self.iou,
                        persist=True, verbose=False,
                    )
                primary_result = pres[0]

                # Accumulate primary stats + track history
                if primary_result.boxes is not None:
                    for box in primary_result.boxes:
                        cls_name = primary_mdl.names[int(box.cls[0])]
                        if cls_name not in primary_cmap:
                            continue
                        cat  = primary_cmap[cls_name]
                        conf = float(box.conf[0])
                        if cat not in detected_objects:
                            detected_objects[cat] = {"count": 0, "confidence": []}
                        detected_objects[cat]["count"]      += 1
                        detected_objects[cat]["confidence"].append(conf)
                        frame_stats[cat]  = frame_stats.get(cat, 0)  + 1
                        frame_counts[cat] = frame_counts.get(cat, 0) + 1

                        if box.id is not None:
                            tid = int(box.id[0])
                            cx  = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                            cy  = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                            if tid not in track_history:
                                track_history[tid] = {
                                    "points": [], "class": cls_name, "category": cat,
                                }
                            track_history[tid]["points"].append((cx, cy))

                            if self._is_loitering(track_history[tid]["points"]):
                                if tid not in loitering_ids:
                                    loitering_ids.add(tid)
                                    if cat == "People":
                                        alerts.append(f"Loitering person (Track #{tid})")
                                    elif cat == "Suspicious Object":
                                        alerts.append(
                                            f"Stationary suspicious object (Track #{tid})"
                                        )

                # ── COCO supplement (when VisDrone active) ─────────────────
                supp_result = None
                if vd_active:
                    cres        = self.model.predict(
                        frame, conf=max(0.15, self.confidence - 0.05),
                        iou=self.iou, verbose=False,
                    )
                    supp_result = cres[0]
                    if supp_result.boxes is not None:
                        for box in supp_result.boxes:
                            cls_name = self.model.names[int(box.cls[0])]
                            if cls_name not in coco_filter:
                                continue
                            cat  = COCO_CATEGORY_MAP[cls_name]
                            conf = float(box.conf[0])
                            if cat not in detected_objects:
                                detected_objects[cat] = {"count": 0, "confidence": []}
                            detected_objects[cat]["count"]      += 1
                            detected_objects[cat]["confidence"].append(conf)
                            frame_stats[cat]  = frame_stats.get(cat, 0)  + 1
                            frame_counts[cat] = frame_counts.get(cat, 0) + 1

                frame_timeline.append(frame_stats)

                # ── Compose annotated frame ─────────────────────────────────
                ann = self._draw_scene_overlay(frame.copy(), si)
                if supp_result is not None:
                    ann = self._draw_supplement(ann, supp_result, coco_filter)
                ann = self._draw_tracked(ann, primary_result, track_history,
                                         loitering_ids, primary_mdl.names, primary_cmap)
                ann = self._draw_hud(ann, si, frame_counts, loitering_ids, vd_active)
                if si["fire_detected"]:
                    ann = self._draw_fire_banner(ann, si)

                last_ann = ann

                step = max(frames_to // 6, self.frame_skip)
                if len(sample_frames) < 6 and frame_idx % step < self.frame_skip:
                    sample_frames.append(
                        Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
                    )
                writer.write(ann)

            else:
                writer.write(last_ann if last_ann is not None else frame)

            frame_idx += 1
            if frame_idx % 60 == 0:
                _cb(
                    min(0.10 + 0.85 * (frame_idx / frames_to), 0.95),
                    f"Processing frame {frame_idx}/{frames_to}…",
                )

        cap.release()
        writer.release()

        # Re-encode to H.264 so the video is playable inline in browsers
        if _need_reencode:
            try:
                import subprocess
                try:
                    from imageio_ffmpeg import get_ffmpeg_exe as _get_ffmpeg
                    _ffmpeg_bin = _get_ffmpeg()
                except Exception:
                    _ffmpeg_bin = "ffmpeg"
                _h264 = out_path.rsplit(".", 1)[0] + "_h264.mp4"
                _r = subprocess.run(
                    [_ffmpeg_bin, "-y", "-i", out_path,
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-movflags", "+faststart", _h264],
                    capture_output=True, timeout=300,
                )
                if _r.returncode == 0:
                    try:
                        os.unlink(out_path)
                    except OSError:
                        pass
                    out_path = _h264
            except Exception:
                pass  # fallback: keep mp4v/avi file; download still works

        if _cleanup:
            try:
                os.unlink(video_path)
            except OSError:
                pass

        # Post-process stats
        for data in detected_objects.values():
            confs = data["confidence"]
            data["avg_confidence"] = round(sum(confs) / len(confs), 3)
            data["count"]          = max(1, data["count"] // self.frame_skip)

        peak_counts: dict = {}
        for entry in frame_timeline:
            for k, v in entry.items():
                if k != "time":
                    peak_counts[k] = max(peak_counts.get(k, 0), v)

        speeds      = [self._track_speed(td["points"]) for td in track_history.values()]
        fast_movers = sum(1 for s in speeds if s > FAST_SPEED_THR)
        avg_speed   = round(float(np.mean(speeds)), 2) if speeds else 0.0
        processed   = max(frame_idx // self.frame_skip, 1)
        dom_scene   = max(scene_counts, key=scene_counts.get) if scene_counts else "Unknown"
        loit_ppl    = sum(
            1 for tid in loitering_ids
            if track_history.get(tid, {}).get("category") == "People"
        )

        _cb(1.0, "Drone analysis complete.")

        return {
            "output_video_path": out_path,
            "detected_objects":  detected_objects,
            "peak_counts":       peak_counts,
            "frame_timeline":    frame_timeline,
            "sample_frames":     sample_frames,
            "track_history":     track_history,
            "total_tracks":      len(track_history),
            "fast_movers":       fast_movers,
            "avg_track_speed":   avg_speed,
            "total_frames":      total_frames,
            "processed_frames":  frame_idx,
            "fps":               fps,
            "video_duration":    duration,
            "video_size":        (W, H),
            "visdrone_active":   vd_active,
            "fire_frames":       fire_frames,
            "smoke_frames":      smoke_frames,
            "fire_pct_frames":   round(100 * fire_frames  / processed, 1),
            "smoke_pct_frames":  round(100 * smoke_frames / processed, 1),
            "dominant_scene":    dom_scene,
            "loitering_count":   len(loitering_ids),
            "loitering_people":  loit_ppl,
            "alerts":            list(dict.fromkeys(alerts))[:12],
        }
