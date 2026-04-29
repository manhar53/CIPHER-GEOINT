"""
Satellite Image Analyzer — Full Edition
=========================================
Analysis stack (all layers applied in sequence):

  1. SegFormer (ADE20K-150) — semantic segmentation
       buildings, roads, vehicles, aircraft, watercraft,
       vegetation, agriculture, water, airport, infrastructure
  2. YOLOv8-OBB (DOTA)    — oriented bounding boxes for aerial objects
       planes, ships, large/small vehicles, helicopters, bridges,
       harbors, storage tanks  (proper OBB polygons, not axis-aligned)
  3. YOLOv8n (COCO)        — tiled detection for cars, people, trucks
  4. HSV-based detectors   — fire, smoke, burn scars, solar panels,
                             cloud masking, vegetation health (ExG)

Annotation types produced on the output image:
  • Polygon outlines      — building footprints (cyan)
  • Road highlight        — road network (yellow semi-transparent)
  • Oriented boxes (OBB)  — DOTA aerial objects (rotated polygons)
  • Axis-aligned boxes    — YOLO COCO objects
  • Colour overlay        — land / terrain classification
  • Tinted overlays       — fire (orange), smoke (grey), burn scars (dark)
  • Colour-coded outlines — solar panel regions (blue)
  • Legend strip          — top-right corner key
"""

import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

try:
    import torch
    import torch.nn.functional as _F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── ADE20K class IDs (0-indexed) → geospatial category ───────────────────────

ADE20K_TO_GEO = {
    # Structures
    1:   "Buildings",      # building / edifice
    25:  "Buildings",      # house
    48:  "Buildings",      # skyscraper
    79:  "Buildings",      # hovel / hut
    84:  "Buildings",      # tower
    # Roads & paved
    6:   "Roads",          # road, route
    11:  "Roads",          # sidewalk / pavement
    52:  "Roads",          # path
    91:  "Roads",          # dirt track
    # Airport
    54:  "Airport",        # runway
    # Infrastructure
    61:  "Infrastructure", # bridge
    140: "Infrastructure", # pier / wharf / dock
    # Vehicles (SegFormer can segment these in aerial views)
    20:  "Vehicles",       # car
    80:  "Vehicles",       # bus
    83:  "Vehicles",       # truck
    102: "Vehicles",       # van
    # Aircraft / watercraft
    90:  "Aircraft",       # airplane
    76:  "Watercraft",     # boat
    103: "Watercraft",     # ship
    # Vegetation
    4:   "Vegetation",     # tree
    9:   "Vegetation",     # grass
    17:  "Vegetation",     # plant / flora
    66:  "Vegetation",     # flower
    72:  "Vegetation",     # palm
    # Agriculture
    29:  "Agriculture",    # field
    # Water
    21:  "Water",          # water
    26:  "Water",          # sea
    60:  "Water",          # river
    128: "Water",          # lake
    # Recreation
    109: "Recreation",     # swimming pool
    # Bare ground
    13:  "Bare Ground",    # earth / ground
    46:  "Bare Ground",    # sand
    94:  "Bare Ground",    # land / soil
}

GEO_COLORS = {
    "Buildings":     [220, 100,  40],
    "Roads":         [ 90,  90,  90],
    "Airport":       [210, 210, 100],
    "Infrastructure":[255, 140,   0],
    "Vehicles":      [  0, 230,  80],
    "Aircraft":      [ 80,  80, 255],
    "Watercraft":    [  0, 200, 255],
    "Vegetation":    [ 34, 170,  34],
    "Agriculture":   [110, 190,  60],
    "Water":         [ 30, 100, 220],
    "Recreation":    [  0, 200, 220],
    "Bare Ground":   [185, 130,  60],
}

# ── COCO whitelist for YOLO ───────────────────────────────────────────────────

CATEGORY_MAP = {
    "person":        "People",
    "car":           "Vehicles",
    "truck":         "Vehicles",
    "bus":           "Vehicles",
    "motorcycle":    "Vehicles",
    "bicycle":       "Vehicles",
    "airplane":      "Aircraft",
    "boat":          "Watercraft",
    "train":         "Vehicles",
    "traffic light": "Infrastructure",
    "stop sign":     "Infrastructure",
    "parking meter": "Infrastructure",
}
_KEEP_CLASSES = set(CATEGORY_MAP.keys())

# ── DOTA OBB class map ────────────────────────────────────────────────────────

OBB_CATEGORY_MAP = {
    0:  ("Aircraft",      "plane"),
    1:  ("Watercraft",    "ship"),
    2:  ("Industrial",    "storage-tank"),
    7:  ("Infrastructure","harbor"),
    8:  ("Infrastructure","bridge"),
    9:  ("Vehicles",      "large-vehicle"),
    10: ("Vehicles",      "small-vehicle"),
    11: ("Aircraft",      "helicopter"),
}

# ── Visual palette ────────────────────────────────────────────────────────────

LAND_COLORS = {
    "Vegetation":  [ 34, 170,  34],
    "Water":       [ 30, 100, 220],
    "Urban":       [160, 160, 160],
    "Roads":       [ 50,  50,  50],
    "Bare Ground": [180, 120,  50],
    "Snow/Clouds": [230, 230, 250],
}

BOX_COLORS = {
    "Vehicles":       (  0, 230,  80),
    "People":         (255,  60,  60),
    "Aircraft":       ( 80,  80, 255),
    "Watercraft":     (  0, 200, 255),
    "Infrastructure": (255, 200,   0),
    "Industrial":     (255, 120,   0),
    "_default":       (200, 200,   0),
}

OBB_COLORS = {
    "Aircraft":      (100, 100, 255),
    "Watercraft":    (  0, 220, 255),
    "Vehicles":      ( 30, 255, 100),
    "Infrastructure":(255, 180,   0),
    "Industrial":    (255, 100,   0),
    "_default":      (200, 200,   0),
}


class SatelliteAnalyzer:
    def __init__(self, confidence=0.25, iou=0.45,
                 model=None, seg_model=None, seg_processor=None,
                 obb_model=None):
        self.model         = model if model is not None else YOLO("yolov8n.pt")
        self.confidence    = confidence
        self.iou           = iou
        self.seg_model     = seg_model
        self.seg_processor = seg_processor
        self.obb_model     = obb_model   # YOLOv8-OBB (DOTA)

    # ══════════════════════════════════════════════════════════════════════
    #  1.  SegFormer Semantic Segmentation
    # ══════════════════════════════════════════════════════════════════════

    def _segment_scene(self, pil_img: Image.Image):
        if not _HAS_TORCH or self.seg_model is None or self.seg_processor is None:
            return None
        orig_w, orig_h = pil_img.size
        max_dim = 1024
        if max(orig_w, orig_h) > max_dim:
            scale    = max_dim / max(orig_w, orig_h)
            work_w   = int(orig_w * scale)
            work_h   = int(orig_h * scale)
            work_img = pil_img.resize((work_w, work_h), Image.LANCZOS)
        else:
            work_img = pil_img
            work_w, work_h = orig_w, orig_h

        seg_size = 512
        stride   = 384
        pred     = np.zeros((work_h, work_w), dtype=np.int16)

        rows = list(range(0, max(1, work_h - stride), stride)) if work_h > seg_size else [0]
        cols = list(range(0, max(1, work_w - stride), stride)) if work_w > seg_size else [0]
        if rows[-1] + seg_size < work_h:
            rows.append(max(0, work_h - seg_size))
        if cols[-1] + seg_size < work_w:
            cols.append(max(0, work_w - seg_size))

        for row in rows:
            for col in cols:
                r2   = min(row + seg_size, work_h)
                c2   = min(col + seg_size, work_w)
                ch   = r2 - row
                cw   = c2 - col
                crop = work_img.crop((col, row, c2, r2))
                if cw < seg_size or ch < seg_size:
                    padded = Image.new("RGB", (seg_size, seg_size), (114, 114, 114))
                    padded.paste(crop, (0, 0))
                    crop = padded
                inputs = self.seg_processor(images=crop, return_tensors="pt")
                with torch.no_grad():
                    logits = self.seg_model(**inputs).logits
                tile_pred = _F.interpolate(
                    logits, size=(seg_size, seg_size),
                    mode="bilinear", align_corners=False,
                ).argmax(dim=1).squeeze().cpu().numpy().astype(np.int16)
                pred[row:r2, col:c2] = tile_pred[:ch, :cw]

        if (work_w, work_h) != (orig_w, orig_h):
            pred = cv2.resize(
                pred.astype(np.int32), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int16)
        return pred

    def _seg_to_overlay(self, seg_pred: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        h, w    = seg_pred.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for ade_id, geo_cat in ADE20K_TO_GEO.items():
            color = GEO_COLORS.get(geo_cat, [128, 128, 128])
            overlay[seg_pred == ade_id] = color
        return cv2.addWeighted(rgb, 0.50, overlay, 0.50, 0)

    def _count_buildings(self, seg_pred: np.ndarray) -> int:
        ids    = [k for k, v in ADE20K_TO_GEO.items() if v == "Buildings"]
        mask   = np.isin(seg_pred, ids).astype(np.uint8)
        if mask.sum() == 0:
            return 0
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        min_px = max(80, int(seg_pred.size * 0.0003))
        n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px)

    def _estimate_road_coverage(self, seg_pred: np.ndarray) -> float:
        ids = [k for k, v in ADE20K_TO_GEO.items() if v == "Roads"]
        return round(float(np.isin(seg_pred, ids).sum()) / seg_pred.size * 100, 1)

    def _seg_land_pct(self, seg_pred: np.ndarray) -> dict:
        total   = seg_pred.size
        cat_ids: dict = {}
        for ade_id, geo_cat in ADE20K_TO_GEO.items():
            cat_ids.setdefault(geo_cat, []).append(ade_id)
        pct = {}
        for geo_cat, ids in cat_ids.items():
            pct[geo_cat] = round(float(np.isin(seg_pred, ids).sum()) / total * 100, 1)
        covered   = sum(pct.values())
        pct["Other"] = max(0.0, round(100.0 - covered, 1))
        return pct

    # ══════════════════════════════════════════════════════════════════════
    #  2.  Building Polygon Extraction
    # ══════════════════════════════════════════════════════════════════════

    def _extract_building_polygons(self, seg_pred: np.ndarray) -> list:
        ids    = [k for k, v in ADE20K_TO_GEO.items() if v == "Buildings"]
        mask   = np.isin(seg_pred, ids).astype(np.uint8)
        if mask.sum() == 0:
            return []
        closed    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area  = max(120, int(seg_pred.size * 0.00015))
        polygons  = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                eps    = 0.025 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                polygons.append(approx)
        return polygons

    # ══════════════════════════════════════════════════════════════════════
    #  3.  SegFormer → Axis-Aligned Boxes (vehicles / aircraft / watercraft)
    # ══════════════════════════════════════════════════════════════════════

    def _extract_seg_boxes(self, seg_pred: np.ndarray) -> list:
        box_cats = {
            "Vehicles":   [20, 80, 83, 102],
            "Aircraft":   [90],
            "Watercraft": [76, 103],
        }
        dets = []
        for cat, ids in box_cats.items():
            mask = np.isin(seg_pred, ids).astype(np.uint8)
            if mask.sum() == 0:
                continue
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 200:
                    continue
                x1 = stats[i, cv2.CC_STAT_LEFT]
                y1 = stats[i, cv2.CC_STAT_TOP]
                w  = stats[i, cv2.CC_STAT_WIDTH]
                h  = stats[i, cv2.CC_STAT_HEIGHT]
                dets.append({"category": cat, "conf": 0.70,
                             "xyxy": [x1, y1, x1+w, y1+h], "source": "segformer"})
        return dets

    # ══════════════════════════════════════════════════════════════════════
    #  4.  Water Body Analysis
    # ══════════════════════════════════════════════════════════════════════

    def _analyze_water_bodies(self, seg_pred: np.ndarray) -> dict:
        ids  = [k for k, v in ADE20K_TO_GEO.items() if v in ("Water", "Recreation")]
        mask = np.isin(seg_pred, ids).astype(np.uint8)
        pct  = round(float(mask.sum()) / seg_pred.size * 100, 1)
        if mask.sum() == 0:
            return {"count": 0, "coverage_pct": 0.0}
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        min_px = max(200, int(seg_pred.size * 0.001))
        n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        count = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px)
        return {"count": count, "coverage_pct": pct}

    # ══════════════════════════════════════════════════════════════════════
    #  5.  YOLOv8-OBB  (DOTA aerial objects — Oriented Bounding Boxes)
    # ══════════════════════════════════════════════════════════════════════

    def _detect_obb(self, rgb: np.ndarray) -> list:
        """
        Run YOLOv8-OBB on the full image for aerial-specific DOTA classes.
        Returns list of dicts with 4-corner polygon points for each detection.
        """
        if self.obb_model is None:
            return []

        conf = max(0.18, self.confidence - 0.07)
        results = self.obb_model(rgb, conf=conf, iou=self.iou, verbose=False)

        obb_dets = []
        if not results or results[0].obb is None:
            return []

        for box in results[0].obb:
            cls_id = int(box.cls[0])
            if cls_id not in OBB_CATEGORY_MAP:
                continue
            cat, name = OBB_CATEGORY_MAP[cls_id]
            # xyxyxyxy shape: (1, 4, 2) — 4 corner points
            pts_raw = box.xyxyxyxy[0].cpu().numpy()     # (4, 2)
            pts = pts_raw.astype(int).tolist()           # [[x,y], ...]
            obb_dets.append({
                "category": cat,
                "cls_name": name,
                "conf":     float(box.conf[0]),
                "points":   pts,                          # 4 × [x, y]
            })
        return obb_dets

    # ══════════════════════════════════════════════════════════════════════
    #  6.  Fire / Smoke Detection
    # ══════════════════════════════════════════════════════════════════════

    def _detect_fire_smoke(self, rgb: np.ndarray) -> dict:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total   = rgb.shape[0] * rgb.shape[1]

        fire_mask  = (((h <= 18) | (h >= 165)) & (s >= 140) & (v >= 100)).astype(np.uint8)
        smoke_mask = ((s <= 45) & (v >= 90) & (v <= 210)).astype(np.uint8)

        fire_pct  = round(float(fire_mask.sum())  / total * 100, 2)
        smoke_pct = round(float(smoke_mask.sum()) / total * 100, 2)

        return {
            "fire_detected":  fire_pct  > 0.4,
            "fire_coverage":  fire_pct,
            "smoke_detected": smoke_pct > 1.5,
            "smoke_coverage": smoke_pct,
            "fire_mask":      fire_mask,
            "smoke_mask":     smoke_mask,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  7.  Burn Scar Detection
    # ══════════════════════════════════════════════════════════════════════

    def _detect_burn_scars(self, rgb: np.ndarray) -> dict:
        """
        Burn scars appear as very dark, low-saturation, brownish-black
        regions — distinct from shadows (which tend to be bluer).
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total   = rgb.shape[0] * rgb.shape[1]

        burn = (
            (v <= 65) & (s <= 90) &
            # exclude bright sky/cloud pixels
            ~((s <= 25) & (v >= 160))
        ).astype(np.uint8)

        opened = cv2.morphologyEx(burn, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        pct    = round(float(opened.sum()) / total * 100, 2)
        n, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        min_area = max(400, int(total * 0.002))
        count  = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)

        return {
            "detected":     pct > 0.8,
            "count":        count,
            "coverage_pct": pct,
            "mask":         opened,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  8.  Solar Panel Detection
    # ══════════════════════════════════════════════════════════════════════

    def _detect_solar_panels(self, rgb: np.ndarray) -> dict:
        """
        Solar panels appear as dark blue-grey rectangular arrays.
        Detected via HSV hue in the blue range with controlled saturation.
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total   = rgb.shape[0] * rgb.shape[1]

        solar = (
            (h >= 85) & (h <= 135) &
            (s >= 18) & (s <= 130) &
            (v >= 15) & (v <= 110)
        ).astype(np.uint8)

        closed = cv2.morphologyEx(solar, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        min_area = max(150, int(total * 0.0004))

        regions = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x1 = stats[i, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            w  = stats[i, cv2.CC_STAT_WIDTH]
            ht = stats[i, cv2.CC_STAT_HEIGHT]
            # Exclude very elongated shapes (likely false positives)
            if max(w, ht) / max(min(w, ht), 1) < 12:
                regions.append((x1, y1, x1+w, y1+ht))

        pct = round(float(closed.sum()) / total * 100, 2)
        return {
            "count":        len(regions),
            "coverage_pct": pct,
            "regions":      regions,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  9.  Vegetation Health (ExG visible-light proxy)
    # ══════════════════════════════════════════════════════════════════════

    def _estimate_vegetation_health(self, rgb: np.ndarray) -> dict:
        """
        Excess Green Index (ExG = 2g − r − b, normalised channels).
        Healthy dense vegetation → high ExG; stressed / sparse → lower ExG.
        """
        r = rgb[:,:,0].astype(float)
        g = rgb[:,:,1].astype(float)
        b = rgb[:,:,2].astype(float)
        denom = r + g + b + 1e-6
        exg   = 2*(g/denom) - (r/denom) - (b/denom)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        veg_mask = ((hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 90) &
                    (hsv[:,:,1] >= 35) & (hsv[:,:,2] >= 30))

        if veg_mask.sum() < 100:
            return {"health_index": 0.0, "status": "No vegetation",
                    "healthy_pct": 0.0, "stressed_pct": 0.0}

        veg_exg   = exg[veg_mask]
        mean_exg  = float(np.mean(veg_exg))
        healthy_p = round(float((veg_exg > 0.05).sum()) / veg_mask.sum() * 100, 1)

        if healthy_p > 70:
            status = "Healthy"
        elif healthy_p > 40:
            status = "Moderate"
        else:
            status = "Stressed / Sparse"

        return {
            "health_index": round(mean_exg, 3),
            "status":       status,
            "healthy_pct":  healthy_p,
            "stressed_pct": round(100 - healthy_p, 1),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  10. Cloud / Shadow Masking
    # ══════════════════════════════════════════════════════════════════════

    def _mask_clouds(self, rgb: np.ndarray) -> dict:
        """
        Identify cloud-covered and shadow areas so they can be
        excluded from analysis and clearly labelled on the map.
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s, v = hsv[:,:,1], hsv[:,:,2]
        total = rgb.shape[0] * rgb.shape[1]

        cloud_mask  = ((s <= 30) & (v >= 200)).astype(np.uint8)
        shadow_mask = ((s <= 50) & (v <= 55)).astype(np.uint8)

        cloud_pct  = round(float(cloud_mask.sum())  / total * 100, 1)
        shadow_pct = round(float(shadow_mask.sum()) / total * 100, 1)

        return {
            "cloud_pct":   cloud_pct,
            "shadow_pct":  shadow_pct,
            "cloud_mask":  cloud_mask,
            "shadow_mask": shadow_mask,
            "usable_pct":  round(100 - cloud_pct - shadow_pct, 1),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  11. Parking Lot Detection
    # ══════════════════════════════════════════════════════════════════════

    def _detect_parking_lots(self, rgb: np.ndarray) -> int:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s, v = hsv[:,:,1], hsv[:,:,2]
        total  = rgb.shape[0] * rgb.shape[1]
        paved  = ((s <= 40) & (v >= 55) & (v <= 185)).astype(np.uint8)
        eroded = cv2.erode(paved, np.ones((12, 12), np.uint8))
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        return sum(1 for i in range(1, n)
                   if stats[i, cv2.CC_STAT_AREA] >= int(total * 0.004))

    # ══════════════════════════════════════════════════════════════════════
    #  12. HSV Terrain Classification (fallback when SegFormer absent)
    # ══════════════════════════════════════════════════════════════════════

    def _classify_land(self, rgb: np.ndarray):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total = rgb.shape[0] * rgb.shape[1]

        masks = {
            "Vegetation":  (h_ >= 35) & (h_ <= 90)  & (s_ >= 35) & (v_ >= 30),
            "Water":       (h_ >= 90) & (h_ <= 140) & (s_ >= 40) & (v_ >= 20),
            "Snow/Clouds": (s_ <= 25) & (v_ >= 215),
            "Roads":       (s_ <= 40) & (v_ >= 45)  & (v_ <= 145),
            "Bare Ground": (
                ((h_ >= 10) & (h_ <= 35) & (s_ >= 25) & (v_ >= 40)) |
                ((h_ <=  10) & (s_ >= 25) & (v_ >= 40))
            ),
        }
        occupied = np.zeros(rgb.shape[:2], bool)
        for m in masks.values():
            occupied |= m
        masks["Urban"] = ~occupied & (v_ >= 80)

        pct     = {k: round(float(m.sum()) / total * 100, 1) for k, m in masks.items()}
        overlay = np.zeros_like(rgb)
        for name, mask in masks.items():
            overlay[mask] = LAND_COLORS.get(name, [128, 128, 128])
        blended  = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)
        dominant = max(pct, key=pct.get)
        return pct, blended, dominant, masks

    def _extract_features_cv(self, rgb: np.ndarray, masks: dict) -> dict:
        h_px, w_px = rgb.shape[:2]
        total = h_px * w_px
        road_mask  = masks.get("Roads",   np.zeros((h_px, w_px), bool))
        road_pct   = round(float(road_mask.sum()) / total * 100, 1)
        urban_mask = masks.get("Urban",   np.zeros((h_px, w_px), bool))
        urban_cl   = cv2.morphologyEx(
            urban_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(urban_cl, 8)
        building_est = sum(1 for i in range(1, n)
                           if stats[i, cv2.CC_STAT_AREA] >= max(50, int(total*0.0004)))
        veg_mask   = masks.get("Vegetation", np.zeros((h_px, w_px), bool))
        veg_cl     = cv2.morphologyEx(
            veg_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((20,20),np.uint8))
        n_veg, _, vstats, _ = cv2.connectedComponentsWithStats(veg_cl, 8)
        green_zones = sum(1 for i in range(1, n_veg)
                          if vstats[i, cv2.CC_STAT_AREA] >= int(total*0.004))
        bare_mask   = masks.get("Bare Ground", np.zeros((h_px, w_px), bool))
        n_bare, _, bstats, _ = cv2.connectedComponentsWithStats(
            bare_mask.astype(np.uint8), 8)
        open_areas  = sum(1 for i in range(1, n_bare)
                          if bstats[i, cv2.CC_STAT_AREA] >= int(total*0.003))
        return {"estimated_structures": building_est, "road_coverage_pct": road_pct,
                "green_zones": green_zones, "open_areas": open_areas}

    # ══════════════════════════════════════════════════════════════════════
    #  13. Tiled YOLO Detection (COCO)
    # ══════════════════════════════════════════════════════════════════════

    def _detect_tiled(self, rgb: np.ndarray, progress_cb=None) -> list:
        h, w    = rgb.shape[:2]
        tile    = 640
        overlap = 120
        stride  = tile - overlap
        raw: list = []
        cols = list(range(0, w, stride))
        rows = list(range(0, h, stride))
        total_tiles = len(cols) * len(rows)
        idx  = 0
        conf = max(0.15, self.confidence - 0.05)

        for row in rows:
            for col in cols:
                r2   = min(row + tile, h)
                c2   = min(col + tile, w)
                crop = rgb[row:r2, col:c2]
                if crop.shape[0] < tile or crop.shape[1] < tile:
                    pad = np.zeros((tile, tile, 3), np.uint8)
                    pad[:crop.shape[0], :crop.shape[1]] = crop
                    crop = pad
                results = self.model(crop, conf=conf, iou=self.iou, verbose=False)
                for box in results[0].boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name not in _KEEP_CLASSES:
                        continue
                    bx = box.xyxy[0].tolist()
                    tile_w, tile_h = c2 - col, r2 - row
                    if bx[0] > tile_w or bx[1] > tile_h:
                        continue
                    raw.append({
                        "cls_id":   cls_id,
                        "cls_name": cls_name,
                        "conf":     float(box.conf[0]),
                        "xyxy":     [bx[0]+col, bx[1]+row, bx[2]+col, bx[3]+row],
                    })
                idx += 1
                if progress_cb:
                    pct = 0.63 + 0.25 * (idx / max(total_tiles, 1))
                    progress_cb(pct, f"Scanning tile {idx}/{total_tiles}…")
        return self._nms(raw)

    def _iou(self, a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        inter = max(0,ix2-ix1) * max(0,iy2-iy1)
        if inter == 0:
            return 0.0
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(u, 1e-6)

    def _nms(self, dets: list) -> list:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
        kept: list = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets
                    if d["cls_id"] != best["cls_id"]
                    or self._iou(best["xyxy"], d["xyxy"]) < self.iou]
        return kept

    # ══════════════════════════════════════════════════════════════════════
    #  14. Full Layered Annotation
    # ══════════════════════════════════════════════════════════════════════

    def _annotate(
        self,
        base:             np.ndarray,
        dets:             list,
        seg_overlay=None,
        building_polygons=None,
        seg_pred=None,
        seg_dets=None,
        obb_dets=None,
        fire_smoke=None,
        burn_scars=None,
        solar_panels=None,
        cloud_info=None,
    ) -> np.ndarray:
        """
        Layer order (bottom → top):
          1. Segmentation colour overlay
          2. Road/airport network yellow highlight
          3. Cloud + shadow grey tint
          4. Burn scar dark overlay
          5. Fire tint / smoke tint
          6. Building polygon outlines (cyan)
          7. Solar panel region outlines (blue)
          8. SegFormer-derived axis-aligned boxes
          9. YOLO-OBB oriented polygons          ← new
         10. YOLO COCO axis-aligned boxes
         11. Legend
        """
        img = seg_overlay.copy() if seg_overlay is not None else base.copy()

        # ── Road / airport highlight (yellow) ─────────────────────────
        if seg_pred is not None:
            rids = [k for k, v in ADE20K_TO_GEO.items() if v in ("Roads", "Airport")]
            road_mask = np.isin(seg_pred, rids).astype(np.uint8)
            if road_mask.sum() > 0:
                dil = cv2.dilate(road_mask, np.ones((3, 3), np.uint8))
                rl  = np.zeros_like(img); rl[dil > 0] = [255, 240, 0]
                img = cv2.addWeighted(img, 1.0, rl, 0.28, 0)

        # ── Cloud tint (white) ─────────────────────────────────────────
        if cloud_info and cloud_info.get("cloud_pct", 0) > 2:
            cm = cloud_info["cloud_mask"]
            cl = np.zeros_like(img); cl[cm > 0] = [240, 240, 255]
            img = cv2.addWeighted(img, 1.0, cl, 0.25, 0)

        # ── Burn scar overlay (dark red) ──────────────────────────────
        if burn_scars and burn_scars.get("detected"):
            bm = burn_scars["mask"]
            bo = np.zeros_like(img); bo[bm > 0] = [40, 0, 100]
            img = cv2.addWeighted(img, 1.0, bo, 0.45, 0)
            h0, w0 = img.shape[:2]
            cv2.putText(img, "BURN SCAR", (10, h0-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 0, 180), 2, cv2.LINE_AA)

        # ── Fire tint (orange) ────────────────────────────────────────
        if fire_smoke:
            if fire_smoke.get("fire_detected"):
                fm = fire_smoke["fire_mask"]
                fo = np.zeros_like(img); fo[fm > 0] = [0, 80, 255]
                img = cv2.addWeighted(img, 1.0, fo, 0.55, 0)
                h0, w0 = img.shape[:2]
                cv2.putText(img, "FIRE DETECTED", (10, h0-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2, cv2.LINE_AA)
            if fire_smoke.get("smoke_detected"):
                sm = fire_smoke["smoke_mask"]
                so = np.zeros_like(img); so[sm > 0] = [200, 200, 200]
                img = cv2.addWeighted(img, 1.0, so, 0.22, 0)

        # ── Building polygon outlines (cyan) ──────────────────────────
        if building_polygons:
            for poly in building_polygons:
                cv2.drawContours(img, [poly], -1, (0, 255, 220), 2)

        # ── Solar panel outlines (royal blue) ────────────────────────
        if solar_panels and solar_panels.get("regions"):
            for (x1, y1, x2, y2) in solar_panels["regions"]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(img, "solar", (x1+2, y1+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1, cv2.LINE_AA)

        # ── SegFormer-derived axis-aligned boxes ──────────────────────
        if seg_dets:
            for det in seg_dets:
                x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
                color = BOX_COLORS.get(det["category"], BOX_COLORS["_default"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                lbl = f"[S]{det['category']} {det['conf']:.0%}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                cv2.rectangle(img, (x1, max(y1-th-5,0)), (x1+tw+3, y1), color, -1)
                cv2.putText(img, lbl, (x1+2, max(y1-2, th)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,0,0), 1, cv2.LINE_AA)

        # ── YOLOv8-OBB oriented bounding boxes ───────────────────────
        if obb_dets:
            for det in obb_dets:
                pts   = np.array(det["points"], dtype=np.int32)   # (4, 2)
                color = OBB_COLORS.get(det["category"], OBB_COLORS["_default"])
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                lbl = f"[OBB]{det['cls_name']} {det['conf']:.2f}"
                x0, y0 = pts[0]
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(img, (x0, max(y0-th-5,0)), (x0+tw+3, y0), color, -1)
                cv2.putText(img, lbl, (x0+2, max(y0-2, th)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)

        # ── YOLO COCO boxes ───────────────────────────────────────────
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            cat   = CATEGORY_MAP.get(det["cls_name"], det["cls_name"].title())
            color = BOX_COLORS.get(cat, BOX_COLORS["_default"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            lbl = f"{det['cls_name']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(y1-th-6,0)), (x1+tw+4, y1), color, -1)
            cv2.putText(img, lbl, (x1+2, max(y1-3, th)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

        self._draw_legend(img)
        return img

    def _draw_legend(self, img: np.ndarray):
        items = [
            ("Buildings (poly)",  (0,   255, 220)),
            ("Roads",             (255, 240,   0)),
            ("Vehicles (YOLO)",   (0,   230,  80)),
            ("OBB objects",       (100, 100, 255)),
            ("Vegetation",        (34,  170,  34)),
            ("Water",             (30,  100, 220)),
            ("Solar panels",      (255, 100,   0)),
        ]
        x0   = img.shape[1] - 180
        y0   = 10
        pad  = 4
        lh   = 18
        bw   = 14
        bg_h = len(items) * lh + pad * 2
        cv2.rectangle(img, (x0-pad, y0-pad), (img.shape[1]-pad, y0+bg_h),
                      (20, 20, 20), -1)
        for i, (name, color) in enumerate(items):
            y = y0 + i * lh + lh // 2
            cv2.rectangle(img, (x0, y-5), (x0+bw, y+5), color, -1)
            cv2.putText(img, name, (x0+bw+5, y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════
    #  15. Scene Characterisation
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _scene(land: dict, objs: dict) -> str:
        veg   = land.get("Vegetation",  0) + land.get("Agriculture", 0)
        water = land.get("Water",       0) + land.get("Recreation",  0)
        urban = land.get("Urban",       0) + land.get("Buildings",   0)
        roads = land.get("Roads",       0) + land.get("Airport",     0)
        if land.get("Airport", 0) > 2 or objs.get("Aircraft", {}).get("count", 0) > 0:
            return "Airfield / Aviation Area"
        if water > 30 or objs.get("Watercraft", {}).get("count", 0) > 2:
            return "Coastal / Maritime Area"
        if veg > 55 and urban < 15:
            return "Natural / Rural"
        if urban > 35 and roads > 6:
            return "Urban Developed"
        if urban > 15 and veg > 25:
            return "Suburban / Campus"
        if objs.get("Vehicles", {}).get("count", 0) > 10:
            return "High-Traffic Zone"
        if land.get("Agriculture", 0) > 20:
            return "Agricultural / Farmland"
        return "Mixed / General"

    # ══════════════════════════════════════════════════════════════════════
    #  Main API
    # ══════════════════════════════════════════════════════════════════════

    def analyze(self, image_input, progress_cb=None):
        def _cb(p, m):
            if progress_cb:
                progress_cb(p, m)

        # ── Decode input ───────────────────────────────────────────────
        if isinstance(image_input, bytes):
            pil = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil = image_input.convert("RGB")
        else:
            pil = Image.fromarray(image_input)
        rgb    = np.array(pil)
        h_px, w_px = rgb.shape[:2]

        # ── 1. SegFormer ───────────────────────────────────────────────
        seg_pred         = None
        seg_overlay_np   = None
        land_pct         = None
        dominant         = None
        seg_building_cnt = 0
        seg_road_pct     = 0.0
        building_polys   = []
        seg_dets         = []
        water_info       = {"count": 0, "coverage_pct": 0.0}

        if self.seg_model is not None:
            _cb(0.03, "Running SegFormer semantic segmentation…")
            try:
                seg_pred = self._segment_scene(pil)
                if seg_pred is not None:
                    _cb(0.28, "Building segmentation overlay…")
                    seg_overlay_np   = self._seg_to_overlay(seg_pred, rgb)
                    land_pct         = self._seg_land_pct(seg_pred)
                    non_other        = {k: v for k, v in land_pct.items() if k != "Other"}
                    dominant         = max(non_other, key=non_other.get) if non_other else "Other"
                    seg_building_cnt = self._count_buildings(seg_pred)
                    seg_road_pct     = self._estimate_road_coverage(seg_pred)
                    _cb(0.33, "Extracting building footprint polygons…")
                    building_polys   = self._extract_building_polygons(seg_pred)
                    _cb(0.37, "Extracting segmentation-based object boxes…")
                    seg_dets         = self._extract_seg_boxes(seg_pred)
                    _cb(0.40, "Analysing water bodies…")
                    water_info       = self._analyze_water_bodies(seg_pred)
            except Exception:
                seg_pred = None; seg_overlay_np = None; land_pct = None

        # ── Fallback: HSV ──────────────────────────────────────────────
        if seg_pred is None:
            _cb(0.08, "Classifying terrain (HSV fallback)…")
            land_pct, hsv_overlay_np, dominant, land_masks = self._classify_land(rgb)
            seg_overlay_np = hsv_overlay_np
            _cb(0.28, "Extracting structural features (CV)…")
            cv_feat = self._extract_features_cv(rgb, land_masks)
            seg_building_cnt = cv_feat["estimated_structures"]
            seg_road_pct     = cv_feat["road_coverage_pct"]

        # ── 2. OBB aerial detection ────────────────────────────────────
        _cb(0.42, "Running OBB aerial detection (planes, ships, vehicles)…")
        obb_dets = self._detect_obb(rgb)

        # ── 3. Fire / smoke ────────────────────────────────────────────
        _cb(0.48, "Scanning for fire and smoke…")
        fire_smoke = self._detect_fire_smoke(rgb)

        # ── 4. Burn scars ──────────────────────────────────────────────
        _cb(0.50, "Detecting burn scars…")
        burn_scars = self._detect_burn_scars(rgb)

        # ── 5. Solar panels ────────────────────────────────────────────
        _cb(0.52, "Detecting solar panel arrays…")
        solar_panels = self._detect_solar_panels(rgb)

        # ── 6. Vegetation health ───────────────────────────────────────
        _cb(0.54, "Estimating vegetation health (ExG index)…")
        veg_health = self._estimate_vegetation_health(rgb)

        # ── 7. Cloud / shadow masking ──────────────────────────────────
        _cb(0.56, "Masking clouds and shadows…")
        cloud_info = self._mask_clouds(rgb)

        # ── 8. Parking lots ────────────────────────────────────────────
        _cb(0.58, "Detecting parking lots…")
        parking_count = self._detect_parking_lots(rgb)

        # ── 9. Green zones & open areas (HSV) ─────────────────────────
        _cb(0.60, "Detecting vegetation zones and open areas…")
        hsv2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h2, s2, v2 = hsv2[:,:,0], hsv2[:,:,1], hsv2[:,:,2]
        total_px   = h_px * w_px
        veg_mask   = ((h2>=35)&(h2<=90)&(s2>=35)&(v2>=30)).astype(np.uint8)
        veg_cl     = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, np.ones((20,20),np.uint8))
        n_veg, _, vst, _ = cv2.connectedComponentsWithStats(veg_cl, 8)
        green_zones = sum(1 for i in range(1,n_veg)
                          if vst[i,cv2.CC_STAT_AREA] >= int(total_px*0.004))
        bare_mask  = (((h2>=10)&(h2<=35)&(s2>=25)&(v2>=40))|
                      ((h2<=10)&(s2>=25)&(v2>=40))).astype(np.uint8)
        n_bare, _, bst, _ = cv2.connectedComponentsWithStats(bare_mask, 8)
        open_areas  = sum(1 for i in range(1,n_bare)
                          if bst[i,cv2.CC_STAT_AREA] >= int(total_px*0.003))

        # ── 10. Tiled YOLO detection ───────────────────────────────────
        _cb(0.62, "Running tiled YOLO object detection (cars, people, trucks)…")
        detections = self._detect_tiled(rgb, progress_cb=progress_cb)

        # ── 11. Compose annotated image ────────────────────────────────
        _cb(0.90, "Compositing annotated map…")
        annotated_np = self._annotate(
            base             = rgb,
            dets             = detections,
            seg_overlay      = seg_overlay_np,
            building_polygons= building_polys,
            seg_pred         = seg_pred,
            seg_dets         = seg_dets,
            obb_dets         = obb_dets,
            fire_smoke       = fire_smoke,
            burn_scars       = burn_scars,
            solar_panels     = solar_panels,
            cloud_info       = cloud_info,
        )

        # ── 12. Object summaries ───────────────────────────────────────
        detected_objects: dict = {}
        for det in detections:
            cat = CATEGORY_MAP.get(det["cls_name"], det["cls_name"].title())
            if cat not in detected_objects:
                detected_objects[cat] = {"count": 0, "confidence": []}
            detected_objects[cat]["count"] += 1
            detected_objects[cat]["confidence"].append(det["conf"])
        for cat, data in detected_objects.items():
            data["avg_confidence"] = round(
                sum(data["confidence"]) / len(data["confidence"]), 3)

        # OBB detections merged into detected_objects
        for od in obb_dets:
            cat = od["category"]
            if cat not in detected_objects:
                detected_objects[cat] = {"count": 0, "confidence": [], "avg_confidence": 0.0}
            detected_objects[cat]["count"] += 1
            detected_objects[cat]["confidence"].append(od["conf"])
            detected_objects[cat]["avg_confidence"] = round(
                sum(detected_objects[cat]["confidence"])
                / len(detected_objects[cat]["confidence"]), 3)

        all_detections = [
            {"class": d["cls_name"],
             "category": CATEGORY_MAP.get(d["cls_name"], d["cls_name"].title()),
             "confidence": d["conf"], "bbox": d["xyxy"]}
            for d in detections
        ]

        features = {
            "estimated_structures":  seg_building_cnt,
            "building_footprints":   len(building_polys),
            "road_coverage_pct":     seg_road_pct,
            "green_zones":           green_zones,
            "open_areas":            open_areas,
            "water_bodies":          water_info["count"],
            "water_coverage_pct":    water_info["coverage_pct"],
            "parking_lots":          parking_count,
            "fire_detected":         fire_smoke["fire_detected"],
            "smoke_detected":        fire_smoke["smoke_detected"],
            "fire_coverage_pct":     fire_smoke["fire_coverage"],
            "smoke_coverage_pct":    fire_smoke["smoke_coverage"],
            "burn_scars":            burn_scars["count"],
            "burn_scar_pct":         burn_scars["coverage_pct"],
            "solar_panel_regions":   solar_panels["count"],
            "solar_coverage_pct":    solar_panels["coverage_pct"],
            "cloud_coverage_pct":    cloud_info["cloud_pct"],
            "shadow_coverage_pct":   cloud_info["shadow_pct"],
            "usable_area_pct":       cloud_info["usable_pct"],
            "obb_detections":        len(obb_dets),
            "veg_health_status":     veg_health["status"],
            "veg_health_index":      veg_health["health_index"],
            "veg_healthy_pct":       veg_health["healthy_pct"],
        }

        scene = self._scene(land_pct, detected_objects)
        _cb(1.0, "Satellite analysis complete.")

        return {
            "annotated_image":     Image.fromarray(annotated_np),
            "land_overlay":        Image.fromarray(seg_overlay_np)
                                   if seg_overlay_np is not None
                                   else Image.fromarray(rgb),
            "land_classification": land_pct,
            "dominant_land":       dominant,
            "scene_type":          scene,
            "detected_objects":    detected_objects,
            "all_detections":      all_detections,
            "total_objects":       len(detections) + len(obb_dets),
            "features":            features,
            "image_size":          (w_px, h_px),
            "segmentation_used":   seg_pred is not None,
            "obb_used":            self.obb_model is not None,
            "veg_health":          veg_health,
            "cloud_info":          cloud_info,
        }
