"""
Fusion Engine — Core Novelty Module
=====================================
Multi-Modal Geospatial Intelligence Fusion

Combines satellite image analysis with drone/UAV video analysis to produce
a single, high-confidence unified area intelligence assessment.

Fusion Strategy
---------------
1. Object Inventory Fusion   — reconcile detections from both sources using
                                probabilistic confidence combination.
2. Scene Characterisation    — classify environment type from land + objects.
3. Activity & Movement Score — quantify dynamics using drone tracking data.
4. Spatial Zone Heatmap      — 3×3 grid object-density map from satellite bboxes.
5. Threat / Alert Level      — tiered assessment (LOW / MEDIUM / HIGH).
6. Quality Metrics           — agreement rate, improvement over single-source.
7. Recommendations           — actionable intelligence derived from all above.
"""

from __future__ import annotations
from datetime import datetime
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────

LAND_ACTIVITY_WEIGHT = {
    "Urban": 0.9,
    "Bare Ground": 0.35,
    "Snow/Clouds": 0.05,
    "Vegetation": 0.20,
    "Water": 0.10,
}

SCENE_RULES = [
    # (condition_fn, label)
    (lambda l, o: l.get("Water", 0) > 30,                      "Coastal / Riverine"),
    (lambda l, o: l.get("Vegetation", 0) > 55,                 "Natural / Rural"),
    (lambda l, o: l.get("Urban", 0) > 50
                  and o.get("Vehicles", {}).get("count", 0) > 15,  "Industrial / Commercial"),
    (lambda l, o: l.get("Urban", 0) > 30
                  and o.get("People", {}).get("count", 0) > 5,     "Residential Urban"),
    (lambda l, o: o.get("Vehicles", {}).get("count", 0) > 25,  "High-Traffic Zone"),
    (lambda l, o: o.get("Aircraft", {}).get("count", 0) > 0,   "Airfield / Aviation Area"),
    (lambda l, o: o.get("Watercraft", {}).get("count", 0) > 0, "Maritime / Port Area"),
]

THREAT_BANDS = [
    (0.70, "HIGH",   "🔴"),
    (0.40, "MEDIUM", "🟡"),
    (0.00, "LOW",    "🟢"),
]

# How much does drone vs satellite count contribute to the fused count?
DRONE_COUNT_WEIGHT = 0.65   # drone is closer → more accurate for small objects
SAT_COUNT_WEIGHT   = 0.35


# ──────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _prob_union(p_a: float, p_b: float) -> float:
    """P(detected by at least one source) = 1 − P(missed by both)."""
    return 1.0 - (1.0 - p_a) * (1.0 - p_b)


def _weighted_count(sat_cnt: int, drn_cnt: int) -> int:
    """Weighted blend of two detection counts, rounded to nearest int."""
    return max(1, round(SAT_COUNT_WEIGHT * sat_cnt + DRONE_COUNT_WEIGHT * drn_cnt))


def _threat_level(score: float):
    for threshold, level, icon in THREAT_BANDS:
        if score >= threshold:
            return level, icon
    return "LOW", "🟢"


def _scene_type(land: dict, objects: dict) -> str:
    for cond, label in SCENE_RULES:
        if cond(land, objects):
            return label
    return "Mixed / General"


# ──────────────────────────────────────────────────────────────────────────────
#  FusionEngine
# ──────────────────────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Fuses satellite and drone analysis results into a unified intelligence report.
    All methods are stateless; results depend only on their inputs.
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fuse(self, satellite: dict, drone: dict) -> dict:
        """
        Parameters
        ----------
        satellite : dict   Output of SatelliteAnalyzer.analyze()
        drone     : dict   Output of DroneAnalyzer.analyze()

        Returns
        -------
        dict   Complete fusion intelligence report
        """
        sat_objs  = satellite.get("detected_objects", {})
        drn_objs  = drone.get("detected_objects", {})
        land      = satellite.get("land_classification", {})
        dominant  = satellite.get("dominant_land", "Unknown")

        # ── 1. Object Inventory Fusion ──────────────────────────────────
        all_classes = set(sat_objs) | set(drn_objs)
        fused_inventory: dict = {}

        for cls in all_classes:
            s = sat_objs.get(cls)
            d = drn_objs.get(cls)

            if s and d:
                fused_conf  = _prob_union(
                    s.get("avg_confidence", 0.5),
                    d.get("avg_confidence", 0.5)
                )
                fused_count = _weighted_count(s["count"], d["count"])
                source      = "BOTH ✓"
            elif s:
                fused_conf  = s.get("avg_confidence", 0.5) * 0.78  # single-source penalty
                fused_count = s["count"]
                source      = "Satellite"
            else:
                fused_conf  = d.get("avg_confidence", 0.5) * 0.88  # drone slightly less penalised
                fused_count = d["count"]
                source      = "Drone"

            fused_inventory[cls] = {
                "count":      fused_count,
                "confidence": round(fused_conf, 3),
                "source":     source,
            }

        # ── 2. Scene Characterisation ────────────────────────────────────
        scene = _scene_type(land, fused_inventory)

        # ── 3. Activity & Movement Score ─────────────────────────────────
        total_fused    = sum(v["count"] for v in fused_inventory.values())
        total_tracks   = drone.get("total_tracks", 0)
        fast_movers    = drone.get("fast_movers", 0)
        avg_speed      = drone.get("avg_track_speed", 0.0)
        fire_pct       = drone.get("fire_pct_frames", 0.0)
        loitering_cnt  = drone.get("loitering_count", 0)
        loitering_ppl  = drone.get("loitering_people", 0)

        movement_ratio = fast_movers / max(total_tracks, 1)
        land_weight    = LAND_ACTIVITY_WEIGHT.get(dominant, 0.5)

        # Hazard bonus: fire and loitering raise the activity score
        hazard_bonus = min(fire_pct / 100.0 * 0.5 + loitering_ppl * 0.05, 0.35)

        activity_score = round(
            min(
                0.35 * min(total_fused / 60.0, 1.0) +
                0.30 * movement_ratio +
                0.20 * land_weight +
                0.15 * hazard_bonus * 6,   # normalise bonus into 0–1 range
                1.0
            ),
            3
        )

        # ── 4. Threat Level ──────────────────────────────────────────────
        threat_level, threat_icon = _threat_level(activity_score)
        # Hard overrides: fire or loitering people always trigger at least MEDIUM
        if fire_pct > 5.0:
            threat_level, threat_icon = "HIGH",   "🔴"
        elif fire_pct > 1.0 or loitering_ppl > 0:
            if threat_level == "LOW":
                threat_level, threat_icon = "MEDIUM", "🟡"

        # ── 5. Fusion Quality Metrics ────────────────────────────────────
        sat_total = sum(v["count"] for v in sat_objs.values()) if sat_objs else 0
        drn_total = sum(v["count"] for v in drn_objs.values()) if drn_objs else 0

        both_count     = sum(1 for c in all_classes if c in sat_objs and c in drn_objs)
        agreement_rate = round(both_count / max(len(all_classes), 1), 3)

        best_single    = max(sat_total, drn_total, 1)
        improvement    = round((total_fused - best_single) / best_single * 100, 1)

        # Fusion score 0–100 (how complementary the two sources are)
        complementarity = round(
            (1.0 - abs(agreement_rate - 0.5) * 2) * 100, 1
        )
        fusion_score = round(
            0.5 * agreement_rate * 100 +
            0.3 * complementarity +
            0.2 * min(improvement + 50, 100),
            1
        )

        # ── 6. Spatial Zone Heatmap (3×3 grid from satellite bboxes) ─────
        zones = self._compute_zones(satellite, drone)

        # ── 7. Co-detection Matrix ───────────────────────────────────────
        co_matrix = self._co_detection_matrix(satellite)

        # ── 8. Summary & Recommendations ─────────────────────────────────
        summary = self._build_summary(
            land, dominant, fused_inventory, scene,
            activity_score, threat_level, fast_movers, total_tracks
        )
        recommendations = self._build_recommendations(
            threat_level, fused_inventory, activity_score, land,
            scene, avg_speed, total_tracks,
            fire_pct=fire_pct, loitering_people=loitering_ppl,
            drone_alerts=drone.get("alerts", []),
        )

        return {
            "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Inventory
            "fused_inventory":         fused_inventory,
            "total_objects_detected":  total_fused,
            "classes_detected":        len(all_classes),
            # Land
            "land_classification":     land,
            "dominant_land":           dominant,
            "scene_type":              scene,
            # Activity
            "activity_score":          activity_score,
            "movement_ratio":          round(movement_ratio, 3),
            "fast_movers":             fast_movers,
            "total_tracks":            total_tracks,
            "avg_track_speed":         avg_speed,
            # Threat
            "threat_level":            threat_level,
            "threat_icon":             threat_icon,
            # Quality metrics
            "agreement_rate":          agreement_rate,
            "fusion_improvement":      improvement,
            "fusion_score":            fusion_score,
            "satellite_object_count":  sat_total,
            "drone_object_count":      drn_total,
            # Spatial
            "zone_analysis":           zones,
            "co_detection_matrix":     co_matrix,
            # Text
            "summary":                 summary,
            "recommendations":         recommendations,
        }

    # ------------------------------------------------------------------ #
    #  Spatial zone heatmap                                                #
    # ------------------------------------------------------------------ #

    def _compute_zones(self, satellite: dict, drone: dict) -> dict:
        """
        Map satellite detections onto a 3×3 compass grid using relative
        pixel coordinates from their bounding boxes.
        Also factor in drone fast-mover count into the centre cell.
        """
        zone_names = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
        zones = {z: 0 for z in zone_names}

        w, h = satellite.get("image_size", (1000, 1000))

        col_map = {0: "W_", 1: "_", 2: "E_"}  # prefix
        row_map = {0: "N",  1: "",  2: "S"}

        for det in satellite.get("all_detections", []):
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2 / max(w, 1)
            cy = (bbox[1] + bbox[3]) / 2 / max(h, 1)
            col = 0 if cx < 0.333 else (1 if cx < 0.667 else 2)
            row = 0 if cy < 0.333 else (1 if cy < 0.667 else 2)

            grid_to_zone = {
                (0, 0): "NW", (1, 0): "N",  (2, 0): "NE",
                (0, 1): "W",  (1, 1): "C",  (2, 1): "E",
                (0, 2): "SW", (1, 2): "S",  (2, 2): "SE",
            }
            zones[grid_to_zone[(col, row)]] += 1

        # Drone fast movers contribute to the central zone (area of interest)
        zones["C"] += drone.get("fast_movers", 0)

        return zones

    # ------------------------------------------------------------------ #
    #  Co-detection matrix                                                 #
    # ------------------------------------------------------------------ #

    def _co_detection_matrix(self, satellite: dict) -> dict:
        """
        Count how many satellite frames/tiles have co-occurring object pairs.
        (Uses satellite all_detections to build pair frequency.)
        """
        frame_classes: dict[str, set] = {}
        for det in satellite.get("all_detections", []):
            cat = det["category"]
            # Use bbox quadrant as a proxy for "same region"
            bbox = det["bbox"]
            key = f"{int(bbox[0]//200)}_{int(bbox[1]//200)}"
            frame_classes.setdefault(key, set()).add(cat)

        co: dict = {}
        for classes in frame_classes.values():
            cls_list = sorted(classes)
            for i in range(len(cls_list)):
                for j in range(i + 1, len(cls_list)):
                    pair = f"{cls_list[i]}+{cls_list[j]}"
                    co[pair] = co.get(pair, 0) + 1

        return co

    # ------------------------------------------------------------------ #
    #  Intelligence summary                                                #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        land: dict,
        dominant: str,
        inventory: dict,
        scene: str,
        activity: float,
        threat: str,
        fast: int,
        tracks: int,
    ) -> str:
        veg   = land.get("Vegetation", 0)
        water = land.get("Water", 0)
        urban = land.get("Urban", 0)

        land_desc = f"{dominant.lower()}-dominant terrain ({urban:.0f}% urban, {veg:.0f}% vegetation)"
        if water > 10:
            land_desc += f", {water:.0f}% water coverage"

        obj_parts = [
            f"{d['count']} {cls.lower()}"
            for cls, d in sorted(inventory.items(), key=lambda x: -x[1]["count"])[:5]
        ]
        obj_str = ", ".join(obj_parts) if obj_parts else "no significant objects"

        activity_str = (
            "high operational activity"  if activity > 0.70 else
            "moderate activity"          if activity > 0.40 else
            "low ambient activity"
        )

        movement_str = ""
        if tracks > 0:
            pct = round(fast / tracks * 100)
            movement_str = (
                f" Drone tracking identified {tracks} unique objects, "
                f"of which {pct}% exhibited significant movement."
            )

        return (
            f"The surveyed area is characterised as {scene} — {land_desc}. "
            f"Multi-modal fusion detected {obj_str}. "
            f"The operational picture indicates {activity_str} "
            f"with a composite threat assessment of {threat}.{movement_str} "
            f"Satellite imagery provided wide-area context and static structure mapping, "
            f"while drone footage contributed real-time micro-level object intelligence "
            f"and movement dynamics."
        )

    # ------------------------------------------------------------------ #
    #  Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _build_recommendations(
        self,
        threat: str,
        inventory: dict,
        activity: float,
        land: dict,
        scene: str,
        avg_speed: float,
        tracks: int,
        fire_pct: float = 0.0,
        loitering_people: int = 0,
        drone_alerts: list | None = None,
    ) -> list[str]:
        recs: list[str] = []
        drone_alerts = drone_alerts or []

        # ── Fire / smoke ───────────────────────────────────────────────
        if fire_pct > 5.0:
            recs.append(
                f"FIRE ALERT — Active fire detected in {fire_pct:.0f}% of drone frames. "
                "Dispatch emergency response immediately and establish exclusion zone."
            )
        elif fire_pct > 1.0:
            recs.append(
                f"Fire signature detected ({fire_pct:.1f}% of frames) — "
                "verify with ground unit; possible wildfire or controlled burn."
            )

        # ── Loitering ─────────────────────────────────────────────────
        if loitering_people > 0:
            recs.append(
                f"Loitering behaviour identified for {loitering_people} person(s) — "
                "review drone footage and assess intent; consider intercept if in restricted zone."
            )

        # ── Drone behavioural alerts (pass-through) ────────────────────
        for alert in drone_alerts[:3]:
            recs.append(f"Drone alert: {alert}")

        # ── Threat-level base recommendation ──────────────────────────
        if threat == "HIGH":
            recs.append(
                "PRIORITY — Immediate enhanced surveillance required; "
                "high object density and activity levels exceed normal thresholds."
            )
            recs.append("Recommend deploying additional monitoring assets and alerting response teams.")
        elif threat == "MEDIUM":
            recs.append("Elevated activity detected — maintain close monitoring and schedule follow-up within 6 hours.")
        else:
            recs.append("Area is within normal operational parameters — routine periodic surveillance is sufficient.")

        vehicles = inventory.get("Vehicles", {}).get("count", 0)
        people   = inventory.get("People",   {}).get("count", 0)
        aircraft = inventory.get("Aircraft", {}).get("count", 0)
        maritime = inventory.get("Watercraft", {}).get("count", 0)

        if vehicles > 20:
            recs.append(
                f"High vehicle density ({vehicles} units) — recommend traffic-pattern analysis "
                "and verify route access points."
            )
        if people > 10:
            recs.append(
                f"Significant human presence ({people} individuals) — apply crowd-density protocols."
            )
        if aircraft > 0:
            recs.append(f"Aircraft detected ({aircraft}) — cross-reference with flight logs for authorisation.")
        if maritime > 0:
            recs.append(f"Watercraft detected ({maritime}) — monitor port/coastal access.")

        water_pct = land.get("Water", 0)
        if water_pct > 25:
            recs.append("Significant water bodies present — assess for flood risk or maritime activity.")

        if "Industrial" in scene or "Commercial" in scene:
            recs.append("Industrial/commercial zone — prioritise infrastructure integrity assessment.")
        if "Aviation" in scene:
            recs.append("Airfield proximity confirmed — enforce airspace clearance protocols.")

        if activity > 0.60 and tracks > 5:
            recs.append(
                f"Elevated movement dynamics (avg track speed {avg_speed:.1f} px/frame) — "
                "temporal correlation analysis recommended to identify patterns."
            )

        if not recs:
            recs.append("No actionable alerts — continue standard monitoring cadence.")

        return recs
