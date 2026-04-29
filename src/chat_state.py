"""
chat_state.py
=============
Persistent shared state + background HTTP server for the CIPHER floating
Intelligence Assistant.

No external API required — all answers come from keyword-matching rules
applied to the stored analysis data.
"""
from __future__ import annotations
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

CIPHER_CHAT_STATE: dict = {
    "data":     {},     # {"sat": ..., "drn": ..., "fus": ...} from last fusion run
    "briefing": "",     # pre-generated plain-language summary
    "ready":    False,
    "_fus_ts":  "",
}

_server_port: int = 0
_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
#  Briefing generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_briefing(data: dict) -> str:
    sat = data.get("sat", {})
    drn = data.get("drn", {})
    fus = data.get("fus", {})

    threat    = fus.get("threat_level", "—")
    score     = fus.get("fusion_score", 0)
    total     = fus.get("total_objects_detected", 0)
    activity  = fus.get("activity_score", 0)
    dom_land  = fus.get("dominant_land", sat.get("dominant_land", "—"))
    alerts    = drn.get("alerts", [])
    recs      = fus.get("recommendations", [])
    fast      = fus.get("fast_movers", 0)
    loitering = drn.get("loitering_people", 0)
    fire_sat  = sat.get("features", {}).get("fire_detected", False)
    smoke_sat = sat.get("features", {}).get("smoke_detected", False)
    fire_pct  = drn.get("fire_pct_frames", 0)
    smoke_pct = drn.get("smoke_pct_frames", 0)

    threat_plain = {
        "LOW":      "LOW — the area looks calm with little unusual activity.",
        "MODERATE": "MODERATE — some activity worth monitoring was detected.",
        "HIGH":     "HIGH — significant activity or hazards found. Closer attention advised.",
        "CRITICAL": "CRITICAL — serious threats detected. Immediate attention may be needed.",
    }.get(threat.upper(), f"{threat} — review the full report for details.")

    lines = [
        "Here is a plain-language summary of this CIPHER analysis.",
        "",
        f"Threat level: {threat_plain}",
        f"{total} objects were detected in total (satellite + drone combined), "
        f"with a fusion quality score of {score:.0f}/100.",
        f"The area is mainly {dom_land} terrain. Activity score: {activity:.0%}.",
    ]

    if fire_sat or fire_pct > 5:
        lines.append(
            f"Fire signals detected — satellite: {'yes' if fire_sat else 'no'}, "
            f"drone video: fire in {fire_pct:.0f}% of frames."
        )
    if smoke_sat or smoke_pct > 5:
        lines.append(f"Smoke detected in {smoke_pct:.0f}% of drone frames — exercise caution.")
    if fast > 0:
        lines.append(f"{fast} fast-moving object(s) were tracked by the drone.")
    if loitering > 0:
        lines.append(f"{loitering} person(s) were flagged for loitering.")
    if alerts:
        lines.append(f"Key alert: {alerts[0]}")
    if recs:
        lines.append(f"Top recommendation: {recs[0]}")
    lines += [
        "",
        "Ask me anything about this analysis — threat level, objects detected, "
        "land cover, alerts, recommendations, or drone stats.",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Rule-based Q&A engine
# ──────────────────────────────────────────────────────────────────────────────

def _kw(text: str, *words: str) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def answer_question(question: str) -> str:
    data = CIPHER_CHAT_STATE.get("data", {})
    if not data:
        return (
            "No analysis data loaded yet. "
            "Please run the Fusion Engine first, then ask your question."
        )

    sat = data.get("sat", {})
    drn = data.get("drn", {})
    fus = data.get("fus", {})
    q   = question.strip()

    # ── Greeting ───────────────────────────────────────────────────────────
    if _kw(q, "hello", "hi ", "hey ", "howdy", "greetings", "good morning",
             "good afternoon", "good evening"):
        threat = fus.get("threat_level", "—")
        return (
            f"Hello! I'm the CIPHER Intelligence Assistant. "
            f"This analysis shows a {threat} threat level. "
            f"Ask me about detected objects, land cover, alerts, "
            f"recommendations, or anything else from this report."
        )

    # ── Summary / overview ─────────────────────────────────────────────────
    if _kw(q, "summar", "overview", "brief", "explain", "what happened",
             "tell me about", "give me", "describe", "what is this",
             "what does this", "what did", "what was found", "what's going on",
             "what happened", "show me"):
        return CIPHER_CHAT_STATE.get("briefing") or generate_briefing(data)

    # ── Threat / safety ────────────────────────────────────────────────────
    if _kw(q, "threat", "danger", "risk", "safe", "hazard", "concern",
             "how bad", "how serious", "severity", "emergency", "critical",
             "how dangerous"):
        threat   = fus.get("threat_level", "—")
        score    = fus.get("fusion_score", 0)
        activity = fus.get("activity_score", 0)
        desc = {
            "LOW":      (
                "The area is calm — no significant threats were detected. "
                "Routine monitoring is sufficient."
            ),
            "MODERATE": (
                "Some unusual activity was spotted — worth monitoring but not an emergency."
            ),
            "HIGH":     (
                "Significant activity or hazards were detected. "
                "Closer attention or investigation is strongly advised."
            ),
            "CRITICAL": (
                "Serious threats or hazards detected. "
                "Immediate attention or response may be required."
            ),
        }.get(threat.upper(), f"Threat level is {threat}.")
        return (
            f"Threat level: {threat}. {desc} "
            f"Activity score: {activity:.0%}. Fusion quality score: {score:.0f}/100."
        )

    # ── Object detection ───────────────────────────────────────────────────
    if _kw(q, "object", "detect", "found", "see", "saw", "spot",
             "how many", "count", "total", "identified", "what is there",
             "what's there", "what are there", "inventory", "things found",
             "items", "things detected"):
        total     = fus.get("total_objects_detected", 0)
        inventory = fus.get("fused_inventory", {})
        sat_cnt   = fus.get("satellite_object_count", 0)
        drn_cnt   = fus.get("drone_object_count", 0)
        top = sorted(inventory.items(), key=lambda x: -x[1]["count"])[:5]
        top_str = ", ".join(f"{c} ×{d['count']}" for c, d in top) if top else "none"
        return (
            f"{total} objects detected in total — "
            f"{sat_cnt} from satellite, {drn_cnt} from drone. "
            f"Top objects: {top_str}."
        )

    # ── People / persons ───────────────────────────────────────────────────
    if _kw(q, "person", "people", "human", "pedestrian", "crowd",
             "loiter", "individual", "civilian", "man", "woman", "who"):
        inventory   = fus.get("fused_inventory", {})
        person_keys = [k for k in inventory
                       if any(p in k.lower() for p in ("person", "pedestrian", "people"))]
        person_count  = sum(inventory[k]["count"] for k in person_keys)
        loiter_tracks = drn.get("loitering_count", 0)
        loiter_people = drn.get("loitering_people", 0)
        sat_people = sum(
            v["count"] for k, v in sat.get("detected_objects", {}).items()
            if any(p in k.lower() for p in ("person", "pedestrian", "people"))
        )
        drn_people = sum(
            v["count"] for k, v in drn.get("detected_objects", {}).items()
            if any(p in k.lower() for p in ("person", "pedestrian", "people"))
        )
        loiter_msg = (
            f"{loiter_people} person(s) flagged for loitering."
            if loiter_tracks > 0
            else "No loitering was detected."
        )
        return (
            f"Approximately {person_count} people detected — "
            f"{sat_people} via satellite, {drn_people} via drone. "
            f"{loiter_msg}"
        )

    # ── Vehicles ───────────────────────────────────────────────────────────
    if _kw(q, "vehicle", "car", "truck", "van", "bus", "motor",
             "tricycle", "bicycle", "transport", "traffic", "automobile"):
        inventory = fus.get("fused_inventory", {})
        veh_keys  = [
            k for k in inventory
            if any(v in k.lower() for v in
                   ("car", "truck", "van", "bus", "motor", "bicycle",
                    "tricycle", "vehicle", "automobile"))
        ]
        if veh_keys:
            parts = [f"{k} ×{inventory[k]['count']}" for k in veh_keys]
            return f"Vehicles detected: {', '.join(parts)}."
        return "No vehicles were specifically detected in this analysis."

    # ── Fire / smoke ───────────────────────────────────────────────────────
    if _kw(q, "fire", "flame", "smoke", "burn", "blaze", "heat", "burning"):
        fire_sat  = sat.get("features", {}).get("fire_detected", False)
        smoke_sat = sat.get("features", {}).get("smoke_detected", False)
        fire_pct  = drn.get("fire_pct_frames", 0)
        smoke_pct = drn.get("smoke_pct_frames", 0)
        if not fire_sat and not smoke_sat and fire_pct < 1 and smoke_pct < 1:
            return (
                "No fire or smoke was detected in either the satellite image "
                "or the drone video. The area appears clear of fire hazards."
            )
        parts = []
        if fire_sat:
            parts.append("Satellite image detected fire signatures.")
        if smoke_sat:
            parts.append("Smoke detected in the satellite image.")
        if fire_pct >= 1:
            parts.append(f"Drone video showed fire in {fire_pct:.0f}% of frames.")
        if smoke_pct >= 1:
            parts.append(f"Smoke appeared in {smoke_pct:.0f}% of drone frames.")
        return " ".join(parts) + " Exercise caution in this area."

    # ── Movement / tracking ────────────────────────────────────────────────
    if _kw(q, "move", "motion", "speed", "fast", "track", "moving",
             "velocity", "rapid", "slow", "activity"):
        fast        = fus.get("fast_movers", 0)
        avg_speed   = fus.get("avg_track_speed", 0)
        total_tracks = fus.get("total_tracks", drn.get("total_tracks", 0))
        mvmt        = fus.get("movement_ratio", 0)
        return (
            f"{total_tracks} unique objects tracked by the drone. "
            f"{fast} were classified as fast-moving. "
            f"Average speed: {avg_speed:.1f} px/frame. "
            f"Scene movement ratio: {mvmt:.0%}."
        )

    # ── Land / terrain / vegetation ────────────────────────────────────────
    if _kw(q, "land", "terrain", "area", "forest", "tree", "vegetation",
             "road", "building", "water", "cover", "urban", "rural",
             "dominant", "ground", "surface", "zone"):
        land = fus.get("land_classification", sat.get("land_classification", {}))
        dom  = fus.get("dominant_land", sat.get("dominant_land", "—"))
        veg  = sat.get("veg_health", {}).get("status", "—")
        if land:
            breakdown = ", ".join(
                f"{k} {v:.0f}%"
                for k, v in sorted(land.items(), key=lambda x: -x[1]) if v > 0
            )
            return (
                f"Dominant land type: {dom}. "
                f"Full breakdown: {breakdown}. "
                f"Vegetation health: {veg}."
            )
        return f"Dominant land type: {dom}. Vegetation health: {veg}."

    # ── Alerts ─────────────────────────────────────────────────────────────
    if _kw(q, "alert", "warn", "flag", "unusual", "anomal", "issue",
             "problem", "incident", "unusual", "suspicious"):
        alerts = drn.get("alerts", [])
        if not alerts:
            return "No behavioural alerts were raised. The scene appears normal."
        return "Alerts raised during this analysis:\n" + "\n".join(f"• {a}" for a in alerts)

    # ── Recommendations ────────────────────────────────────────────────────
    if _kw(q, "recommend", "suggest", "action", "should", "next step",
             "what to do", "advice", "do next", "response", "plan"):
        recs = fus.get("recommendations", [])
        if not recs:
            return "No specific recommendations were generated for this analysis."
        return "Intelligence recommendations:\n" + "\n".join(
            f"{i+1}. {r}" for i, r in enumerate(recs)
        )

    # ── Fusion score / quality ─────────────────────────────────────────────
    if _kw(q, "fusion", "score", "quality", "accuracy", "agreement",
             "confidence", "reliable", "improvement", "combined"):
        score   = fus.get("fusion_score", 0)
        agree   = fus.get("agreement_rate", 0)
        improve = fus.get("fusion_improvement", 0)
        sat_cnt = fus.get("satellite_object_count", 0)
        drn_cnt = fus.get("drone_object_count", 0)
        return (
            f"Fusion quality score: {score:.0f}/100. "
            f"Source agreement rate: {agree:.0%}. "
            f"Combining both sensors improved detection by {improve:+.1f}% "
            f"over the best single source "
            f"(satellite: {sat_cnt}, drone: {drn_cnt} detections)."
        )

    # ── Drone / video ──────────────────────────────────────────────────────
    if _kw(q, "drone", "video", "footage", "uav", "aerial", "clip",
             "frame", "duration", "camera"):
        dur    = drn.get("video_duration", 0)
        frames = drn.get("total_frames", 0)
        scene  = drn.get("dominant_scene", "—")
        tracks = drn.get("total_tracks", fus.get("total_tracks", 0))
        model  = "VisDrone + COCO" if drn.get("visdrone_active") else "standard COCO"
        return (
            f"Drone video: {dur:.0f} s, {frames} frames analysed. "
            f"Detection model: {model}. "
            f"Dominant scene: {scene}. "
            f"{tracks} unique objects tracked."
        )

    # ── Satellite ──────────────────────────────────────────────────────────
    if _kw(q, "satellite", "overhead", "remote sens", "sat img",
             "sensor", "imagery", "image"):
        scene   = sat.get("scene_type", "—")
        dom     = sat.get("dominant_land", "—")
        sat_cnt = fus.get("satellite_object_count", 0)
        veg     = sat.get("veg_health", {}).get("status", "—")
        return (
            f"Satellite image type: {scene}. "
            f"Dominant land cover: {dom}. "
            f"{sat_cnt} objects detected from satellite. "
            f"Vegetation health: {veg}."
        )

    # ── Off-topic fallback ─────────────────────────────────────────────────
    return (
        "I can only answer questions about this specific CIPHER analysis. "
        "Try asking about: threat level, detected objects, people/vehicles, "
        "fire/smoke, land cover, alerts, recommendations, or the fusion score."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  HTTP request handler
# ──────────────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        payload = json.dumps({
            "ready":    CIPHER_CHAT_STATE["ready"],
            "briefing": CIPHER_CHAT_STATE["briefing"],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        n       = int(self.headers.get("Content-Length", 0))
        body    = json.loads(self.rfile.read(n))
        history = body.get("history", [])
        # Use the last user message for rule matching
        user_msg = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
        )
        reply = answer_question(user_msg)
        payload = json.dumps({"response": reply}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  Server lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def ensure_server(start_port: int = 8756) -> int:
    """Start the background HTTP server exactly once; return the bound port (0 = failed)."""
    global _server_port
    with _lock:
        if _server_port:
            return _server_port
        for port in range(start_port, start_port + 10):
            try:
                srv = HTTPServer(("127.0.0.1", port), _Handler)
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                _server_port = port
                return port
            except OSError:
                continue
    return 0
