"""
CIPHER — Combined Intelligence Platform for High-resolution Environmental Recon
================================================================================
Multi-Modal Geospatial Intelligence: Satellite + Drone Fusion
Run with:  streamlit run app.py
"""

from __future__ import annotations
import sys
import os
import io
import time
import base64 as _b64_top

# Make sure src/ is importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image

from src.satellite_analyzer import SatelliteAnalyzer
from src.drone_analyzer import DroneAnalyzer
from src.fusion_engine import FusionEngine
from src.report_generator import generate_text_report, generate_csv_report, generate_pdf_report
from src.chat_state import CIPHER_CHAT_STATE, ensure_server, generate_briefing as _gen_briefing
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CIPHER — Combined Intelligence Platform for High-resolution Environmental Recon",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_chat_port = ensure_server()   # start background chat server (no-op on reruns)

# Clear stale analysis data for every new browser session so the chatbot
# never shows results from a previous session before a new fusion is run.
if "cipher_session_init" not in st.session_state:
    st.session_state.cipher_session_init = True
    CIPHER_CHAT_STATE.update({"data": {}, "briefing": "", "ready": False, "_fus_ts": ""})

# ──────────────────────────────────────────────────────────────────────────────
#  Cached model — loaded ONCE for the lifetime of the server process
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def _get_yolo() -> YOLO:
    model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    return YOLO(model_path)


@st.cache_resource(show_spinner="Loading YOLOv8n-VisDrone aerial model…")
def _get_visdrone_model():
    """
    YOLOv8n fine-tuned on VisDrone dataset (10,000+ drone images).
    Classes: pedestrian · people · bicycle · car · van · truck
             tricycle · awning-tricycle · bus · motor
    Dramatically better than COCO for aerial/drone footage perspective.
    Weights: mshamrai/yolov8n-visdrone on HuggingFace (6.2 MB).
    """
    local = os.path.join(os.path.dirname(__file__), "yolov8n-visdrone.pt")
    if os.path.exists(local):
        try:
            return YOLO(local)
        except Exception:
            pass
    # Auto-download from HuggingFace if not present
    try:
        import requests
        url  = "https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/best.pt"
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 200:
            with open(local, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return YOLO(local)
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner="Loading YOLOv8-OBB aerial detection model (first run — ~6 MB download)…")
def _get_obb_model():
    """
    YOLOv8n-OBB trained on DOTA dataset (aerial / satellite objects).
    Detects: plane, ship, storage-tank, harbor, bridge, large/small vehicle,
    helicopter — with proper Oriented Bounding Boxes (rotated polygons).
    """
    try:
        return YOLO("yolov8n-obb.pt")   # auto-downloads DOTA weights
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading SegFormer semantic segmentation model (first run — ~14 MB download)…")
def _get_segformer():
    """
    SegFormer-B0 fine-tuned on ADE20K-150.
    Detects buildings, roads, vegetation, water, bare-ground, etc.
    This is the RIGHT model for satellite / aerial image analysis.
    Returns (model, processor) tuple, or (None, None) if unavailable.
    """
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        name = "nvidia/segformer-b0-finetuned-ade-512-512"
        processor = SegformerImageProcessor.from_pretrained(name)
        model     = SegformerForSemanticSegmentation.from_pretrained(name)
        model.eval()
        return model, processor
    except Exception:
        return None, None

# ──────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: #000 !important;
    color: #d4d4d4;
}
.stApp { background: #000 !important; }
.main .block-container {
    background: #000 !important;
    padding-top: 1.5rem;
}

/* ── Header ── */
.geointel-header {
    background: #000;
    border-bottom: 1px solid #1c1c1c;
    padding: 0;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}
.geointel-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.022) 39px, rgba(255,255,255,0.022) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 79px, rgba(255,255,255,0.018) 79px, rgba(255,255,255,0.018) 80px);
    pointer-events: none;
    z-index: 2;
}
.geointel-header > * { position: relative; z-index: 1; }
.geointel-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 5px;
    margin: 0;
    text-transform: uppercase;
    text-shadow: 0 0 24px rgba(255,255,255,0.18);
}
.geointel-subtitle {
    font-size: 0.68rem;
    color: #aaa;
    letter-spacing: 3px;
    margin-top: 0.3rem;
    text-transform: uppercase;
}

/* ── Section headers ── */
.section-hdr {
    font-size: 0.7rem;
    font-weight: 600;
    color: #fff;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #1c1c1c;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Threat banner ── */
.threat-banner {
    border-radius: 2px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 0.8rem 0 1rem 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    border-width: 1px;
    border-style: solid;
}
.threat-text {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* ── Info cards ── */
.info-card {
    background: #0a0a0a;
    border: 1px solid #1c1c1c;
    border-radius: 2px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    line-height: 1.8;
    font-size: 0.86rem;
    color: #bbb;
}
.info-card b { color: #eee; }

/* ── Recommendation items ── */
.rec-item {
    background: #0a0a0a;
    border-left: 2px solid #fff;
    padding: 0.65rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.86rem;
    color: #bbb;
    line-height: 1.6;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #1c1c1c;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 10px 28px;
    font-size: 0.75rem;
    font-weight: 500;
    background: transparent;
    color: #999;
    border: none;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #fff !important;
    border-bottom: 2px solid #fff !important;
    text-shadow: 0 0 16px rgba(255,255,255,0.3);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0a0a0a;
    border: 1px dashed #222;
    border-radius: 2px;
    padding: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid #111;
}
[data-testid="stSidebar"] .stMarkdown { color: #aaa; }
[data-testid="stSidebar"] h3 {
    color: #fff;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0a0a0a;
    border: 1px solid #1c1c1c;
    border-radius: 2px;
    padding: 0.75rem 0.9rem;
    animation: cipher-fadein 0.55s ease-out both;
}
@keyframes cipher-fadein {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes cipher-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.7; transform: scale(1.22); }
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.68rem !important;
    color: #aaa !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
}
[data-testid="stMetricValue"] > div { color: #fff !important; font-weight: 600; }
[data-testid="stMetricDelta"] > div { font-size: 0.75rem !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 0 20px rgba(255,255,255,0.08);
}
.stButton > button[kind="primary"]:hover {
    background: #e8e8e8 !important;
    box-shadow: 0 0 28px rgba(255,255,255,0.15) !important;
}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #bbb !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #555 !important;
    color: #fff !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #bbb !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #555 !important;
    color: #fff !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    border-width: 1px !important;
    border-style: solid !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1c1c1c; border-radius: 2px; }
iframe[title="st_aggrid.agGrid"] { background: #0a0a0a; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: #fff !important; }

/* ── Divider ── */
hr { border-color: #111 !important; margin: 1.2rem 0 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #fff; }

/* ── Caption ── */
.stCaption { color: #999 !important; font-size: 0.75rem !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #fff !important; }

/* ── Chat interface ── */
[data-testid="stChatMessage"] {
    background: #080808 !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 2px !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stChatMessageContent"] p { color: #ccc !important; line-height: 1.7 !important; }
[data-testid="stChatInput"] > div {
    background: #050505 !important;
    border-top: 1px solid #1c1c1c !important;
}
[data-testid="stChatInputTextArea"] {
    background: #0a0a0a !important;
    color: #ddd !important;
    border-color: #2a2a2a !important;
    border-radius: 2px !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: #555 !important; }

/* ── Hover effects ── */
[data-testid="metric-container"] {
    transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: #383838;
    background: #0f0f0f;
    box-shadow: 0 0 20px rgba(255,255,255,0.05);
    transform: translateY(-2px);
}

.info-card {
    transition: border-color 0.2s ease, background 0.2s ease;
}
.info-card:hover {
    border-color: #2e2e2e;
    background: #0f0f0f;
}

.rec-item {
    transition: border-left-color 0.2s ease, background 0.2s ease, color 0.2s ease;
}
.rec-item:hover {
    border-left-color: #fff;
    background: #0f0f0f;
    color: #ccc;
}

.cipher-strip-cell {
    text-align: center;
    padding: 0.35rem 1.4rem;
    border-radius: 2px;
    cursor: default;
    transition: background 0.2s ease, transform 0.2s ease;
}
.cipher-strip-cell:hover {
    background: rgba(255,255,255,0.04);
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #444;
    background: #0f0f0f;
}

[data-testid="stDataFrame"] {
    transition: border-color 0.2s ease;
}
[data-testid="stDataFrame"]:hover {
    border-color: #2a2a2a;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Session State Initialisation
# ──────────────────────────────────────────────────────────────────────────────

_STATE_KEYS = ["sat_results", "drn_results", "fusion_results", "sat_analyzer", "drn_analyzer"]
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None
st.session_state.setdefault("sat_show_annotated", True)
st.session_state.setdefault("sat_original_image", None)


def _reset():
    for k in _STATE_KEYS:
        st.session_state[k] = None
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  Plotly theme helper
# ──────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,10,0.95)",
    font=dict(color="#bbb", size=11, family="Inter, Segoe UI, sans-serif"),
    margin=dict(l=8, r=8, t=28, b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#bbb")),
    xaxis=dict(gridcolor="#1c1c1c", linecolor="#2a2a2a", tickcolor="#555"),
    yaxis=dict(gridcolor="#1c1c1c", linecolor="#2a2a2a", tickcolor="#555"),
)


def _style_conf(df: pd.DataFrame, col: str) -> "pd.io.formats.style.Styler":
    """Color-code a confidence percentage column: green / amber / red."""
    def _color(val: str) -> str:
        try:
            pct = float(str(val).replace("%", "").strip()) / 100
        except ValueError:
            return ""
        if pct >= 0.75:
            return "background-color:rgba(34,197,94,0.12);color:#22c55e;font-weight:500"
        if pct >= 0.50:
            return "background-color:rgba(245,158,11,0.12);color:#f59e0b;font-weight:500"
        return "background-color:rgba(239,68,68,0.12);color:#ef4444;font-weight:500"
    return df.style.map(_color, subset=[col])


def _empty_state(step: str, title: str, body: str, note: str = "") -> None:
    """Render a centered onboarding card for tabs with no data yet."""
    note_html = (
        f'<div style="font-size:0.68rem;color:#666;letter-spacing:3px;'
        f'margin-top:1.4rem;text-transform:uppercase;">{note}</div>'
    ) if note else ""
    st.markdown(
        f"""
<div style="border:1px dashed #2a2a2a;border-radius:2px;padding:3.5rem 2rem;
  text-align:center;margin:2rem 0 1rem 0;">
  <div style="font-size:0.62rem;letter-spacing:5px;color:#666;
    text-transform:uppercase;margin-bottom:1rem;">{step}</div>
  <div style="font-size:1rem;font-weight:600;color:#fff;letter-spacing:3px;
    text-transform:uppercase;margin-bottom:1rem;">{title}</div>
  <div style="font-size:0.83rem;color:#999;max-width:460px;margin:0 auto;
    line-height:1.8;">{body}</div>
  {note_html}
</div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Intelligence Assistant helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_chat_context(sat: dict, drn: dict, fus: dict) -> str:
    sat_objs = ", ".join(
        f"{k} ×{v['count']}" for k, v in sat.get("detected_objects", {}).items()
    ) or "none"
    drn_objs = ", ".join(
        f"{k} ×{v['count']}" for k, v in drn.get("detected_objects", {}).items()
    ) or "none"
    land = ", ".join(
        f"{k} {v:.0f}%" for k, v in sat.get("land_classification", {}).items() if v > 0
    ) or "unknown"
    alerts = "; ".join(drn.get("alerts", [])) or "none"
    recs   = "\n".join(f"  - {r}" for r in fus.get("recommendations", []))
    return f"""You are the CIPHER Intelligence Assistant — a friendly analyst helping non-technical users understand geospatial intelligence reports.

CURRENT ANALYSIS DATA:
Satellite:
  Scene: {sat.get('scene_type','—')} | Dominant land: {sat.get('dominant_land','—')}
  Objects detected: {sat_objs}
  Land cover: {land}
  Vegetation health: {sat.get('veg_health',{}).get('status','—')}
  Fire: {'YES' if sat.get('features',{}).get('fire_detected') else 'No'} | Smoke: {'YES' if sat.get('features',{}).get('smoke_detected') else 'No'}

Drone:
  Video: {drn.get('video_duration',0):.0f}s | Unique tracks: {drn.get('total_tracks',0)} | Fast movers: {drn.get('fast_movers',0)}
  Objects: {drn_objs}
  Fire frames: {drn.get('fire_pct_frames',0):.1f}% | Smoke: {drn.get('smoke_pct_frames',0):.1f}%
  Loitering: {drn.get('loitering_count',0)} tracks ({drn.get('loitering_people',0)} people)
  Alerts: {alerts}

Fusion result:
  Threat level: {fus.get('threat_level','—')}
  Activity score: {fus.get('activity_score',0):.0%} | Fusion score: {fus.get('fusion_score',0):.0f}/100
  Total objects fused: {fus.get('total_objects_detected',0)} | Movement ratio: {fus.get('movement_ratio',0):.0%}
  Source agreement: {fus.get('agreement_rate',0):.0%}
  Summary: {fus.get('summary','—')}

Recommendations:
{recs}

STRICT SCOPE RULE:
You are ONLY allowed to answer questions directly related to this specific CIPHER analysis session — the satellite image, drone video, detected objects, threat level, land classification, alerts, fusion results, and recommendations shown above.

If the user asks ANYTHING outside this scope (general knowledge, news, coding, weather, personal advice, other topics, or hypothetical scenarios unrelated to this analysis), you must politely refuse with a short message like:
  "I can only help with questions about this specific CIPHER analysis. Try asking about the threat level, detected objects, or recommendations."

Do NOT answer off-topic questions even if asked nicely or rephrased. Stay strictly within the analysis data above.

ANSWER INSTRUCTIONS:
- Always answer in plain, everyday language — no jargon or acronyms
- Use simple analogies when helpful (e.g. "like a security camera counting people")
- Keep answers concise: 2-5 sentences for most questions
- Reference actual numbers from the data above when relevant
- Never invent data not present in the analysis
- Be reassuring and helpful in tone unless the threat level is genuinely HIGH"""


# ──────────────────────────────────────────────────────────────────────────────
#  Annotated satellite image helper
# ──────────────────────────────────────────────────────────────────────────────

def _annotate_sat_clean(pil_img: Image, all_detections: list) -> Image:
    """Draw clean bounding boxes + labels on a copy of the satellite image."""
    import colorsys as _cs
    from PIL import ImageDraw as _ID
    img  = pil_img.copy().convert("RGB")
    draw = _ID.Draw(img)
    cats = list(dict.fromkeys(
        d.get("category", d.get("class", "object")) for d in all_detections
    ))
    palette: dict = {}
    for i, cat in enumerate(cats):
        r, g, b = [int(x * 255) for x in _cs.hsv_to_rgb(i / max(len(cats), 1), 0.85, 0.95)]
        palette[cat] = (r, g, b)
    for det in all_detections:
        bbox = det.get("bbox") or det.get("xyxy")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cat   = det.get("category", det.get("class", "object"))
        conf  = det.get("confidence", 0)
        color = palette.get(cat, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cat} {conf:.0%}"
        lw    = len(label) * 6 + 4
        lh    = 15
        ty    = max(0, y1 - lh)
        draw.rectangle([x1, ty, x1 + lw, ty + lh], fill=color)
        draw.text((x1 + 2, ty + 1), label, fill=(0, 0, 0))
    return img


# ──────────────────────────────────────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────────────────────────────────────

_logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
_logo_uri  = ""
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _lf:
        _logo_uri = "data:image/png;base64," + _b64_top.b64encode(_lf.read()).decode()

if _logo_uri:
    st.markdown(
        f'<div class="geointel-header"><img src="{_logo_uri}" '
        f'style="width:100%;max-height:220px;object-fit:cover;'
        f'object-position:center;display:block;" /></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="geointel-header">
    <div class="geointel-title">CIPHER</div>
    <div class="geointel-subtitle">
        Combined Intelligence Platform for High-resolution Environmental Recon
        &nbsp;&mdash;&nbsp; Satellite &amp; UAV Drone Fusion Analysis
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")

    # ── Detection Confidence ──────────────────────────────────────────
    confidence = st.slider(
        "Detection Confidence", 0.10, 0.90, 0.25, 0.05,
    )
    with st.expander("What does this do?"):
        st.markdown(
            '<div class="info-card" style="margin-bottom:0;">'
            "<b>Detection Confidence</b> sets the minimum probability score "
            "a YOLO detection must reach before it is accepted.<br><br>"
            "<b>Low (0.10–0.20)</b><br>"
            "Detects more objects — useful for distant or partially occluded targets. "
            "Increases false positives (shadows, textures flagged as objects).<br><br>"
            "<b>High (0.60–0.90)</b><br>"
            "Only accepts highly certain detections. Misses small or partially "
            "visible objects but almost no false positives.<br><br>"
            "<b>Suggested:</b> <span style='color:#eee;'>0.25 – 0.35</span><br>"
            "<b>Best for satellite:</b> <span style='color:#eee;'>0.30</span> "
            "(balanced for high-res imagery)<br>"
            "<b>Best for drone:</b> <span style='color:#eee;'>0.25</span> "
            "(aerial perspective reduces contrast)"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── IoU Threshold ─────────────────────────────────────────────────
    iou_thresh = st.slider(
        "IoU Threshold (NMS)", 0.10, 0.90, 0.45, 0.05,
    )
    with st.expander("What does this do?"):
        st.markdown(
            '<div class="info-card" style="margin-bottom:0;">'
            "<b>IoU Threshold</b> controls Non-Maximum Suppression (NMS). "
            "When two bounding boxes overlap by more than this ratio, "
            "the weaker one is removed.<br><br>"
            "<b>Low (0.10–0.30)</b><br>"
            "Aggressively removes overlapping boxes — good for avoiding "
            "duplicate detections. May merge nearby distinct objects.<br><br>"
            "<b>High (0.60–0.90)</b><br>"
            "Keeps overlapping boxes — useful for tightly packed objects "
            "like parking lots or crowds. Risk of duplicate detections.<br><br>"
            "<b>Suggested:</b> <span style='color:#eee;'>0.40 – 0.50</span><br>"
            "<b>Best for satellite:</b> <span style='color:#eee;'>0.45</span> "
            "(standard default, works well for spread-out objects)<br>"
            "<b>Best for dense scenes:</b> <span style='color:#eee;'>0.30</span> "
            "(crowded areas, tight vehicle clusters)"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Drone Frame Sampling ──────────────────────────────────────────
    frame_skip = st.slider(
        "Drone Frame Sampling", 1, 8, 2,
    )
    with st.expander("What does this do?"):
        st.markdown(
            '<div class="info-card" style="margin-bottom:0;">'
            "<b>Frame Sampling</b> sets how many frames are skipped between "
            "each analysis pass. A value of 2 means every 2nd frame is processed.<br><br>"
            "<b>1 (every frame)</b><br>"
            "Maximum accuracy — no motion missed. "
            "Significantly slower; recommended only for short clips (under 30 s).<br><br>"
            "<b>3 – 5 (skip frames)</b><br>"
            "Fast overview scan. May miss fast-moving objects between sampled frames. "
            "Good for long videos where speed matters.<br><br>"
            "<b>Suggested:</b> <span style='color:#eee;'>2</span> "
            "(best speed / accuracy balance)<br>"
            "<b>Critical analysis:</b> <span style='color:#eee;'>1</span> — miss nothing<br>"
            "<b>Quick scan:</b> <span style='color:#eee;'>4 – 6</span> — rapid overview"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Pipeline Status")
    sat_ok  = st.session_state.sat_results  is not None
    drn_ok  = st.session_state.drn_results  is not None
    fus_ok  = st.session_state.fusion_results is not None

    def _status(label, ready):
        dot = "●" if ready else "○"
        color = "#22c55e" if ready else "#333"
        text  = "Ready" if ready else "Pending"
        st.markdown(
            f'<span style="color:{color};font-size:0.8rem;">{dot}</span>'
            f'<span style="color:#aaa;font-size:0.8rem;margin-left:6px;">'
            f'{label}</span>'
            f'<span style="color:{"#22c55e" if ready else "#666"};font-size:0.75rem;'
            f'margin-left:6px;">{text}</span>',
            unsafe_allow_html=True,
        )

    _status("Satellite", sat_ok)
    _status("Drone", drn_ok)
    _status("Fusion", fus_ok)
    st.markdown("")

    if st.button("Reset Analysis", use_container_width=True):
        _reset()

    st.divider()
    st.markdown("### About")
    st.markdown(
        """
CIPHER fuses satellite imagery with drone video
using deep learning to produce unified intelligence.

**Satellite**
SegFormer (ADE20K-150) — land classification
YOLOv8 tiled detection — vehicles, people, aircraft

**Drone**
YOLOv8n-VisDrone — aerial-trained detection
COCO supplement — wildlife, aircraft, suspicious
Fire / smoke / loitering / fast-mover alerts

**Fusion**
Probabilistic confidence merge
Unified threat assessment + report

*Major Project — Geospatial AI*
"""
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Stat Strip  (persistent ticker bar between header and tabs)
# ──────────────────────────────────────────────────────────────────────────────

_sf = st.session_state.fusion_results
_ss_objects  = str(_sf["total_objects_detected"])      if _sf else "—"
_ss_threat   = _sf.get("threat_level", "—")            if _sf else "—"
_ss_score    = f"{_sf['fusion_score']:.0f}"            if _sf else "—"
_ss_tracks   = str(_sf.get("total_tracks", "—"))       if _sf else "—"
_ss_activity = f"{_sf['activity_score']:.0%}"          if _sf else "—"
_ss_land     = _sf.get("dominant_land", "—")           if _sf else "—"

_tc = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#22c55e"}.get(_ss_threat, "#fff")

def _sc(label: str, value: str, color: str = "#fff") -> str:
    return (
        f'<div class="cipher-strip-cell">'
        f'<div style="font-size:0.52rem;letter-spacing:3px;color:#999;'
        f'text-transform:uppercase;margin-bottom:0.22rem;">{label}</div>'
        f'<div style="font-size:1rem;font-weight:600;color:{color};'
        f'letter-spacing:1px;">{value}</div>'
        f'</div>'
    )

_bar = '<div style="width:1px;background:#1c1c1c;height:1.8rem;align-self:center;flex-shrink:0;"></div>'

st.markdown(
    '<div style="display:flex;align-items:center;justify-content:center;'
    'border-top:1px solid #0e0e0e;border-bottom:1px solid #0e0e0e;'
    'background:#020202;padding:0.6rem 0;margin-bottom:1.4rem;">'
    + _sc("Total Objects", _ss_objects) + _bar
    + _sc("Threat Level", _ss_threat, _tc) + _bar
    + _sc("Fusion Score", f"{_ss_score}/100" if _ss_score != "—" else "—") + _bar
    + _sc("Active Tracks", _ss_tracks) + _bar
    + _sc("Activity", _ss_activity) + _bar
    + _sc("Dominant Land", _ss_land)
    + '</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_sat, tab_drn, tab_fusion, tab_report = st.tabs(
    ["Satellite", "Drone", "Fusion", "Report"]
)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SATELLITE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sat:
    col_up, col_tip = st.columns([2, 1])

    with col_up:
        st.markdown('<div class="section-hdr">Upload Satellite / Aerial Image</div>', unsafe_allow_html=True)
        sat_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="sat_file_uploader",
            label_visibility="collapsed",
        )

    with col_tip:
        st.markdown('<div class="section-hdr">Tips</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-card">'
            "Accepted: JPEG · PNG · GeoTIFF<br>"
            "Best results with high-resolution images (≥1024 px)<br>"
            "Minimal cloud cover recommended<br>"
            "Works with Google Earth exports, Sentinel, Landsat"
            "</div>",
            unsafe_allow_html=True,
        )

    if not sat_file and not st.session_state.sat_results:
        _empty_state(
            step="Step 01 of 03",
            title="Upload a Satellite Image",
            body=(
                "Drag a satellite or aerial photograph into the uploader above. "
                "CIPHER will detect vehicles, buildings, roads, vegetation, "
                "water bodies, and produce a full land-use classification."
            ),
            note="JPEG &middot; PNG &middot; GeoTIFF &middot; Sentinel &middot; Landsat",
        )

    if sat_file:
        img_bytes = sat_file.read()
        orig_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.session_state.sat_original_image = orig_pil

        prev_col, meta_col = st.columns([3, 1])
        with prev_col:
            st.image(orig_pil, caption="Uploaded Satellite Image", use_container_width=True)
        with meta_col:
            st.markdown("**Image Info**")
            st.write(f"Dimensions: `{orig_pil.width} × {orig_pil.height} px`")
            st.write(f"File size: `{len(img_bytes)/1024:.1f} KB`")

        # Model status badges
        _seg_m, _ = _get_segformer()
        _obb_m    = _get_obb_model()
        b1, b2 = st.columns(2)
        with b1:
            if _seg_m is not None:
                st.success("SegFormer — buildings, roads, vegetation, water (ADE20K)")
            else:
                st.warning("SegFormer unavailable — HSV terrain classifier fallback")
        with b2:
            if _obb_m is not None:
                st.success("YOLOv8-OBB — planes, ships, vehicles, bridges (DOTA)")
            else:
                st.warning("OBB model unavailable — standard boxes only")

        if st.button(
            "Analyse Satellite Image",
            type="primary",
            use_container_width=True,
            key="btn_sat_analyse",
        ):
            yolo_model            = _get_yolo()
            seg_model, seg_proc   = _get_segformer()
            obb_model             = _get_obb_model()
            if st.session_state.sat_analyzer is None:
                st.session_state.sat_analyzer = SatelliteAnalyzer(
                    confidence, iou_thresh,
                    model=yolo_model,
                    seg_model=seg_model,
                    seg_processor=seg_proc,
                    obb_model=obb_model,
                )
            ana = st.session_state.sat_analyzer
            ana.confidence    = confidence
            ana.iou           = iou_thresh
            ana.seg_model     = seg_model
            ana.seg_processor = seg_proc
            ana.obb_model     = obb_model

            pb = st.progress(0.0)
            st_txt = st.empty()

            def _sat_cb(pct, msg):
                pb.progress(float(pct))
                st_txt.caption(msg)

            sat_res = ana.analyze(img_bytes, progress_cb=_sat_cb)
            st.session_state.sat_results = sat_res
            st.session_state.fusion_results = None  # invalidate previous fusion
            pb.empty()
            st_txt.empty()
            st.success("Satellite analysis complete.")

    # ── Results ────────────────────────────────────────────────────────
    if st.session_state.sat_results:
        res = st.session_state.sat_results
        st.divider()
        st.markdown('<div class="section-hdr">Analysis Results</div>', unsafe_allow_html=True)

        # Model badges
        seg_used = res.get("segmentation_used", False)
        obb_used = res.get("obb_used", False)
        bb1, bb2 = st.columns(2)
        with bb1:
            if seg_used:
                st.success("SegFormer — building footprints, roads, vegetation, water")
            else:
                st.info("HSV terrain classifier (SegFormer unavailable)")
        with bb2:
            obb_n = res.get("features", {}).get("obb_detections", 0)
            if obb_used:
                st.success(f"OBB detection active — {obb_n} oriented objects found")
            else:
                st.warning("OBB model not loaded (standard boxes only)")

        # Metrics row — primary
        feat = res.get("features", {})
        mc = st.columns(4)
        mc[0].metric("Vehicles / People",
                     res["total_objects"],
                     help="YOLO tiled detection (aerial classes only)")
        mc[1].metric("Building Footprints",
                     feat.get("building_footprints", feat.get("estimated_structures", "—")),
                     help="SegFormer building mask → polygon contours" if seg_used else "CV blob analysis")
        mc[2].metric("Road Coverage",
                     f"{feat.get('road_coverage_pct', 0):.1f}%",
                     help="SegFormer road class pixels" if seg_used else "HSV grey-pixel estimate")
        mc[3].metric("Green Zones",
                     feat.get("green_zones", "—"),
                     help="Distinct large vegetation areas")

        # Metrics row — secondary
        mc2 = st.columns(4)
        mc2[0].metric("Water Bodies",
                      feat.get("water_bodies", "—"),
                      help=f"Coverage: {feat.get('water_coverage_pct', 0):.1f}%")
        mc2[1].metric("Parking Lots",    feat.get("parking_lots", "—"))
        mc2[2].metric("Scene Type",      res["scene_type"])
        mc2[3].metric("Image Size",
                      f"{res['image_size'][0]}×{res['image_size'][1]} px")

        # Metrics row — environment & land
        mc3 = st.columns(4)
        mc3[0].metric("Dominant Land",   res["dominant_land"])
        mc3[1].metric("Open Areas",      feat.get("open_areas", "—"))

        fire_val    = "YES" if feat.get("fire_detected")  else "No"
        smoke_val   = "YES" if feat.get("smoke_detected") else "No"
        fire_delta  = f"{feat.get('fire_coverage_pct', 0):.2f}% area" if feat.get("fire_detected") else None
        smoke_delta = f"{feat.get('smoke_coverage_pct', 0):.2f}% area" if feat.get("smoke_detected") else None
        mc3[2].metric("Fire Alert",  fire_val,  delta=fire_delta,
                      delta_color="inverse" if feat.get("fire_detected") else "off")
        mc3[3].metric("Smoke Alert", smoke_val, delta=smoke_delta,
                      delta_color="inverse" if feat.get("smoke_detected") else "off")

        # Metrics row — advanced environmental
        mc4 = st.columns(4)
        veg_h = res.get("veg_health", {})
        mc4[0].metric("Vegetation Health",
                      veg_h.get("status", "—"),
                      delta=f"ExG {veg_h.get('health_index', 0):.3f}  •  {veg_h.get('healthy_pct', 0):.0f}% healthy")
        burn_n = feat.get("burn_scars", 0)
        mc4[1].metric("Burn Scars",
                      f"{burn_n} area(s)" if burn_n else "None",
                      delta=f"{feat.get('burn_scar_pct', 0):.2f}% coverage" if burn_n else None,
                      delta_color="inverse" if burn_n else "off")
        sol_n = feat.get("solar_panel_regions", 0)
        mc4[2].metric("Solar Panel Arrays",
                      sol_n if sol_n else "None",
                      delta=f"{feat.get('solar_coverage_pct', 0):.2f}% coverage" if sol_n else None)
        ci = res.get("cloud_info", {})
        mc4[3].metric("Usable Area",
                      f"{feat.get('usable_area_pct', 100):.1f}%",
                      delta=f"Cloud {ci.get('cloud_pct',0):.1f}%  Shadow {ci.get('shadow_pct',0):.1f}%")

        # Images — before/after toggle
        _img_modes   = ["Detections", "Land Classification", "Original"]
        _img_default = _img_modes.index(
            st.session_state.get("sat_img_mode", "Detections")
        )
        _img_col, _lbl_col = st.columns([3, 1])
        with _lbl_col:
            st.markdown("<br>", unsafe_allow_html=True)
            _chosen = st.radio(
                "View",
                _img_modes,
                index=_img_default,
                key="sat_img_mode",
                label_visibility="collapsed",
            )
        with _img_col:
            if _chosen == "Detections":
                st.markdown("**Object Detection Map**")
                st.image(res["annotated_image"], use_container_width=True)
            elif _chosen == "Land Classification":
                st.markdown("**Land Classification Overlay**")
                st.image(res["land_overlay"], use_container_width=True)
            else:
                st.markdown("**Original Image**")
                _orig = st.session_state.get("sat_original_image")
                if _orig is not None:
                    st.image(_orig, use_container_width=True)
                else:
                    st.info("Original image not available — re-upload the file.")

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Terrain Composition**")
            land_df = pd.DataFrame(
                [(k, v) for k, v in res["land_classification"].items() if v > 0],
                columns=["Type", "Pct"],
            )
            cmap = {
                "Vegetation":    "#27ae60",
                "Agriculture":   "#6ab04c",
                "Water":         "#2980b9",
                "Recreation":    "#00cec9",
                "Buildings":     "#e17055",
                "Roads":         "#636e72",
                "Airport":       "#fdcb6e",
                "Infrastructure":"#f39c12",
                "Vehicles":      "#00b894",
                "Aircraft":      "#6c5ce7",
                "Watercraft":    "#0984e3",
                "Urban":         "#95a5a6",
                "Bare Ground":   "#c0933a",
                "Snow/Clouds":   "#dfe6e9",
                "Other":         "#2d3436",
            }
            fig_pie = px.pie(
                land_df, values="Pct", names="Type",
                color="Type", color_discrete_map=cmap,
                hole=0.42,
            )
            fig_pie.update_traces(textfont_color="white")
            fig_pie.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            st.markdown("**Detected Objects**")
            if res["detected_objects"]:
                obj_rows = [
                    {
                        "Object":         cls,
                        "Count":          d["count"],
                        "Avg Confidence": f"{d['avg_confidence']:.1%}",
                    }
                    for cls, d in sorted(res["detected_objects"].items(), key=lambda x: -x[1]["count"])
                ]
                st.dataframe(
                    _style_conf(pd.DataFrame(obj_rows), "Avg Confidence"),
                    use_container_width=True,
                    hide_index=True,
                )
                # Bar chart
                fig_bar = px.bar(
                    pd.DataFrame(obj_rows),
                    x="Object", y="Count",
                    color="Object",
                    text="Count",
                )
                fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                fig_bar.update_traces(textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info(
                    "No objects detected at current confidence. "
                    "Try lowering the threshold in the sidebar."
                )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DRONE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_drn:
    col_up2, col_tip2 = st.columns([2, 1])

    with col_up2:
        st.markdown('<div class="section-hdr">Upload Drone / UAV Video</div>', unsafe_allow_html=True)
        drn_file = st.file_uploader(
            "Drag & drop or browse",
            type=["mp4", "avi", "mov", "mkv"],
            key="drn_file_uploader",
            label_visibility="collapsed",
        )

    with col_tip2:
        st.markdown('<div class="section-hdr">Capabilities</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-card">'
            "35+ class detection — people, vehicles,<br>"
            "&nbsp;&nbsp;aircraft, wildlife, suspicious objects<br>"
            "ByteTrack unique IDs + motion trails<br>"
            "Fire &amp; smoke hazard detection<br>"
            "Loitering and fast-mover alerts<br>"
            "Scene zone overlay (vegetation, water)<br>"
            "Live HUD panel + annotated video output<br><br>"
            "Formats: MP4 &middot; AVI &middot; MOV &middot; MKV<br>"
            "Lower frame-skip = higher tracking accuracy"
            "</div>",
            unsafe_allow_html=True,
        )

    if not drn_file and not st.session_state.drn_results:
        _empty_state(
            step="Step 02 of 03",
            title="Upload a Drone Video",
            body=(
                "Drag a UAV or drone video into the uploader above. "
                "CIPHER will track every moving object with unique IDs, "
                "detect fire and smoke hazards, flag loitering individuals, "
                "and produce a fully annotated output video."
            ),
            note="MP4 &middot; AVI &middot; MOV &middot; MKV",
        )

    if drn_file:
        vid_bytes = drn_file.read()
        st.markdown("**Original Video Preview**")
        st.video(vid_bytes)
        size_mb = len(vid_bytes) / 1_048_576
        st.caption(f"File size: {size_mb:.2f} MB")

        # Model status badges
        _vd = _get_visdrone_model()
        mb1, mb2 = st.columns(2)
        with mb1:
            if _vd is not None:
                st.success("VisDrone model — aerial-trained (10 classes, drone perspective)")
            else:
                st.warning("VisDrone model not found — using COCO fallback")
        with mb2:
            st.info("COCO supplement — wildlife, aircraft, suspicious objects")

        if st.button(
            "Analyse Drone Video",
            type="primary",
            use_container_width=True,
            key="btn_drn_analyse",
        ):
            yolo_model = _get_yolo()
            vd_model   = _get_visdrone_model()
            if st.session_state.drn_analyzer is None:
                st.session_state.drn_analyzer = DroneAnalyzer(
                    confidence, iou_thresh, frame_skip,
                    model=yolo_model, visdrone_model=vd_model,
                )
            ana2 = st.session_state.drn_analyzer
            ana2.confidence     = confidence
            ana2.iou            = iou_thresh
            ana2.frame_skip     = frame_skip
            ana2.visdrone_model = vd_model

            pb2   = st.progress(0.0)
            st_t2 = st.empty()

            def _drn_cb(pct, msg):
                pb2.progress(float(pct))
                st_t2.caption(msg)

            drn_res = ana2.analyze(vid_bytes, progress_cb=_drn_cb)
            st.session_state.drn_results = drn_res
            st.session_state.fusion_results = None
            pb2.empty()
            st_t2.empty()
            st.success("Drone analysis complete.")

    # ── Results ────────────────────────────────────────────────────────
    if st.session_state.drn_results:
        res2 = st.session_state.drn_results
        st.divider()
        st.markdown('<div class="section-hdr">Analysis Results</div>', unsafe_allow_html=True)

        # Active model badge
        if res2.get("visdrone_active"):
            st.success(
                "VisDrone + COCO dual-model — aerial-trained primary (people/vehicles) "
                "+ COCO supplement (wildlife/aircraft/suspicious objects)"
            )
        else:
            st.info("COCO single-model — VisDrone weights not found at analysis time")

        # Row 1 — tracking / motion
        dm1 = st.columns(5)
        dm1[0].metric("Unique Tracks",   res2["total_tracks"])
        dm1[1].metric("Fast Movers",     res2["fast_movers"])
        dm1[2].metric("Object Classes",  len(res2["detected_objects"]))
        dm1[3].metric("Duration",        f"{res2['video_duration']:.1f} s")
        dm1[4].metric("Avg Speed",       f"{res2['avg_track_speed']:.1f} px/f")

        # Row 2 — environmental hazards + scene + loitering
        dm2 = st.columns(4)
        _fire_pct  = res2.get("fire_pct_frames", 0)
        _smoke_pct = res2.get("smoke_pct_frames", 0)
        _loiter    = res2.get("loitering_count", 0)
        _loiter_p  = res2.get("loitering_people", 0)
        dm2[0].metric(
            "Fire Alert",
            f"{_fire_pct:.0f}% of frames" if _fire_pct > 0 else "None",
            delta=f"{res2.get('fire_frames', 0)} frames" if _fire_pct > 0 else None,
            delta_color="inverse" if _fire_pct > 0 else "off",
        )
        dm2[1].metric(
            "Smoke Alert",
            f"{_smoke_pct:.0f}% of frames" if _smoke_pct > 0 else "None",
            delta_color="off",
        )
        dm2[2].metric(
            "Loitering Alerts",
            f"{_loiter} track(s)" if _loiter else "None",
            delta=f"{_loiter_p} person(s)" if _loiter_p else None,
            delta_color="inverse" if _loiter_p else "off",
        )
        dm2[3].metric("Scene Type", res2.get("dominant_scene", "—"))

        # Alerts panel
        _alerts = res2.get("alerts", [])
        if _alerts:
            st.error("**Security / Behavioral Alerts**\n\n" +
                     "\n".join(f"• {a}" for a in _alerts))

        # Annotated video
        out_vp = res2.get("output_video_path", "")
        if out_vp and os.path.exists(out_vp):
            st.markdown("**Annotated Detection Video**")
            with open(out_vp, "rb") as fv:
                out_bytes = fv.read()
            st.video(out_bytes)
            st.download_button(
                "Download Annotated Video",
                data=out_bytes,
                file_name="cipher_drone_annotated.mp4",
                mime="video/mp4",
            )

        # Sample frames
        if res2["sample_frames"]:
            st.markdown("**Sample Detection Frames**")
            n = len(res2["sample_frames"])
            frame_cols = st.columns(n)
            for i, (fc, frm) in enumerate(zip(frame_cols, res2["sample_frames"])):
                fc.image(frm, caption=f"Frame {i+1}", use_container_width=True)

        # Stats + timeline
        stats_col, tl_col = st.columns([1, 2])

        with stats_col:
            if res2["detected_objects"]:
                st.markdown("**Object Statistics**")
                obj_rows2 = [
                    {
                        "Object":   cls,
                        "Total":    d["count"],
                        "Peak":     res2["peak_counts"].get(cls, 0),
                        "Avg Conf": f"{d['avg_confidence']:.1%}",
                    }
                    for cls, d in sorted(
                        res2["detected_objects"].items(), key=lambda x: -x[1]["count"]
                    )
                ]
                st.dataframe(
                    _style_conf(pd.DataFrame(obj_rows2), "Avg Conf"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No objects detected at current confidence threshold. "
                    "Try lowering it in the sidebar."
                )

        with tl_col:
            if res2["frame_timeline"]:
                st.markdown("**Object Count Timeline**")
                df_tl = pd.DataFrame(res2["frame_timeline"]).fillna(0)
                fig_tl = go.Figure()
                for col in df_tl.columns:
                    if col != "time":
                        fig_tl.add_trace(
                            go.Scatter(
                                x=df_tl["time"], y=df_tl[col],
                                name=col, mode="lines",
                                fill="tozeroy", line=dict(width=2),
                            )
                        )
                fig_tl.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="Time (s)",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_tl, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — FUSION INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_fusion:
    st.markdown('<div class="section-hdr">Multi-Modal Fusion Intelligence Engine</div>', unsafe_allow_html=True)

    sat_ok  = st.session_state.sat_results  is not None
    drn_ok  = st.session_state.drn_results  is not None

    sc1, sc2 = st.columns(2)
    with sc1:
        if sat_ok:
            st.success("Satellite data ready")
        else:
            st.warning("Satellite data not yet analysed — complete the Satellite tab first")
    with sc2:
        if drn_ok:
            st.success("Drone data ready")
        else:
            st.warning("Drone data not yet analysed — complete the Drone tab first")

    if not sat_ok or not drn_ok:
        missing = []
        if not sat_ok: missing.append("satellite image")
        if not drn_ok: missing.append("drone video")
        _empty_state(
            step="Step 03 of 03",
            title="Run the Fusion Engine",
            body=(
                f"Complete the <b>{' and '.join(missing)}</b> analysis first, "
                "then return here to fuse both data sources into a unified "
                "intelligence assessment with threat scoring, zone analysis, "
                "and actionable recommendations."
            ),
        )

    if sat_ok and drn_ok:
        if st.button(
            "Run Fusion Engine",
            type="primary",
            use_container_width=True,
            key="btn_run_fusion",
        ):
            with st.spinner("Running multi-modal fusion algorithm…"):
                engine = FusionEngine()
                fusion = engine.fuse(
                    st.session_state.sat_results,
                    st.session_state.drn_results,
                )
                st.session_state.fusion_results = fusion
            st.success("Fusion analysis complete.")

    if st.session_state.fusion_results:
        fus = st.session_state.fusion_results
        sat = st.session_state.sat_results
        drn = st.session_state.drn_results

        # Sync floating-assistant state whenever fusion results change or server restarts
        if CIPHER_CHAT_STATE.get("_fus_ts") != fus.get("timestamp", ""):
            _data = {"sat": sat, "drn": drn, "fus": fus}
            CIPHER_CHAT_STATE.update({
                "data":     _data,
                "_fus_ts":  fus.get("timestamp", ""),
                "briefing": _gen_briefing(_data),
                "ready":    True,
            })

        st.divider()

        # ── Module Highlights ────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Module Highlights</div>', unsafe_allow_html=True)
        hi_sat_col, hi_drn_col = st.columns(2)

        with hi_sat_col:
            sat_top = sorted(sat.get("detected_objects", {}).items(),
                             key=lambda x: -x[1]["count"])[:4]
            sat_top_str = "  |  ".join(
                f"<b>{k}</b> &times;{v['count']}" for k, v in sat_top
            ) if sat_top else "None"
            veg_h = sat.get("veg_health", {})
            cloud_cov = sat.get("cloud_info", {}).get("cloud_coverage_pct", 0)
            st.markdown(
                f"""<div class="info-card">
<span style="font-size:0.7rem;font-weight:600;letter-spacing:3px;color:#fff;text-transform:uppercase;">Satellite Module</span><br><br>
<b>Scene Type:</b> {sat.get('scene_type','—')}<br>
<b>Dominant Land:</b> {sat.get('dominant_land','—')}<br>
<b>Top Detections:</b> {sat_top_str}<br>
<b>Veg Health:</b> {veg_h.get('status','—')} (index {veg_h.get('health_index',0):.2f})<br>
<b>Cloud Cover:</b> {cloud_cov:.1f}%<br>
<b>Segmentation:</b> {'SegFormer' if sat.get('segmentation_used') else 'Fallback'} &nbsp;
<b>OBB:</b> {'Active' if sat.get('obb_used') else 'Off'}
</div>""",
                unsafe_allow_html=True,
            )

        with hi_drn_col:
            drn_top = sorted(drn.get("detected_objects", {}).items(),
                             key=lambda x: -x[1]["count"])[:4]
            drn_top_str = "  |  ".join(
                f"<b>{k}</b> &times;{v['count']}" for k, v in drn_top
            ) if drn_top else "None"
            fire_flag = " — FIRE DETECTED" if drn.get("fire_pct_frames", 0) > 1 else ""
            loit_flag = " — LOITERING"     if drn.get("loitering_people", 0) > 0 else ""
            st.markdown(
                f"""<div class="info-card">
<span style="font-size:0.7rem;font-weight:600;letter-spacing:3px;color:#fff;text-transform:uppercase;">Drone Module</span><br><br>
<b>Dominant Scene:</b> {drn.get('dominant_scene','—')}<br>
<b>Model:</b> {'VisDrone + COCO' if drn.get('visdrone_active') else 'COCO only'}<br>
<b>Top Detections:</b> {drn_top_str}<br>
<b>Fire:</b> {drn.get('fire_pct_frames',0):.1f}%{fire_flag} &nbsp;
<b>Smoke:</b> {drn.get('smoke_pct_frames',0):.1f}%<br>
<b>Loitering:</b> {drn.get('loitering_count',0)} tracks ({drn.get('loitering_people',0)} person(s)){loit_flag}<br>
<b>Unique Tracks:</b> {drn.get('total_tracks',0)} &nbsp;
<b>Fast Movers:</b> {drn.get('fast_movers',0)}
</div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Threat Banner ────────────────────────────────────────────────
        threat = fus["threat_level"]
        color_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
        bg_map    = {"HIGH": "#3a0a0a", "MEDIUM": "#3a2a00", "LOW": "#0a2e1a"}
        t_color   = color_map[threat]
        t_bg      = bg_map[threat]
        st.markdown(
            f"""<div class="threat-banner" style="background:{t_bg};border-color:{t_color};">
<span style="width:10px;height:10px;border-radius:50%;background:{t_color};
  display:inline-block;box-shadow:0 0 10px {t_color};
  animation:cipher-pulse 1.8s ease-in-out infinite;"></span>
<span class="threat-text" style="color:{t_color};">THREAT LEVEL: {threat}</span>
<span style="font-size:0.78rem;color:#444;margin-left:1rem;letter-spacing:2px;">
  ACTIVITY {fus['activity_score']:.0%} &nbsp;&mdash;&nbsp;
  FUSION SCORE {fus['fusion_score']:.0f}/100
</span>
</div>""",
            unsafe_allow_html=True,
        )

        # ── KPI Row ──────────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Fused Objects",    fus["total_objects_detected"])
        k2.metric("Classes",          fus["classes_detected"])
        k3.metric("Activity Score",   f"{fus['activity_score']:.0%}")
        k4.metric("Fusion Score",     f"{fus['fusion_score']:.0f}/100")
        k5.metric("Src Agreement",    f"{fus['agreement_rate']:.0%}")
        k6.metric("Improvement",      f"{fus['fusion_improvement']:+.1f}%")

        st.divider()

        # ── Sensor Side-by-Side + Drone-on-Satellite Overlay ────────────
        st.markdown(
            '<div class="section-hdr">Sensor Imagery Comparison</div>',
            unsafe_allow_html=True,
        )
        img_l, img_r = st.columns(2)

        with img_l:
            st.markdown("**Satellite View** — drone detections overlaid")
            from PIL import ImageDraw
            import math as _math

            sat_img  = sat["annotated_image"].copy().convert("RGB")
            draw     = ImageDraw.Draw(sat_img, "RGBA")
            W_si, H_si = sat_img.size
            _pal = ["#ff4757","#ffa502","#2ed573","#1e90ff","#ff6b81","#eccc68","#a29bfe"]

            drn_cls_list = sorted(
                drn.get("detected_objects", {}).items(), key=lambda x: -x[1]["count"]
            )
            n_cls = len(drn_cls_list)

            if drn_cls_list:
                # Distribute markers in a ring around the image centre
                for i, (cls, d) in enumerate(drn_cls_list[:6]):
                    angle = (i / max(n_cls, 1)) * 2 * _math.pi - _math.pi / 2
                    cx = int(W_si / 2 + (W_si * 0.28) * _math.cos(angle))
                    cy = int(H_si / 2 + (H_si * 0.24) * _math.sin(angle))
                    hx = _pal[i % len(_pal)]
                    rc, gc, bc = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
                    draw.ellipse([cx-20,cy-20,cx+20,cy+20],
                                 fill=(rc,gc,bc,55), outline=(rc,gc,bc,210), width=2)
                    draw.ellipse([cx-7,cy-7,cx+7,cy+7], fill=(rc,gc,bc,220))
                    draw.text((cx+14, cy-8), f"▲ {cls} ×{d['count']}",
                              fill=(rc,gc,bc,255))

                # Legend strip at bottom
                leg_h = min(10 + n_cls * 20, H_si // 3)
                draw.rectangle([0, H_si-leg_h, W_si, H_si], fill=(0,0,0,175))
                draw.text((8, H_si-leg_h+4), "UAV Drone Detections:", fill=(220,220,220,255))
                for i, (cls, d) in enumerate(drn_cls_list[:6]):
                    ry = H_si - leg_h + 20 + i * 19
                    hx = _pal[i % len(_pal)]
                    rc, gc, bc = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
                    draw.rectangle([8, ry, 18, ry+13], fill=(rc,gc,bc,220))
                    conf = d.get("avg_confidence", d.get("confidence", 0))
                    draw.text((22, ry), f"{cls}  ×{d['count']}  conf {conf:.0%}",
                              fill=(rc,gc,bc,255))

            st.image(sat_img, use_container_width=True,
                     caption="Satellite annotated image + UAV overlay markers")

        with img_r:
            st.markdown("**Drone UAV View** — annotated video")
            out_vp = drn.get("output_video_path", "")
            if out_vp and os.path.exists(out_vp):
                with open(out_vp, "rb") as _vf:
                    st.video(_vf.read())
            else:
                # fallback: show a sample frame
                sample_frames = drn.get("sample_frames", [])
                if sample_frames:
                    mid_frm = sample_frames[len(sample_frames) // 2]
                    # sample_frames are PIL Images (RGB)
                    st.image(mid_frm, use_container_width=True,
                             caption="Mid-video annotated drone frame")
                else:
                    st.info("No drone video or sample frames available.")
            d1, d2, d3 = st.columns(3)
            d1.metric("Tracks",       drn.get("total_tracks", 0))
            d2.metric("Duration",     f"{drn.get('video_duration', 0):.0f} s")
            d3.metric("Fast Movers",  drn.get("fast_movers", 0))

        st.divider()

        # ── Activity Timeline ────────────────────────────────────────────
        timeline = drn.get("frame_timeline", [])
        if timeline:
            st.markdown(
                '<div class="section-hdr">Drone Activity Timeline</div>',
                unsafe_allow_html=True,
            )
            tl_df = pd.DataFrame(timeline)
            tl_classes = [c for c in tl_df.columns if c != "time" and tl_df[c].sum() > 0]
            if tl_classes:
                tl_pal = ["#00d4ff","#00ffaa","#ff6b6b","#feca57","#ff9ff3","#54a0ff","#a29bfe"]
                fig_tl = go.Figure()
                for j, cls in enumerate(tl_classes):
                    hx = tl_pal[j % len(tl_pal)]
                    rc, gc, bc = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
                    fig_tl.add_trace(go.Scatter(
                        x=tl_df["time"], y=tl_df[cls],
                        mode="lines", name=cls,
                        line=dict(color=hx, width=2),
                        fill="tozeroy",
                        fillcolor=f"rgba({rc},{gc},{bc},0.12)",
                    ))
                fig_tl.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="Time (seconds)",
                    yaxis_title="Object Count",
                    height=230,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_tl, use_container_width=True)

        st.divider()

        # ── Fused Inventory + Source Comparison ─────────────────────────
        st.markdown(
            '<div class="section-hdr">Fused Object Intelligence</div>',
            unsafe_allow_html=True,
        )
        inv_col, chart_col = st.columns([1, 1])

        with inv_col:
            st.markdown("**Fused Object Inventory**")
            inv_rows = [
                {
                    "Object":     cls,
                    "Count":      d["count"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Source":     d["source"],
                }
                for cls, d in sorted(
                    fus["fused_inventory"].items(), key=lambda x: -x[1]["count"]
                )
            ]
            if inv_rows:
                st.dataframe(
                    _style_conf(pd.DataFrame(inv_rows), "Confidence"),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No objects detected in either source.")

            if fus["fused_inventory"]:
                st.markdown("**Drone vs Satellite — Object Count**")
                sat_objs = sat.get("detected_objects", {})
                drn_objs = drn.get("detected_objects", {})
                cls_list = list(fus["fused_inventory"].keys())
                fig_cmp  = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name="Satellite", x=cls_list,
                    y=[sat_objs.get(c, {}).get("count", 0) for c in cls_list],
                    marker_color="#0066cc",
                ))
                fig_cmp.add_trace(go.Bar(
                    name="Drone", x=cls_list,
                    y=[drn_objs.get(c, {}).get("count", 0) for c in cls_list],
                    marker_color="#00aadd",
                ))
                fig_cmp.add_trace(go.Bar(
                    name="Fused", x=cls_list,
                    y=[fus["fused_inventory"][c]["count"] for c in cls_list],
                    marker_color="#00ffaa",
                ))
                fig_cmp.update_layout(
                    **PLOTLY_LAYOUT,
                    barmode="group",
                    xaxis_title="Object Class",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

        with chart_col:
            # Activity gauge
            st.markdown("**Activity & Fusion Confidence**")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fus["activity_score"] * 100,
                delta={"reference": 40, "valueformat": ".1f"},
                title={"text": "Activity Score (%)", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#00d4ff"},
                    "bgcolor": "#091828",
                    "steps": [
                        {"range": [0,  40], "color": "#0a2030"},
                        {"range": [40, 70], "color": "#152840"},
                        {"range": [70,100], "color": "#1e3a55"},
                    ],
                    "threshold": {
                        "line": {"color": "#e74c3c", "width": 4},
                        "thickness": 0.8, "value": 70,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
                height=230, margin=dict(l=20,r=20,t=30,b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Fusion Confidence Score per class
            st.markdown("**Fusion Confidence Score — per class**")
            fused_inv_sorted = sorted(
                fus["fused_inventory"].items(), key=lambda x: -x[1]["confidence"]
            )
            for cls, d in fused_inv_sorted[:8]:
                conf = d["confidence"]
                bar_color = (
                    "#27ae60" if conf >= 0.75 else
                    "#f39c12" if conf >= 0.50 else "#e74c3c"
                )
                st.markdown(
                    f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="width:110px;font-size:0.78rem;color:#ccc;">{cls}</span>
<div style="flex:1;background:#1a2a3a;border-radius:4px;height:14px;position:relative;">
  <div style="width:{conf*100:.0f}%;background:{bar_color};height:100%;border-radius:4px;"></div>
</div>
<span style="width:38px;font-size:0.78rem;color:{bar_color};text-align:right;">{conf:.0%}</span>
</div>""",
                    unsafe_allow_html=True,
                )

            st.divider()

            # Movement stats
            st.markdown("**Movement Dynamics**")
            mv_c1, mv_c2 = st.columns(2)
            mv_c1.metric("Total Tracks",    fus["total_tracks"])
            mv_c1.metric("Fast Movers",     fus["fast_movers"])
            mv_c2.metric("Movement Ratio",  f"{fus['movement_ratio']:.0%}")
            mv_c2.metric("Avg Track Speed", f"{fus['avg_track_speed']:.1f} px/f")

            # Terrain doughnut
            if fus["land_classification"]:
                st.markdown("**Terrain Composition**")
                ld = {k: v for k, v in fus["land_classification"].items() if v > 0}
                _lcmap = {
                    "Vegetation":"#27ae60","Agriculture":"#6ab04c","Water":"#2980b9",
                    "Recreation":"#00cec9","Buildings":"#e17055","Roads":"#636e72",
                    "Airport":"#fdcb6e","Urban":"#7f8c8d","Bare Ground":"#c0933a",
                    "Snow/Clouds":"#bdc3c7","Other":"#2d3436",
                }
                fig_land2 = px.pie(
                    pd.DataFrame(list(ld.items()), columns=["Type","Pct"]),
                    values="Pct", names="Type",
                    color="Type", color_discrete_map=_lcmap, hole=0.4,
                )
                fig_land2.update_traces(textfont_color="white")
                fig_land2.update_layout(
                    **{**PLOTLY_LAYOUT, "margin": dict(l=4,r=4,t=4,b=4)}
                )
                st.plotly_chart(fig_land2, use_container_width=True)

        st.divider()

        # ── Zone Heatmap + Zone Correlation Table ───────────────────────
        st.markdown(
            '<div class="section-hdr">Spatial Zone Analysis</div>',
            unsafe_allow_html=True,
        )
        zone_col, corr_col = st.columns([1, 1])

        with zone_col:
            st.markdown("**Satellite Zone Heatmap (3×3 Grid)**")
            zones = fus["zone_analysis"]
            Z = np.array([
                [zones["NW"], zones["N"],  zones["NE"]],
                [zones["W"],  zones["C"],  zones["E"]],
                [zones["SW"], zones["S"],  zones["SE"]],
            ], dtype=float)
            fig_heat = go.Figure(go.Heatmap(
                z=Z,
                x=["West", "Centre", "East"],
                y=["North", "Middle", "South"],
                colorscale="Blues",
                text=Z.astype(int), texttemplate="%{text}",
                showscale=True,
                colorbar=dict(tickfont=dict(color="white")),
            ))
            fig_heat.update_layout(**PLOTLY_LAYOUT, height=290)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                "Object density per zone from satellite detections. "
                "Centre zone includes drone fast-mover count."
            )

        with corr_col:
            st.markdown("**Zone Correlation Table**")
            _zone_order = [
                ("NW","North-West"), ("N","North"), ("NE","North-East"),
                ("W","West"),        ("C","Centre"),("E","East"),
                ("SW","South-West"), ("S","South"), ("SE","South-East"),
            ]
            _zone_total = max(sum(zones.values()), 1)
            _drn_total  = max(fus["total_tracks"], 1)
            corr_rows   = []
            for zk, zname in _zone_order:
                s_cnt = zones.get(zk, 0)
                s_pct = s_cnt / _zone_total
                d_est = round(_drn_total * s_pct)
                corr  = min(s_pct * 2.5, 1.0)
                status = "High" if corr > 0.45 else "Med" if corr > 0.18 else "Low"
                corr_rows.append({
                    "Zone":        zname,
                    "Sat Objects": s_cnt,
                    "Drone Est.":  d_est,
                    "Correlation": f"{corr:.0%}",
                    "Activity":    status,
                })
            st.dataframe(
                pd.DataFrame(corr_rows),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Drone Est. = drone tracks distributed proportionally to satellite density. "
                "Correlation = normalised zone-weight score."
            )

        # ── Zone Intelligence Summary ─────────────────────────────────────
        _z      = fus["zone_analysis"]
        _ztotal = max(sum(_z.values()), 1)
        _znames = {
            "NW":"North-West","N":"North","NE":"North-East",
            "W":"West","C":"Centre","E":"East",
            "SW":"South-West","S":"South","SE":"South-East",
        }
        _zsorted  = sorted(_z.items(), key=lambda x: -x[1])
        _top_k, _top_v = _zsorted[0]
        _top_pct  = _top_v / _ztotal
        _runner   = [f"• {_znames[k]}: {v} obj ({v/_ztotal:.0%})" for k, v in _zsorted[1:3]]
        _warn     = (
            f"<br><span style='color:#f59e0b;'>&#9888; High concentration in "
            f"{_znames[_top_k]} — consider targeted surveillance of this zone.</span>"
            if _top_pct > 0.35 else ""
        )
        st.markdown(
            f'<div class="info-card">'
            f'<strong style="color:#a3e635;">Activity Hotspot: {_znames[_top_k]}</strong>'
            f' &mdash; {_top_v} objects ({_top_pct:.0%} of all satellite detections)<br>'
            f'{"<br>".join(_runner)}'
            f'{_warn}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Intelligence Summary + Co-detection ─────────────────────────
        summ_col, co_col = st.columns([1, 1])

        with summ_col:
            st.markdown("**Intelligence Summary**")
            st.markdown(
                f'<div class="info-card">{fus["summary"]}</div>',
                unsafe_allow_html=True,
            )

        with co_col:
            st.markdown("**Co-detected Object Pairs**")
            co = fus.get("co_detection_matrix", {})
            if co:
                top_co = sorted(co.items(), key=lambda x: -x[1])[:8]
                co_df  = pd.DataFrame(top_co, columns=["Pair", "Regions"])
                st.dataframe(co_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No co-detection pairs found.")

        st.divider()

        # ── Recommendations ──────────────────────────────────────────────
        st.markdown(
            '<div class="section-hdr">Intelligence Recommendations</div>',
            unsafe_allow_html=True,
        )
        _rec_dot_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}[threat]
        for i, rec in enumerate(fus["recommendations"], 1):
            st.markdown(
                f'<div class="rec-item">'
                f'<span style="color:{_rec_dot_color};margin-right:8px;">&#9632;</span>'
                f'<strong style="color:#ccc;">{i}.</strong> {rec}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — INTELLIGENCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_report:
    st.markdown('<div class="section-hdr">Intelligence Report</div>', unsafe_allow_html=True)

    if not st.session_state.fusion_results:
        st.info(
            "Complete satellite analysis, drone analysis, "
            "and run the Fusion Engine to generate a report."
        )
    else:
        fus2  = st.session_state.fusion_results
        sat2  = st.session_state.sat_results
        drn2  = st.session_state.drn_results

        report_txt = generate_text_report(sat2, drn2, fus2)
        report_csv = generate_csv_report(fus2)

        # Annotated satellite image (generated once, reused for display + PDF)
        _ann_img = None
        if st.session_state.get("sat_original_image") and sat2.get("all_detections"):
            _ann_img = _annotate_sat_clean(
                st.session_state.sat_original_image,
                sat2["all_detections"],
            )

        report_pdf = generate_pdf_report(
            sat2, drn2, fus2,
            logo_path=_logo_path,
            annotated_image=_ann_img,
        )

        ts = fus2["timestamp"].replace(" ", "_").replace(":", "-")

        dl1, dl2, dl3, _ = st.columns([1, 1, 1, 1])
        with dl1:
            st.download_button(
                "⬇ Text Report",
                data=report_txt,
                file_name=f"cipher_report_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇ CSV Data",
                data=report_csv,
                file_name=f"cipher_objects_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "⬇ PDF Report",
                data=report_pdf,
                file_name=f"cipher_report_{ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Drone video download
        drn_vp = drn2.get("output_video_path", "")
        if drn_vp and os.path.exists(drn_vp):
            with open(drn_vp, "rb") as fv2:
                vid_dl = fv2.read()
            dv1, _ = st.columns([1, 3])
            with dv1:
                st.download_button(
                    "⬇ Drone Video",
                    data=vid_dl,
                    file_name="cipher_drone_annotated.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        # ── Annotated Detection Map ───────────────────────────────────────
        if _ann_img is not None:
            st.divider()
            st.markdown(
                '<div class="section-hdr">Annotated Detection Map</div>',
                unsafe_allow_html=True,
            )
            ann_l, ann_r = st.columns([3, 1])
            with ann_l:
                st.image(_ann_img, use_container_width=True,
                         caption="Satellite image with bounding boxes drawn from all_detections")
            with ann_r:
                st.markdown("**Detection Key**")
                inventory2 = fus2.get("fused_inventory", {})
                for cls, d in sorted(inventory2.items(), key=lambda x: -x[1]["count"])[:8]:
                    st.markdown(
                        f'<div style="font-size:0.78rem;margin-bottom:4px;">'
                        f'<span style="color:#a3e635;font-weight:600;">{cls}</span>'
                        f'<span style="color:#888;margin-left:6px;">×{d["count"]}'
                        f' &nbsp;{d["confidence"]:.0%}</span></div>',
                        unsafe_allow_html=True,
                    )
            # PNG download
            import io as _io2
            _ann_buf = _io2.BytesIO()
            _ann_img.save(_ann_buf, format="PNG")
            st.download_button(
                "⬇ Download Annotated Image (PNG)",
                data=_ann_buf.getvalue(),
                file_name=f"cipher_annotated_{ts}.png",
                mime="image/png",
            )

        st.divider()
        st.markdown("**Full Intelligence Report Preview**")
        _rep_esc = report_txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<pre style="background:#020c02;border:1px solid #0d2a0d;border-radius:2px;'
            f'padding:1.5rem 1.8rem;font-family:\'Courier New\',Courier,monospace;'
            f'font-size:0.76rem;color:#a3e635;line-height:1.75;max-height:65vh;'
            f'overflow-y:auto;white-space:pre;'
            f'background-image:repeating-linear-gradient(0deg,transparent,transparent 27px,'
            f'rgba(0,220,0,0.025) 27px,rgba(0,220,0,0.025) 28px);">'
            f'{_rep_esc}</pre>',
            unsafe_allow_html=True,
        )

        # Summary cards at the bottom
        st.divider()
        st.markdown('<div class="section-hdr">Quick Summary</div>', unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Threat Level",      f"{fus2['threat_level']}  {fus2['threat_icon']}")
        q2.metric("Total Objects",     fus2["total_objects_detected"])
        q3.metric("Fusion Score",      f"{fus2['fusion_score']:.0f}/100")
        q4.metric("Scene Type",        fus2["scene_type"])

# ──────────────────────────────────────────────────────────────────────────────
#  Floating Intelligence Assistant Widget
#  Injected into the parent DOM via window.parent so it stays fixed on screen
#  regardless of which tab or scroll position the user is on.
# ──────────────────────────────────────────────────────────────────────────────

if _chat_port:
    import streamlit.components.v1 as _stc
    import base64 as _b64

    _avatar_path = os.path.join(os.path.dirname(__file__), "assets", "avatar.png")
    _avatar_uri  = ""
    if os.path.exists(_avatar_path):
        with open(_avatar_path, "rb") as _f:
            _avatar_uri = "data:image/png;base64," + _b64.b64encode(_f.read()).decode()

    _FLOAT_JS = r"""
(function () {
  try {
    var pd = window.parent.document;
    if (pd.getElementById('cph-btn')) return;   // already injected

    var PORT = __PORT__;

    // ── Floating button ────────────────────────────────────────────────
    var btn = pd.createElement('div');
    btn.id = 'cph-btn';
    btn.title = 'CIPHER Intelligence Assistant';
    var AVATAR = '__AVATAR__';
    btn.innerHTML = AVATAR
      ? '<img src="'+AVATAR+'" style="width:100%;height:100%;border-radius:50%;object-fit:cover;object-position:top center;display:block;" />'
      : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
    btn.style.cssText = 'position:fixed;bottom:24px;right:24px;width:62px;height:62px;'
      + 'background:#111;border-radius:50%;cursor:pointer;z-index:99999;'
      + 'display:flex;align-items:center;justify-content:center;color:#000;'
      + 'box-shadow:0 4px 24px rgba(0,0,0,0.7);overflow:hidden;'
      + 'border:2px solid #f97316;'
      + 'transition:transform 0.18s ease,box-shadow 0.18s ease;user-select:none;';

    // ── Panel ──────────────────────────────────────────────────────────
    var panel = pd.createElement('div');
    panel.id = 'cph-panel';
    panel.style.cssText = 'position:fixed;bottom:86px;right:24px;width:370px;height:520px;'
      + 'background:#070707;border:1px solid #232323;border-radius:4px;'
      + 'z-index:99998;display:none;flex-direction:column;'
      + 'box-shadow:0 12px 48px rgba(0,0,0,0.8);'
      + 'font-family:Inter,"Segoe UI",system-ui,sans-serif;overflow:hidden;';

    panel.innerHTML = ''
      + '<div style="padding:0.6rem 1rem;border-bottom:1px solid #1a1a1a;background:#040404;'
      +   'display:flex;align-items:center;justify-content:space-between;flex-shrink:0;">'
      +   '<div style="display:flex;align-items:center;gap:0.6rem;">'
      +     (AVATAR ? '<img src="'+AVATAR+'" style="width:38px;height:38px;border-radius:50%;'
      +       'object-fit:cover;object-position:top center;border:1.5px solid #f97316;flex-shrink:0;" />' : '')
      +     '<div>'
      +       '<div style="font-size:0.62rem;letter-spacing:4px;color:#fff;'
      +         'text-transform:uppercase;font-weight:600;">Intelligence Assistant</div>'
      +       '<div style="font-size:0.56rem;letter-spacing:2px;color:#f97316;'
      +         'text-transform:uppercase;margin-top:2px;">CIPHER — Analysis Q&amp;A</div>'
      +     '</div>'
      +   '</div>'
      +   '<span id="cph-x" style="cursor:pointer;color:#444;font-size:1.3rem;'
      +     'line-height:1;padding:2px 6px;">&times;</span>'
      + '</div>'
      + '<div id="cph-msgs" style="flex:1;overflow-y:auto;padding:0.72rem;'
      +   'font-size:0.81rem;color:#ccc;line-height:1.65;"></div>'
      + '<div id="cph-sugg" style="padding:0.38rem 0.72rem;border-top:1px solid #131313;'
      +   'display:flex;flex-wrap:wrap;gap:0.28rem;background:#040404;flex-shrink:0;"></div>'
      + '<div style="padding:0.5rem 0.72rem;border-top:1px solid #131313;background:#040404;'
      +   'display:flex;gap:0.45rem;flex-shrink:0;">'
      +   '<input id="cph-inp" type="text" placeholder="Ask about this analysis…" '
      +     'style="flex:1;background:#0d0d0d;border:1px solid #252525;color:#ddd;'
      +     'padding:0.4rem 0.65rem;border-radius:2px;outline:none;'
      +     'font-family:Inter,sans-serif;font-size:0.79rem;" />'
      +   '<button id="cph-go" style="background:#fff;color:#000;border:none;'
      +     'padding:0.4rem 0.82rem;cursor:pointer;border-radius:2px;'
      +     'font-weight:700;font-size:0.68rem;letter-spacing:1.5px;'
      +     'text-transform:uppercase;flex-shrink:0;">Send</button>'
      + '</div>';

    pd.body.appendChild(btn);
    pd.body.appendChild(panel);

    // ── Helpers ────────────────────────────────────────────────────────
    var history = [], open = false, busy = false;
    function g(id) { return pd.getElementById(id); }

    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                      .replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    }

    function avatarEl() {
      return AVATAR
        ? '<img src="'+AVATAR+'" style="width:28px;height:28px;border-radius:50%;'
          + 'object-fit:cover;object-position:top center;flex-shrink:0;'
          + 'border:1.5px solid #f97316;margin-top:2px;" />'
        : '<div style="width:28px;height:28px;border-radius:50%;background:#1a1a1a;'
          + 'border:1px solid #333;flex-shrink:0;"></div>';
    }

    // Instant user message
    function addMsg(role, text) {
      var d = pd.createElement('div');
      d.style.marginBottom = '0.5rem';
      d.innerHTML = '<div style="text-align:right"><span style="background:#181818;'
        + 'padding:0.32rem 0.68rem;border-radius:2px;display:inline-block;'
        + 'max-width:88%;word-wrap:break-word;color:#e0e0e0;">' + esc(text) + '</span></div>';
      var msgs = g('cph-msgs');
      msgs.appendChild(d);
      msgs.scrollTop = msgs.scrollHeight;
    }

    // Bot message with avatar + typewriter effect
    function addBotMsg(text, callback) {
      var d = pd.createElement('div');
      d.style.marginBottom = '0.55rem';
      d.innerHTML = '<div style="display:flex;align-items:flex-start;gap:0.45rem;">'
        + avatarEl()
        + '<span style="background:#091509;padding:0.32rem 0.68rem;border-radius:2px;'
        + 'display:inline-block;max-width:calc(88% - 36px);word-wrap:break-word;'
        + 'color:#a3e635;border-left:2px solid rgba(163,230,53,0.35);"></span></div>';
      var msgs = g('cph-msgs');
      msgs.appendChild(d);
      var span = d.querySelector('span');
      var i = 0;
      var iv = setInterval(function() {
        if (i < text.length) {
          span.innerHTML = esc(text.slice(0, ++i));
          msgs.scrollTop = msgs.scrollHeight;
        } else {
          clearInterval(iv);
          if (callback) callback();
        }
      }, 16);
    }

    // Loader: avatar + animated dots
    function loader(show) {
      var l = g('cph-load');
      if (show && !l) {
        var d = pd.createElement('div');
        d.id = 'cph-load';
        d.style.marginBottom = '0.5rem';
        d.innerHTML = '<div style="display:flex;align-items:flex-start;gap:0.45rem;">'
          + avatarEl()
          + '<span style="background:#091509;padding:0.32rem 0.68rem;border-radius:2px;'
          + 'display:inline-block;color:#a3e635;border-left:2px solid rgba(163,230,53,0.35);'
          + 'font-size:1rem;letter-spacing:2px;">&#8226;&#8226;&#8226;</span></div>';
        msgs.appendChild(d);
        msgs.scrollTop = msgs.scrollHeight;
        // Animate the dots
        var dots = ['&bull;&bull;&bull;','&bull;&bull;&nbsp;','&bull;&nbsp;&nbsp;'];
        var di = 0;
        d._dotIv = setInterval(function(){
          var sp = d.querySelector('span');
          if (sp) sp.innerHTML = dots[di++ % dots.length];
        }, 380);
      } else if (!show && l) {
        if (l._dotIv) clearInterval(l._dotIv);
        l.remove();
      }
    }

    var msgs = g('cph-msgs');

    function send(text) {
      if (!text || busy) return;
      busy = true;
      addMsg('user', text);
      history.push({role:'user', content:text});
      loader(true);
      fetch('http://127.0.0.1:' + PORT, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({history:history})
      })
      .then(function(r){ return r.json(); })
      .then(function(d) {
        loader(false);
        var r = d.response || '(no response)';
        addBotMsg(r, function() {
          history.push({role:'assistant', content:r});
          busy = false;
        });
      })
      .catch(function() {
        loader(false);
        addBotMsg('Could not reach the Intelligence Assistant. Please try again.', function(){ busy = false; });
      });
    }

    function loadInit() {
      msgs.innerHTML = '';
      history = [];
      loader(true);
      fetch('http://127.0.0.1:' + PORT)
      .then(function(r){ return r.json(); })
      .then(function(d) {
        loader(false);
        msgs.innerHTML = '';
        var initText = d.ready && d.briefing
          ? d.briefing
          : 'Run the Fusion Engine first — then come back here for a plain-language briefing on the results.';
        addBotMsg(initText, function() {
          if (d.ready && d.briefing) history.push({role:'assistant', content:d.briefing});
          var sugg = g('cph-sugg');
          sugg.innerHTML = '';
          ['What does the threat level mean?','Is this area safe?',
           'Main objects detected?','Explain the fusion score',
           'What action is needed?'].forEach(function(q) {
            var b = pd.createElement('button');
            b.textContent = q;
            b.style.cssText = 'background:#0d0d0d;border:1px solid #1e1e1e;color:#999;'
              + 'padding:0.2rem 0.48rem;font-size:0.65rem;border-radius:2px;'
              + 'cursor:pointer;white-space:nowrap;';
            b.onmouseover = function(){ b.style.borderColor='#444'; b.style.color='#ddd'; };
            b.onmouseout  = function(){ b.style.borderColor='#1e1e1e'; b.style.color='#999'; };
            b.addEventListener('click', function(){ send(q); });
            sugg.appendChild(b);
          });
        });
      })
      .catch(function() {
        loader(false);
        msgs.innerHTML = '';
        addBotMsg('Intelligence Assistant starting up — try again in a moment.', null);
      });
    }

    // ── Toggle ────────────────────────────────────────────────────────
    function openChat() {
      open = true;
      panel.style.display = 'flex';
      btn.style.transform = 'scale(0.9)';
      loadInit();   // always fresh — no stale history between openings
      setTimeout(function(){ g('cph-inp').focus(); }, 120);
    }
    function closeChat() {
      open = false;
      panel.style.display = 'none';
      btn.style.transform = 'scale(1)';
    }

    btn.addEventListener('click', function(){ open ? closeChat() : openChat(); });
    g('cph-x').addEventListener('click', closeChat);

    function submitInput() {
      var v = g('cph-inp').value.trim();
      if (v) { send(v); g('cph-inp').value = ''; }
    }
    g('cph-go').addEventListener('click', submitInput);
    g('cph-inp').addEventListener('keydown', function(e){ if(e.key==='Enter') submitInput(); });

    btn.onmouseover = function(){ if(!open) btn.style.boxShadow='0 6px 28px rgba(249,115,22,0.45)'; };
    btn.onmouseout  = function(){ if(!open) btn.style.boxShadow='0 4px 24px rgba(0,0,0,0.7)'; };

  } catch(e) { /* cross-origin or env error — fail silently */ }
})();
"""

    _stc.html(
        "<script>" + _FLOAT_JS
            .replace("__PORT__", str(_chat_port))
            .replace("__AVATAR__", _avatar_uri) + "</script>",
        height=0,
        scrolling=False,
    )
