"""
Report Generator
================
Produces text, CSV, and PDF intelligence reports from fusion results.
"""

from __future__ import annotations
import io
import os
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────

def generate_text_report(
    satellite: dict,
    drone: dict,
    fusion: dict,
) -> str:
    lines: list[str] = []
    sep = "=" * 68

    def add(text: str = ""):
        lines.append(text)

    add(sep)
    add("  CIPHER INTELLIGENCE REPORT")
    add("  Combined Intelligence Platform for High-resolution Environmental Recon")
    add(sep)
    add(f"  Generated  : {fusion['timestamp']}")
    add(f"  Scene Type : {fusion['scene_type']}")
    add(f"  Classification: UNCLASSIFIED — DEMO")
    add(sep)
    add()

    # ── Executive Summary ──────────────────────────────────────────────
    add("EXECUTIVE SUMMARY")
    add("-" * 48)
    for line in _wrap(fusion["summary"], 66):
        add(f"  {line}")
    add()

    # ── Threat Assessment ──────────────────────────────────────────────
    add("THREAT / ACTIVITY ASSESSMENT")
    add("-" * 48)
    add(f"  Threat Level    : {fusion['threat_level']}  {fusion['threat_icon']}")
    add(f"  Activity Score  : {fusion['activity_score']:.1%}")
    add(f"  Fusion Score    : {fusion['fusion_score']:.1f} / 100")
    add(f"  Movement Ratio  : {fusion['movement_ratio']:.1%}  ({fusion['fast_movers']} fast-moving objects)")
    add()

    # ── Terrain Analysis ───────────────────────────────────────────────
    add("TERRAIN / LAND ANALYSIS  (Satellite-derived)")
    add("-" * 48)
    for k, v in fusion["land_classification"].items():
        bar = "█" * int(v / 4)
        add(f"  {k:<20} {v:>5.1f}%  {bar}")
    add(f"  Dominant Type  : {fusion['dominant_land']}")
    add()

    # ── Fused Object Inventory ─────────────────────────────────────────
    add("FUSED OBJECT INVENTORY")
    add("-" * 48)
    header = f"  {'Object':<20} {'Count':>6}  {'Confidence':>10}  {'Source':<12}"
    add(header)
    add(f"  {'-'*20} {'-'*6}  {'-'*10}  {'-'*12}")
    for cls, data in sorted(
        fusion["fused_inventory"].items(), key=lambda x: -x[1]["count"]
    ):
        add(
            f"  {cls:<20} {data['count']:>6}  "
            f"{data['confidence']:>10.1%}  {data['source']:<12}"
        )
    add(f"\n  TOTAL OBJECTS DETECTED: {fusion['total_objects_detected']}")
    add()

    # ── Fusion Quality ─────────────────────────────────────────────────
    add("FUSION QUALITY METRICS")
    add("-" * 48)
    add(f"  Source Agreement Rate    : {fusion['agreement_rate']:.1%}")
    add(f"  Satellite Detections     : {fusion['satellite_object_count']}")
    add(f"  Drone Detections         : {fusion['drone_object_count']}")
    add(f"  Fused Detections         : {fusion['total_objects_detected']}")
    add(f"  Improvement vs Best-Solo : {fusion['fusion_improvement']:+.1f}%")
    add()

    # ── Drone Video Stats ──────────────────────────────────────────────
    add("DRONE VIDEO ANALYTICS")
    add("-" * 48)
    vd_flag = "VisDrone + COCO (aerial-trained)" if drone.get("visdrone_active") else "COCO only"
    add(f"  Detection Model : {vd_flag}")
    add(f"  Duration        : {drone['video_duration']:.1f} s")
    add(f"  Total Frames    : {drone['total_frames']}")
    add(f"  Unique Tracks   : {fusion['total_tracks']}")
    add(f"  Fast Movers     : {fusion['fast_movers']}")
    add(f"  Avg Track Speed : {fusion['avg_track_speed']:.1f} px/frame")
    add(f"  Dominant Scene  : {drone.get('dominant_scene', '—')}")
    fire_pct = drone.get('fire_pct_frames', 0)
    smoke_pct = drone.get('smoke_pct_frames', 0)
    loiter = drone.get('loitering_count', 0)
    add(f"  Fire Frames     : {fire_pct:.1f}%  |  Smoke Frames: {smoke_pct:.1f}%")
    add(f"  Loitering Tracks: {loiter}  ({drone.get('loitering_people', 0)} person(s))")
    alerts = drone.get("alerts", [])
    if alerts:
        add("  Behavioural Alerts:")
        for a in alerts[:5]:
            add(f"    • {a}")
    add()

    # ── Co-detection ───────────────────────────────────────────────────
    if fusion.get("co_detection_matrix"):
        add("CO-DETECTED OBJECT PAIRS  (Satellite regions)")
        add("-" * 48)
        for pair, cnt in sorted(
            fusion["co_detection_matrix"].items(), key=lambda x: -x[1]
        )[:5]:
            add(f"  {pair:<30} {cnt} region(s)")
        add()

    # ── Recommendations ────────────────────────────────────────────────
    add("INTELLIGENCE RECOMMENDATIONS")
    add("-" * 48)
    for i, rec in enumerate(fusion["recommendations"], 1):
        for j, line in enumerate(_wrap(rec, 64)):
            prefix = f"  {i}. " if j == 0 else "     "
            add(f"{prefix}{line}")
    add()

    add(sep)
    add("  END OF REPORT")
    add(sep)

    return "\n".join(lines)


def generate_csv_report(fusion: dict) -> str:
    rows = [
        {
            "Object Class": cls,
            "Count": data["count"],
            "Confidence": f"{data['confidence']:.1%}",
            "Source": data["source"],
        }
        for cls, data in sorted(
            fusion["fused_inventory"].items(), key=lambda x: -x[1]["count"]
        )
    ]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


# ──────────────────────────────────────────────────────────────────────────────
#  PDF report
# ──────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    satellite: dict,
    drone: dict,
    fusion: dict,
    logo_path: str | None = None,
    annotated_image=None,   # PIL Image or None
) -> bytes:
    """Return a professional PDF intelligence report as raw bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.lib import colors as RL
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.platypus import Image as RLImage

    # ── Palette ────────────────────────────────────────────────────────────
    BLACK  = RL.HexColor("#000000")
    WHITE  = RL.HexColor("#ffffff")
    GREEN  = RL.HexColor("#a3e635")
    DARK   = RL.HexColor("#0a0a0a")
    GREY1  = RL.HexColor("#1a1a1a")
    GREY2  = RL.HexColor("#2a2a2a")
    GREY3  = RL.HexColor("#444444")
    LGREY  = RL.HexColor("#cccccc")
    THREAT_COLORS = {
        "HIGH":     RL.HexColor("#ef4444"),
        "CRITICAL": RL.HexColor("#dc2626"),
        "MEDIUM":   RL.HexColor("#f59e0b"),
        "LOW":      RL.HexColor("#22c55e"),
    }
    threat   = fusion.get("threat_level", "LOW").upper()
    THREAT_C = THREAT_COLORS.get(threat, GREEN)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="CIPHER Intelligence Report",
        author="CIPHER Platform",
    )

    def _para(text, size=10, color=WHITE, bold=False, align=TA_LEFT, top=0, bot=2):
        s = ParagraphStyle(
            "p", fontSize=size, textColor=color,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            alignment=align, spaceBefore=top*mm, spaceAfter=bot*mm,
            leading=size * 1.45,
        )
        return Paragraph(text, s)

    story = []

    # ── Logo / header ──────────────────────────────────────────────────────
    if logo_path and os.path.exists(logo_path):
        try:
            story.append(RLImage(logo_path, width=14*cm, height=3.5*cm, kind="proportional"))
            story.append(Spacer(1, 3*mm))
        except Exception:
            story.append(_para("CIPHER", 26, WHITE, bold=True, align=TA_CENTER))
    else:
        story.append(_para("CIPHER", 26, WHITE, bold=True, align=TA_CENTER))
        story.append(_para(
            "Combined Intelligence Platform for High-resolution Environmental Recon",
            8, LGREY, align=TA_CENTER,
        ))
        story.append(Spacer(1, 3*mm))

    story.append(HRFlowable(width="100%", thickness=1, color=GREY2))
    story.append(Spacer(1, 2*mm))

    # Metadata
    meta = [
        ["Generated",      fusion.get("timestamp", "—")],
        ["Scene Type",     fusion.get("scene_type", "—")],
        ["Classification", "UNCLASSIFIED — DEMO"],
    ]
    mt = Table(meta, colWidths=[3.5*cm, 13*cm])
    mt.setStyle(TableStyle([
        ("TEXTCOLOR",    (0,0), (-1,-1), LGREY),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [DARK, GREY1]),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
    ]))
    story.append(mt)
    story.append(Spacer(1, 4*mm))

    # ── Threat banner ──────────────────────────────────────────────────────
    threat_desc = {
        "LOW":      "Area appears calm with minimal unusual activity.",
        "MEDIUM":   "Moderate activity detected. Monitoring recommended.",
        "HIGH":     "Significant threats detected. Immediate attention advised.",
        "CRITICAL": "Critical threats detected. Immediate response required.",
    }.get(threat, f"{threat} — review full report.")

    tt = Table(
        [[_para(f"THREAT LEVEL: {threat}", 14, WHITE, bold=True),
          _para(threat_desc, 9, WHITE)]],
        colWidths=[5*cm, 11.5*cm],
    )
    tt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), THREAT_C),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tt)
    story.append(Spacer(1, 5*mm))

    # ── Key metrics ────────────────────────────────────────────────────────
    story.append(_para("KEY METRICS", 11, GREEN, bold=True, bot=2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
    story.append(Spacer(1, 2*mm))
    mdata = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Objects",        str(fusion.get("total_objects_detected", 0)),
         "Fusion Score",         f"{fusion.get('fusion_score', 0):.0f}/100"],
        ["Activity Score",       f"{fusion.get('activity_score', 0):.0%}",
         "Source Agreement",     f"{fusion.get('agreement_rate', 0):.0%}"],
        ["Satellite Detections", str(fusion.get("satellite_object_count", 0)),
         "Drone Detections",     str(fusion.get("drone_object_count", 0))],
        ["Dominant Land",        fusion.get("dominant_land", "—"),
         "Scene Type",           fusion.get("scene_type", "—")],
        ["Total Tracks",         str(fusion.get("total_tracks", 0)),
         "Fast Movers",          str(fusion.get("fast_movers", 0))],
    ]
    mt2 = Table(mdata, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm])
    mt2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), GREY2),
        ("TEXTCOLOR",     (0,0), (-1,0), GREEN),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (2,1), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,1), (-1,-1), LGREY),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [DARK, GREY1]),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,0), 0.5, GREEN),
        ("GRID",          (0,0), (-1,-1), 0.3, GREY3),
    ]))
    story.append(mt2)
    story.append(Spacer(1, 5*mm))

    # ── Annotated satellite image ──────────────────────────────────────────
    if annotated_image is not None:
        try:
            img_buf = io.BytesIO()
            annotated_image.save(img_buf, format="PNG")
            img_buf.seek(0)
            story.append(_para("ANNOTATED SATELLITE IMAGE", 11, GREEN, bold=True, bot=2))
            story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
            story.append(Spacer(1, 2*mm))
            story.append(RLImage(img_buf, width=16*cm, height=9*cm, kind="proportional"))
            story.append(Spacer(1, 5*mm))
        except Exception:
            pass

    # ── Zone analysis ──────────────────────────────────────────────────────
    zones = fusion.get("zone_analysis", {})
    if zones:
        story.append(_para("SPATIAL ZONE ANALYSIS (3×3 Grid)", 11, GREEN, bold=True, bot=2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
        story.append(Spacer(1, 2*mm))
        zone_max = max(zones.values()) if zones else 1

        def _zbg(key):
            v = zones.get(key, 0)
            t = v / max(zone_max, 1)
            if t > 0.7: return RL.HexColor("#1a4a1a")
            if t > 0.4: return RL.HexColor("#0d2a0d")
            if t > 0.2: return RL.HexColor("#0a1a0a")
            return DARK

        zd = [
            [f"NW\n{zones.get('NW',0)}", f"N\n{zones.get('N',0)}",  f"NE\n{zones.get('NE',0)}"],
            [f"W\n{zones.get('W',0)}",   f"C\n{zones.get('C',0)}",  f"E\n{zones.get('E',0)}"],
            [f"SW\n{zones.get('SW',0)}", f"S\n{zones.get('S',0)}",  f"SE\n{zones.get('SE',0)}"],
        ]
        zt = Table(zd, colWidths=[5.3*cm]*3, rowHeights=[1.4*cm]*3)
        zs = [
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0,0), (-1,-1), GREEN),
            ("ALIGN",     (0,0), (-1,-1), "CENTER"),
            ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
            ("GRID",      (0,0), (-1,-1), 0.5, GREY3),
        ]
        for ri, row in enumerate([["NW","N","NE"],["W","C","E"],["SW","S","SE"]]):
            for ci, k in enumerate(row):
                zs.append(("BACKGROUND", (ci,ri), (ci,ri), _zbg(k)))
        zt.setStyle(TableStyle(zs))
        story.append(zt)

        zone_full = {
            "NW":"North-West","N":"North","NE":"North-East",
            "W":"West","C":"Centre","E":"East",
            "SW":"South-West","S":"South","SE":"South-East",
        }
        top_k  = max(zones, key=zones.get)
        ztotal = max(sum(zones.values()), 1)
        story.append(Spacer(1, 2*mm))
        story.append(_para(
            f"Activity Hotspot: {zone_full.get(top_k, top_k)} — "
            f"{zones[top_k]} objects ({zones[top_k]/ztotal:.0%} of all detections)",
            8, LGREY,
        ))
        story.append(Spacer(1, 5*mm))

    # ── Object inventory ───────────────────────────────────────────────────
    inventory = fusion.get("fused_inventory", {})
    if inventory:
        story.append(_para("FUSED OBJECT INVENTORY", 11, GREEN, bold=True, bot=2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
        story.append(Spacer(1, 2*mm))
        inv_data = [["Object Class", "Count", "Confidence", "Source"]]
        for cls, d in sorted(inventory.items(), key=lambda x: -x[1]["count"])[:12]:
            inv_data.append([cls, str(d["count"]), f"{d['confidence']:.0%}", d["source"]])
        it = Table(inv_data, colWidths=[6*cm, 3*cm, 3.5*cm, 4*cm])
        it.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), GREY2),
            ("TEXTCOLOR",     (0,0), (-1,0), GREEN),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("TEXTCOLOR",     (0,1), (-1,-1), LGREY),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [DARK, GREY1]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("GRID",          (0,0), (-1,-1), 0.3, GREY3),
        ]))
        story.append(it)
        story.append(Spacer(1, 5*mm))

    # ── Recommendations ────────────────────────────────────────────────────
    recs = fusion.get("recommendations", [])
    if recs:
        story.append(_para("INTELLIGENCE RECOMMENDATIONS", 11, GREEN, bold=True, bot=2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
        story.append(Spacer(1, 2*mm))
        for i, rec in enumerate(recs, 1):
            story.append(_para(f"{i}. {rec}", 9, LGREY, bot=2))
        story.append(Spacer(1, 5*mm))

    # ── Summary ────────────────────────────────────────────────────────────
    summary = fusion.get("summary", "")
    if summary:
        story.append(_para("INTELLIGENCE SUMMARY", 11, GREEN, bold=True, bot=2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
        story.append(Spacer(1, 2*mm))
        story.append(_para(summary, 9, LGREY))
        story.append(Spacer(1, 5*mm))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY2))
    story.append(Spacer(1, 2*mm))
    story.append(_para(
        "CIPHER — Combined Intelligence Platform for High-resolution "
        "Environmental Recon  |  UNCLASSIFIED — DEMO",
        7, GREY3, align=TA_CENTER,
    ))

    doc.build(story)
    return buf.getvalue()
