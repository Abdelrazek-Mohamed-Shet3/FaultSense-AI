"""Executive PDF and JSON inspection report generator (Zero-Blue Calm White Palette)."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


CHARCOAL = colors.HexColor("#1C1E1B")
WARM_GRAY = colors.HexColor("#5A5D55")
STONE_BORDER = colors.HexColor("#E6E2D8")
ALABASTER = colors.HexColor("#FAF9F6")
SAGE = colors.HexColor("#4E7A65")
SAGE_TINT = colors.HexColor("#EFF5F1")
AMBER = colors.HexColor("#B3833E")
AMBER_TINT = colors.HexColor("#FBF5EA")
TERRACOTTA = colors.HexColor("#B85D4E")
TERRACOTTA_TINT = colors.HexColor("#FDF0EE")


def _severity_colors(level: str):
    if level == "critical":
        return TERRACOTTA, TERRACOTTA_TINT
    if level == "warning":
        return AMBER, AMBER_TINT
    return SAGE, SAGE_TINT


def generate_pdf_report(record: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FSTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=CHARCOAL,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "FSSub",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=WARM_GRAY,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "FSH2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=CHARCOAL,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "FSBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=CHARCOAL,
    )

    sev_fg, sev_bg = _severity_colors(record.get("severity_level", "pass"))
    story = []

    story.append(Paragraph("FaultSense AI — Deep Learning Inspection Report", title_style))
    ts = record.get("created_at") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    story.append(
        Paragraph(
            f"Inspection ID: <b>{record['id']}</b> &nbsp;|&nbsp; Media: <b>{record['media_type'].upper()}</b> "
            f"&nbsp;|&nbsp; Domain: <b>{record['domain'].title()}</b> &nbsp;|&nbsp; Generated: {ts}",
            subtitle_style,
        )
    )

    # Summary Banner Table
    conf_pct = f"{round(float(record['confidence']) * 100, 1)}%"
    sev_score = f"{record['severity_score']} / 100"
    summary_data = [
        ["Predicted Classification", "Confidence", "Severity Index", "Status"],
        [
            record["label_en"],
            conf_pct,
            sev_score,
            record["severity_level"].upper(),
        ],
    ]
    summary_tbl = Table(summary_data, colWidths=[68 * mm, 32 * mm, 36 * mm, 38 * mm])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ALABASTER),
                ("TEXTCOLOR", (0, 0), (-1, 0), WARM_GRAY),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, 1), sev_bg),
                ("TEXTCOLOR", (0, 1), (-1, 1), sev_fg),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 10.5),
                ("BOX", (0, 0), (-1, -1), 0.75, STONE_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, STONE_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Diagnostic Summary & Recommended Action", h2_style))
    story.append(Paragraph(f"<b>Observation:</b> {record.get('description_en', '')}", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Action Plan:</b> {record.get('recommendation_en', '')}", body_style))
    story.append(Spacer(1, 10))

    # Visual Evidence Row (Original, Grad-CAM, Localized)
    file_paths = record.get("file_paths", {})
    img_cells = []
    cap_cells = []
    for key, label in [
        ("original", "Input / Peak Frame"),
        ("gradcam", "Grad-CAM Attention"),
        ("localized", "Localized Fault Regions"),
    ]:
        p = file_paths.get(key)
        if p and Path(p).exists():
            img_cells.append(RLImage(p, width=54 * mm, height=54 * mm, kind="proportional"))
            cap_cells.append(label)

    if img_cells:
        story.append(Paragraph("Explainable AI Visual Evidence (Grad-CAM & Region Localization)", h2_style))
        vis_tbl = Table([img_cells, cap_cells], colWidths=[58 * mm] * len(img_cells))
        vis_tbl.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 1), (-1, 1), 8.5),
                    ("TEXTCOLOR", (0, 1), (-1, 1), WARM_GRAY),
                    ("BOX", (0, 0), (-1, -1), 0.5, STONE_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(vis_tbl)
        story.append(Spacer(1, 10))

    # Probability Distribution Table
    probs = record.get("probabilities", [])
    if probs:
        story.append(Paragraph("Top Class Probability Breakdown", h2_style))
        p_rows = [["Class", "Domain", "Severity Tier", "Probability"]]
        for item in probs[:6]:
            p_rows.append(
                [
                    item["label_en"],
                    item["domain"].title(),
                    item["severity_level"].upper(),
                    f"{round(float(item['probability']) * 100, 2)}%",
                ]
            )
        p_tbl = Table(p_rows, colWidths=[72 * mm, 36 * mm, 34 * mm, 32 * mm])
        p_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ALABASTER),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WARM_GRAY),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOX", (0, 0), (-1, -1), 0.5, STONE_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, STONE_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(p_tbl)

    doc.build(story)
    return out_path
