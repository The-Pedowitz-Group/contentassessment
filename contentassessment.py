import os
import io
import sys
import json
import time
import requests
from functools import wraps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PyPDF2 import PdfReader
import docx
from openai import OpenAI

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
#  CONFIGURATION
#  Set these as environment variables on Render.
#  Never hardcode API keys in source code.
# ─────────────────────────────────────────────
HUBSPOT_ACCESS_TOKEN    = os.getenv("HUBSPOT_ACCESS_TOKEN")   # Private App token from HubSpot
HUBSPOT_FORM_PORTAL_ID  = os.getenv("HUBSPOT_PORTAL_ID")      # Your HubSpot portal ID (numeric string)
HUBSPOT_FORM_GUID       = os.getenv("HUBSPOT_FORM_GUID")      # Form GUID from HubSpot Forms

app = Flask(__name__)

# Wildcard CORS — accepts requests from any origin.
# This is the simplest approach and works with HubSpot's CSP.
# The API has no sensitive data that requires origin restriction.
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Disposition"])

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Expose-Headers"]= "Content-Disposition"
    return response


# ─────────────────────────────────────────────
#  RETRY DECORATOR
# ─────────────────────────────────────────────
def retry_on_timeout(max_retries=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "timeout" in str(e).lower() and attempt < max_retries:
                        print(f"Timeout on attempt {attempt + 1}, retrying...", flush=True)
                        time.sleep(2 ** attempt)
                    else:
                        raise
            return None
        return wrapper
    return decorator


# ─────────────────────────────────────────────
#  TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_stream):
    try:
        reader = PdfReader(file_stream)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"PDF extraction error: {e}", flush=True)
        return ""

def extract_text_from_docx(file_stream):
    try:
        doc = docx.Document(file_stream)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"DOCX extraction error: {e}", flush=True)
        return ""


# ─────────────────────────────────────────────
#  OPENAI SCORING
# ─────────────────────────────────────────────
@retry_on_timeout(max_retries=2)
def summarize_insights(text, persona, stage):
    prompt = f"""
You are an expert B2B marketing strategist and content evaluator working for The Pedowitz Group.

Assess how effectively the following content influences a {persona} in the {stage} stage of their
buying journey. Reflect not only general quality but how well the content meets the specific
informational, emotional, and strategic needs of this persona at this stage.

Consider:
- What this persona cares about (ROI, technical fit, risk mitigation, scalability, etc.)
- What this buying stage requires (education, differentiation, trust-building, justification)

Evaluate across these 9 criteria:
1. Clarity & Structure
2. Audience Relevance
3. Value & Insight
4. Call to Action
5. Brand Voice & Tone
6. SEO & Discoverability
7. Visual/Design Integration
8. Performance Readiness
9. AXO / AI Discoverability — How well will this content be surfaced and cited by AI answer
   engines (ChatGPT, Claude, Perplexity, Gemini) when a {persona} at the {stage} stage
   searches for relevant topics? Evaluate: FAQ-style direct-answer blocks, specific claims
   with proof, header hierarchy AI can parse, absence of vague filler language.

For each criterion return:
- score: integer 1-5
- label: criterion name
- reason: concise rationale specific to this persona and stage (not generic)
- recommendation: one clear actionable improvement for this persona and stage
- excerpt_original: the single weakest or most vague sentence from the content for this criterion
- excerpt_rewrite: a specific improved version of that sentence

Return ONLY a valid JSON array. No preamble, no markdown fences, no explanation outside the JSON.

Example format:
[
  {{
    "label": "Clarity & Structure",
    "score": 4,
    "reason": "...",
    "recommendation": "...",
    "excerpt_original": "...",
    "excerpt_rewrite": "..."
  }}
]

Content to evaluate:
{text[:3000]}
"""
    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2500,
            timeout=180
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}", flush=True)
        if "timeout" in str(e).lower():
            return "Error: Request timed out. Please try again."
        return f"Error: {str(e)}"


# ─────────────────────────────────────────────
#  RESPONSE PARSER
# ─────────────────────────────────────────────
def parse_response(feedback):
    try:
        # Strip markdown fences if model adds them despite instructions
        clean = feedback.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        scores = json.loads(clean)
        if isinstance(scores, list) and all("label" in i and "score" in i for i in scores):
            total = sum(int(i.get("score", 0)) for i in scores)
            print(f"Parsed {len(scores)} criteria, total: {total}", flush=True)
            return scores, total

        raise ValueError("Unexpected JSON shape")

    except Exception as e:
        print(f"Parse error: {e} — falling back to text parser", flush=True)

        # Text fallback for non-JSON responses
        scores = []
        current = {}
        for line in feedback.split("\n"):
            line = line.strip()
            if " - Score:" in line:
                if current.get("label"):
                    scores.append(current)
                parts = line.split(" - Score:")
                label = parts[0].split(". ", 1)[-1].strip()
                try:
                    current = {
                        "label": label,
                        "score": int(parts[1].strip()),
                        "reason": "",
                        "recommendation": "",
                        "excerpt_original": "",
                        "excerpt_rewrite": ""
                    }
                except ValueError:
                    pass
            elif line.startswith("Reason:") and current:
                current["reason"] = line.replace("Reason:", "").strip()
            elif line.startswith("Recommendation:") and current:
                current["recommendation"] = line.replace("Recommendation:", "").strip()

        if current.get("label"):
            scores.append(current)

        total = sum(int(i.get("score", 0)) for i in scores)
        return scores, total


# ─────────────────────────────────────────────
#  HUBSPOT FORM SUBMISSION
#  Submits contact data to HubSpot Forms API v3.
#  No email service required — HubSpot workflow
#  handles the follow-up email natively.
# ─────────────────────────────────────────────
def submit_to_hubspot(firstname, lastname, email, company,
                      overall_score, persona, stage, filename, source_page):
    if not HUBSPOT_FORM_PORTAL_ID or not HUBSPOT_FORM_GUID:
        print("HubSpot portal ID or form GUID not configured — skipping HS submission", flush=True)
        return False

    url = (
        f"https://api.hsforms.com/submissions/v3/integration/submit"
        f"/{HUBSPOT_FORM_PORTAL_ID}/{HUBSPOT_FORM_GUID}"
    )

    payload = {
        "fields": [
            {"objectTypeId": "0-1", "name": "firstname",        "value": firstname},
            {"objectTypeId": "0-1", "name": "lastname",         "value": lastname},
            {"objectTypeId": "0-1", "name": "email",            "value": email},
            {"objectTypeId": "0-1", "name": "company",          "value": company},
            {"objectTypeId": "0-1", "name": "axo_score",        "value": str(overall_score)},
            {"objectTypeId": "0-1", "name": "persona",          "value": persona},
            {"objectTypeId": "0-1", "name": "buyer_stage",      "value": stage},
            {"objectTypeId": "0-1", "name": "content_filename", "value": filename},
            {"objectTypeId": "0-1", "name": "source_page",      "value": source_page or ""},
        ],
        "context": {
            "pageUri":  source_page or "https://www.pedowitzgroup.com",
            "pageName": "TPG Content Assessment Tool"
        }
    }

    headers = {"Content-Type": "application/json"}

    # Use Private App token if set (preferred), otherwise anonymous submission
    if HUBSPOT_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {HUBSPOT_ACCESS_TOKEN}"

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"HubSpot submission: {resp.status_code} — {resp.text[:200]}", flush=True)
        return resp.status_code in (200, 204)
    except Exception as e:
        print(f"HubSpot submission error: {e}", flush=True)
        return False


# ─────────────────────────────────────────────
#  PDF REPORT GENERATION
#  Builds a branded TPG report with ReportLab.
#  Returns a BytesIO stream ready for send_file.
# ─────────────────────────────────────────────
def generate_pdf_report(scores, overall_score, persona, stage, filename):
    """
    Premium TPG-branded PDF report matching RM6 assessment quality.
    Uses ReportLab canvas for precise layout control.
    """
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
    from reportlab.graphics import renderPDF

    # ── Brand colors ──
    NAVY   = colors.HexColor("#004963")
    LIME   = colors.HexColor("#abcf37")
    BLUE   = colors.HexColor("#46c5e6")
    TEAL   = colors.HexColor("#168FB1")
    GRAY   = colors.HexColor("#636466")
    LGRAY  = colors.HexColor("#f8fafc")
    WHITE  = colors.white
    AXO    = colors.HexColor("#7c3aed")

    # Score band colors (matching JS colorForScore)
    def score_color(s):
        s = int(s)
        if s >= 5: return colors.HexColor("#48930D")
        if s >= 4: return colors.HexColor("#168FB1")
        if s >= 3: return colors.HexColor("#d97706")
        if s >= 2: return colors.HexColor("#c4361f")
        return colors.HexColor("#8b1d10")

    # Score band label
    def score_band(s):
        s = int(s)
        if s >= 5: return "Excellent"
        if s >= 4: return "Strong"
        if s >= 3: return "Average"
        if s >= 2: return "Needs Work"
        return "Critical"

    # Page setup
    buf = io.BytesIO()
    PAGE_W, PAGE_H = letter
    M = 0.65 * inch   # margin

    # ── CANVAS-BASED APPROACH for precise header/footer control ──
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    def sty(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    s_body  = sty("body",  fontSize=9,  fontName="Helvetica",      textColor=GRAY,  leading=14, spaceAfter=6)
    s_small = sty("small", fontSize=8,  fontName="Helvetica",      textColor=GRAY,  leading=11)
    s_label = sty("label", fontSize=7,  fontName="Helvetica-Bold",  textColor=TEAL,  spaceAfter=3, spaceBefore=8,
                  textTransform="uppercase", letterSpacing=0.5)
    s_rec   = sty("rec",   fontSize=9,  fontName="Helvetica",      textColor=colors.HexColor("#1e293b"),
                  backColor=colors.HexColor("#f0f9ff"), borderPad=8, leading=13, spaceAfter=8,
                  borderColor=BLUE, borderWidth=0.5)
    s_exc   = sty("exc",   fontSize=8,  fontName="Helvetica-Oblique", textColor=colors.HexColor("#991b1b"),
                  backColor=colors.HexColor("#fff0f0"), borderPad=7, leading=12, spaceAfter=4,
                  borderColor=colors.HexColor("#fca5a5"), borderWidth=0.5)
    s_rw    = sty("rw",    fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#166534"),
                  backColor=colors.HexColor("#f0fdf4"), borderPad=7, leading=12, spaceAfter=8,
                  borderColor=colors.HexColor("#86efac"), borderWidth=0.5)
    s_h1    = sty("h1",    fontSize=22, fontName="Helvetica-Bold",  textColor=WHITE, leading=26, spaceAfter=4)
    s_h2    = sty("h2",    fontSize=15, fontName="Helvetica-Bold",  textColor=NAVY,  leading=19, spaceAfter=8, spaceBefore=14)
    s_h3    = sty("h3",    fontSize=11, fontName="Helvetica-Bold",  textColor=NAVY,  leading=15, spaceAfter=6, spaceBefore=10)
    s_foot  = sty("foot",  fontSize=7,  fontName="Helvetica",      textColor=GRAY,  alignment=TA_CENTER, leading=10)

    max_score = len(scores) * 5
    pct = round((overall_score / max_score) * 100) if max_score else 0

    # ── DRAWING HELPERS ──
    def score_bar_drawing(score_val, width=3.8*inch, height=0.18*inch):
        """Segmented 1-5 score bar with marker diamond — mirrors JS journey bar."""
        d = Drawing(width, height + 0.25*inch)
        seg_w = width / 5
        seg_colors = [
            colors.HexColor("#8b1d10"),
            colors.HexColor("#c4361f"),
            colors.HexColor("#d97706"),
            colors.HexColor("#168FB1"),
            colors.HexColor("#48930D"),
        ]
        bar_y = 0.2 * inch
        # Draw segments
        for i, c in enumerate(seg_colors):
            r = Rect(i * seg_w, bar_y, seg_w, height, fillColor=c, strokeColor=None)
            d.add(r)
        # Diamond marker
        s = int(score_val)
        marker_x = ((s - 0.5) / 5) * width
        marker_y = bar_y + height / 2
        sz = 0.07 * inch
        diamond = Polygon([
            marker_x, marker_y + sz,
            marker_x + sz, marker_y,
            marker_x, marker_y - sz,
            marker_x - sz, marker_y,
        ], fillColor=colors.HexColor("#1e293b"), strokeColor=WHITE, strokeWidth=1)
        d.add(diamond)
        # Labels
        labels = ["1", "2", "3", "4", "5"]
        for i, lbl in enumerate(labels):
            lx = (i + 0.5) * seg_w
            s_lbl = String(lx, 0.04*inch, lbl,
                           fontSize=6, fontName="Helvetica",
                           fillColor=colors.HexColor("#636466"),
                           textAnchor="middle")
            d.add(s_lbl)
        return d

    def overall_bar_drawing(pct_val, width=5*inch, height=0.2*inch):
        """Wide gradient bar for overall score."""
        d = Drawing(width, height + 0.3*inch)
        seg_colors = [
            colors.HexColor("#8b1d10"),
            colors.HexColor("#c4361f"),
            colors.HexColor("#d97706"),
            colors.HexColor("#168FB1"),
            colors.HexColor("#48930D"),
        ]
        bar_y = 0.22*inch
        seg_w = width / 5
        for i, c in enumerate(seg_colors):
            d.add(Rect(i*seg_w, bar_y, seg_w, height, fillColor=c, strokeColor=None))
        # Marker
        mx = (pct_val / 100) * width
        my = bar_y + height / 2
        sz = 0.08*inch
        d.add(Polygon([mx, my+sz, mx+sz, my, mx, my-sz, mx-sz, my],
                      fillColor=colors.HexColor("#1e293b"), strokeColor=WHITE, strokeWidth=1.2))
        # Pct labels
        for i, lbl in enumerate(["0%","25%","50%","75%","100%"]):
            d.add(String(i*seg_w, 0.05*inch, lbl,
                         fontSize=6, fontName="Helvetica",
                         fillColor=GRAY, textAnchor="middle"))
        return d

    # ── HEADER / FOOTER via page callback ──
    def on_page(canv, doc):
        canv.saveState()
        pn = doc.page
        # Top rule
        canv.setStrokeColor(LIME)
        canv.setLineWidth(2)
        canv.line(M, PAGE_H - 0.38*inch, PAGE_W - M, PAGE_H - 0.38*inch)
        # Header text (skip cover)
        if pn > 1:
            canv.setFillColor(NAVY)
            canv.setFont("Helvetica-Bold", 8)
            canv.drawString(M, PAGE_H - 0.3*inch, "TPG.ai Content Assessment Report")
            canv.setFillColor(GRAY)
            canv.setFont("Helvetica", 8)
            canv.drawRightString(PAGE_W - M, PAGE_H - 0.3*inch, f"Page {pn}")
        # Footer rule
        canv.setStrokeColor(LIME)
        canv.setLineWidth(1)
        canv.line(M, 0.45*inch, PAGE_W - M, 0.45*inch)
        canv.setFillColor(GRAY)
        canv.setFont("Helvetica", 7)
        canv.drawString(M, 0.28*inch, "© 2026 The Pedowitz Group | pedowitzgroup.com | 404-990-9616")
        canv.drawRightString(PAGE_W - M, 0.28*inch, "Revenue Marketing since 2007")
        canv.restoreState()

    doc = BaseDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=M, leftMargin=M,
        topMargin=0.55*inch, bottomMargin=0.65*inch
    )
    frame = Frame(M, 0.65*inch, PAGE_W - 2*M, PAGE_H - 1.25*inch, id="main")
    doc.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=on_page)])

    # ════════════════════════════════════════════
    # PAGE 1: COVER
    # ════════════════════════════════════════════

    # Navy header band
    cover_header = [[
        Paragraph("TPG.ai Content Assessment Report", s_h1),
        Paragraph(
            f"<font color='#abcf37'><b>{overall_score}/{max_score}</b></font><br/>"
            f"<font size='11' color='white'>{pct}% Performance</font>",
            ParagraphStyle("cs", parent=styles["Normal"], fontSize=24, fontName="Helvetica-Bold",
                           textColor=WHITE, alignment=TA_RIGHT, leading=30)
        )
    ]]
    cover_t = Table(cover_header, colWidths=[4.2*inch, 2.5*inch])
    cover_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), NAVY),
        ("TOPPADDING",   (0,0),(-1,-1), 18),
        ("BOTTOMPADDING",(0,0),(-1,-1), 18),
        ("LEFTPADDING",  (0,0),(-1,-1), 16),
        ("RIGHTPADDING", (0,0),(-1,-1), 16),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(cover_t)
    story.append(Spacer(1, 10))

    # Meta pills row
    meta_row = [[
        Paragraph(f"<b>Persona:</b> {persona}", s_body),
        Paragraph(f"<b>Stage:</b> {stage}", s_body),
        Paragraph(f"<b>File:</b> {filename}", s_body),
        Paragraph(f"<b>Date:</b> {__import__('datetime').date.today().strftime('%B %d, %Y')}", s_body),
    ]]
    meta_t = Table(meta_row, colWidths=[1.7*inch, 1.4*inch, 2.2*inch, 1.4*inch])
    meta_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), LGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("LINEBELOW",    (0,0),(-1,-1), 1.5, LIME),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 18))

    # Overall score visual
    story.append(Paragraph("Overall Performance", s_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=LIME, spaceAfter=10))

    overall_band_data = [[
        Paragraph(f"<b><font size='36' color='#004963'>{overall_score}</font>"
                  f"<font size='18' color='#636466'>/{max_score}</font></b>",
                  ParagraphStyle("bignum", parent=styles["Normal"],
                                 fontName="Helvetica-Bold", fontSize=36, leading=40)),
        Paragraph(
            f"<b>{pct}% Performance</b><br/>"
            f"<font color='#636466'>{'Top Performer' if pct>=75 else 'Approaching Top Quartile' if pct>=50 else 'Needs Improvement'}</font><br/>"
            f"<br/>"
            f"<font size='8' color='#636466'>Top quartile benchmark for {persona}: ~{round(len(scores)*5*0.8)}/{max_score}</font>",
            ParagraphStyle("meta2", parent=styles["Normal"],
                           fontSize=13, fontName="Helvetica-Bold",
                           textColor=NAVY, leading=18)
        )
    ]]
    overall_band_t = Table(overall_band_data, colWidths=[1.4*inch, 5.3*inch])
    overall_band_t.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 12),
        ("RIGHTPADDING", (0,0),(-1,-1), 12),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("BACKGROUND",   (0,0),(-1,-1), LGRAY),
        ("LINEAFTER",    (0,0),(0,-1),  1.5, LIME),
    ]))
    story.append(overall_band_t)
    story.append(Spacer(1, 10))

    # Overall bar
    from reportlab.platypus import Image as RLImage
    story.append(overall_bar_drawing(pct))
    story.append(Spacer(1, 16))

    # ── SCORE SUMMARY TABLE ──
    story.append(Paragraph("Score Summary", s_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=LIME, spaceAfter=10))

    sum_header = [
        Paragraph("<b>Criterion</b>", ParagraphStyle("th", parent=styles["Normal"],
            fontSize=8, fontName="Helvetica-Bold", textColor=WHITE)),
        Paragraph("<b>Score</b>", ParagraphStyle("th2", parent=styles["Normal"],
            fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Band</b>", ParagraphStyle("th3", parent=styles["Normal"],
            fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Score Bar</b>", ParagraphStyle("th4", parent=styles["Normal"],
            fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
    ]
    sum_rows = [sum_header]
    for item in scores:
        s = int(item.get("score", 0))
        sc = score_color(s)
        is_axo = "axo" in (item.get("label","")).lower() or "ai discoverability" in (item.get("label","")).lower()
        lbl = item.get("label", "")
        if is_axo:
            lbl = "🤖 " + lbl
        sum_rows.append([
            Paragraph(lbl, ParagraphStyle("td", parent=styles["Normal"],
                fontSize=9, fontName="Helvetica", textColor=NAVY if not is_axo else AXO)),
            Paragraph(f"<b>{s}/5</b>", ParagraphStyle("sc", parent=styles["Normal"],
                fontSize=10, fontName="Helvetica-Bold", textColor=sc, alignment=TA_CENTER)),
            Paragraph(score_band(s), ParagraphStyle("bd", parent=styles["Normal"],
                fontSize=8, fontName="Helvetica", textColor=sc, alignment=TA_CENTER)),
            score_bar_drawing(s, width=2.5*inch, height=0.12*inch),
        ])
    # Total row
    sum_rows.append([
        Paragraph("<b>OVERALL</b>", ParagraphStyle("tot", parent=styles["Normal"],
            fontSize=9, fontName="Helvetica-Bold", textColor=NAVY)),
        Paragraph(f"<b>{overall_score}/{max_score}</b>", ParagraphStyle("tots", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica-Bold", textColor=NAVY, alignment=TA_CENTER)),
        Paragraph(f"<b>{pct}%</b>", ParagraphStyle("totp", parent=styles["Normal"],
            fontSize=9, fontName="Helvetica-Bold", textColor=NAVY, alignment=TA_CENTER)),
        Paragraph("", s_body),
    ])

    sum_t = Table(sum_rows, colWidths=[2.8*inch, 0.7*inch, 0.9*inch, 2.35*inch])
    ts = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#e8f4f8")),
        ("ROWBACKGROUNDS",(0,1), (-1,-2), [WHITE, LGRAY]),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,-2), 0.5, colors.HexColor("#e2e8f0")),
        ("LINEBELOW",     (0,-1),(-1,-1), 2,   NAVY),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ])
    sum_t.setStyle(ts)
    story.append(sum_t)

    # ════════════════════════════════════════════
    # PAGES 2+: DETAILED CRITERIA CARDS
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Detailed Analysis", s_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=LIME, spaceAfter=12))

    for i, item in enumerate(scores):
        s     = int(item.get("score", 0))
        sc    = score_color(s)
        label = item.get("label", f"Criterion {i+1}")
        is_axo = "axo" in label.lower() or "ai discoverability" in label.lower()
        badge_color = AXO if is_axo else sc
        eyebrow = f"Criterion 0{i+1}" if i < 9 else f"Criterion {i+1}"

        # Card header bar
        card_hdr = [[
            Paragraph(
                f"<font size='7' color='#abcf37'>{eyebrow}</font><br/>"
                f"{'🤖 ' if is_axo else ''}{label}",
                ParagraphStyle("ch", parent=styles["Normal"], fontSize=12,
                               fontName="Helvetica-Bold", textColor=WHITE, leading=16)
            ),
            Paragraph(
                f"<b>{s}/5</b><br/><font size='9'>{score_band(s)}</font>",
                ParagraphStyle("cs2", parent=styles["Normal"], fontSize=18,
                               fontName="Helvetica-Bold", textColor=WHITE,
                               alignment=TA_RIGHT, leading=22)
            ),
        ]]
        card_hdr_t = Table(card_hdr, colWidths=[4.8*inch, 1.9*inch])
        card_hdr_t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), badge_color),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",  (0,0),(-1,-1), 14),
            ("RIGHTPADDING", (0,0),(-1,-1), 14),
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(card_hdr_t)

        # Score bar
        story.append(Spacer(1, 4))
        story.append(score_bar_drawing(s, width=6.7*inch, height=0.14*inch))
        story.append(Spacer(1, 8))

        # Analysis text
        if item.get("reason"):
            story.append(Paragraph(item["reason"], s_body))

        # Snippet block
        if item.get("excerpt_original"):
            story.append(Paragraph("WEAKEST EXCERPT", s_label))
            story.append(Paragraph(f'"{item["excerpt_original"]}"', s_exc))
        if item.get("excerpt_rewrite"):
            story.append(Paragraph("SUGGESTED REWRITE", s_label))
            story.append(Paragraph(item["excerpt_rewrite"], s_rw))

        # Recommendation
        if item.get("recommendation"):
            story.append(Paragraph("RECOMMENDATION", s_label))
            story.append(Paragraph(item["recommendation"], s_rec))

        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#e2e8f0"), spaceAfter=10))

    # ════════════════════════════════════════════
    # FINAL PAGE: NEXT STEPS + AXO CALLOUT
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Your Next Steps", s_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=LIME, spaceAfter=12))

    # Adaptive CTA
    if pct < 50:
        cta_text  = ("Your content scores below 50%. A full rebuild is the fastest path to pipeline impact. "
                     "TPG's Content Strategy Sprint rebuilds pieces like this in 4 weeks.")
        cta_link  = "https://www.pedowitzgroup.com/contact"
        cta_label = "Book a Strategy Call"
        cta_bg    = colors.HexColor("#991b1b")
    elif pct < 75:
        cta_text  = ("You're approaching top-quartile performance. Targeted fixes to your lowest 2-3 criteria "
                     "will close the gap. Our Content Optimization Sprint does exactly that in 4 weeks.")
        cta_link  = "https://www.pedowitzgroup.com/contact"
        cta_label = "See the Content Sprint"
        cta_bg    = colors.HexColor("#713f12")
    else:
        cta_text  = ("Strong content foundation. AXO optimization is your highest-leverage next move — "
                     "ensuring AI answer engines surface this content when your buyers search.")
        cta_link  = "https://www.pedowitzgroup.com/the-complete-guide-to-answer-engine-optimization-aeo"
        cta_label = "Explore AXO Optimization"
        cta_bg    = colors.HexColor("#166534")

    cta_data = [[
        Paragraph(cta_text, ParagraphStyle("cta", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica", textColor=WHITE, leading=15)),
        Paragraph(f"<b>{cta_label}</b><br/><font size='8'>{cta_link}</font>",
            ParagraphStyle("ctab", parent=styles["Normal"], fontSize=11,
                fontName="Helvetica-Bold", textColor=LIME, alignment=TA_CENTER, leading=16))
    ]]
    cta_t = Table(cta_data, colWidths=[4.5*inch, 2.2*inch])
    cta_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), NAVY),
        ("TOPPADDING",   (0,0),(-1,-1), 16),
        ("BOTTOMPADDING",(0,0),(-1,-1), 16),
        ("LEFTPADDING",  (0,0),(-1,-1), 16),
        ("RIGHTPADDING", (0,0),(-1,-1), 16),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LINEAFTER",    (0,0),(0,-1),  1.5, LIME),
    ]))
    story.append(cta_t)
    story.append(Spacer(1, 20))

    # AXO callout
    axo_items = [i for i in scores if "axo" in i.get("label","").lower()
                 or "ai discoverability" in i.get("label","").lower()]
    if axo_items:
        axo_s = int(axo_items[0].get("score", 0))
        story.append(Paragraph(
            f"<b>Your AXO Score: {axo_s}/5</b> — AI Discoverability",
            ParagraphStyle("axo_h", parent=styles["Normal"], fontSize=12,
                           fontName="Helvetica-Bold", textColor=AXO, spaceAfter=6)
        ))
        story.append(Paragraph(
            f"67% of B2B buyers now use AI answer engines (ChatGPT, Claude, Perplexity, Gemini) "
            f"before visiting a vendor website. Your AXO score of {axo_s}/5 "
            f"{'means your content is being surfaced and cited by AI. Maintain this by refreshing FAQ content and proof points.' if axo_s >= 4 else 'means AI answer engines are unlikely to cite this content when your buyers search. Adding FAQ blocks, direct-answer formatting, and specific data points can change that.'}",
            ParagraphStyle("axo_b", parent=styles["Normal"], fontSize=9,
                           fontName="Helvetica", textColor=NAVY,
                           backColor=colors.HexColor("#faf5ff"),
                           borderPad=12, leading=14, spaceAfter=16,
                           borderColor=AXO, borderWidth=1)
        ))

    # Priority fixes table
    sorted_scores = sorted(scores, key=lambda x: int(x.get("score", 0)))
    low_three = sorted_scores[:3]
    story.append(Paragraph("Priority Fixes", s_h2))
    story.append(HRFlowable(width="100%", thickness=1.5, color=LIME, spaceAfter=8))
    story.append(Paragraph(
        "These three criteria have the most pipeline leverage. Fix them first.",
        s_body
    ))
    story.append(Spacer(1, 8))

    for rank, item in enumerate(low_three, 1):
        s = int(item.get("score", 0))
        sc = score_color(s)
        is_axo = "axo" in item.get("label","").lower()
        badge_c = AXO if is_axo else sc

        pfix_data = [[
            Paragraph(f"<b>#{rank}</b>", ParagraphStyle("rnk", parent=styles["Normal"],
                fontSize=16, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
            Paragraph(
                f"<font size='8' color='#abcf37'>{'🤖 ' if is_axo else ''}Score {s}/5</font><br/>"
                f"<b>{item.get('label','')}</b><br/>"
                f"<font size='8'>{item.get('recommendation','')}</font>",
                ParagraphStyle("pfix", parent=styles["Normal"], fontSize=9,
                               fontName="Helvetica", textColor=WHITE, leading=13)
            ),
        ]]
        pfix_t = Table(pfix_data, colWidths=[0.45*inch, 6.2*inch])
        pfix_t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), badge_c),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",  (0,0),(-1,-1), 12),
            ("RIGHTPADDING", (0,0),(-1,-1), 12),
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ("LINEAFTER",    (0,0),(0,-1),  1, LIME),
        ]))
        story.append(pfix_t)
        story.append(Spacer(1, 6))

    # Final footer CTA
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=LIME, spaceAfter=8))
    story.append(Paragraph(
        "The Pedowitz Group &nbsp;|&nbsp; pedowitzgroup.com &nbsp;|&nbsp; 404-990-9616 &nbsp;|&nbsp; Revenue Marketing since 2007",
        s_foot
    ))
    story.append(Paragraph(
        f"This report was generated by TPG.ai Content Assessment for a {persona} at the {stage} stage. "
        f"Scores are based on TPG's 9-criterion framework including AXO: AI Discoverability.",
        s_foot
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Content Assessment API is running", "status": "healthy"})


@app.route("/test", methods=["GET", "POST"])
def test():
    return jsonify({
        "message": "Test endpoint working",
        "openai_key_set":   bool(openai.api_key),
        "hubspot_configured": bool(HUBSPOT_FORM_PORTAL_ID and HUBSPOT_FORM_GUID),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts: multipart/form-data
      file     — PDF, DOCX, or TXT
      persona  — target persona string
      stage    — buyer journey stage string

    Returns: JSON { scores: [...], overall_score: int }
    """
    try:
        file    = request.files.get("file")
        persona = request.form.get("persona", "General")
        stage   = request.form.get("stage", "Unaware")

        print(f"=== ANALYSIS: {persona} / {stage} ===", flush=True)

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = file.filename.lower()
        print(f"Processing: {filename}", flush=True)

        if filename.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        elif filename.endswith(".docx"):
            content = extract_text_from_docx(file)
        elif filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="ignore")
        else:
            return jsonify({"error": "Unsupported file format. Use PDF, DOCX, or TXT."}), 400

        if not content.strip():
            return jsonify({"error": "Could not extract text from file."}), 400

        print(f"Extracted {len(content)} chars", flush=True)

        feedback = summarize_insights(content, persona, stage)

        if feedback.startswith("Error:"):
            return jsonify({"error": feedback}), 500

        scores, total_score = parse_response(feedback)

        if not scores:
            return jsonify({
                "error": "Could not parse AI response",
                "raw_response": feedback,
                "scores": [],
                "overall_score": 0
            })

        return jsonify({"scores": scores, "overall_score": total_score})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/submit-and-pdf", methods=["POST"])
def submit_and_pdf():
    """
    Called when the user completes the HubSpot gate form in the modal.

    Accepts: application/json
    {
      "firstname":     "Jane",
      "lastname":      "Smith",
      "email":         "jane@company.com",
      "company":       "Acme Corp",
      "scores":        [...],          // array from /analyze
      "overall_score": 29,
      "persona":       "CMO",
      "stage":         "Decision",
      "filename":      "brief.pdf",
      "source_page":   "https://..."
    }

    Returns: PDF file as attachment
             AND submits contact to HubSpot Forms API
             (HubSpot workflow then sends the follow-up email natively)
    """
    try:
        data         = request.get_json(force=True)
        firstname    = data.get("firstname", "")
        lastname     = data.get("lastname", "")
        email        = data.get("email", "")
        company      = data.get("company", "")
        scores       = data.get("scores", [])
        overall_score= int(data.get("overall_score", 0))
        persona      = data.get("persona", "General")
        stage        = data.get("stage", "Unaware")
        filename     = data.get("filename", "content")
        source_page  = data.get("source_page", "")

        if not email:
            return jsonify({"error": "Email is required"}), 400
        if not scores:
            return jsonify({"error": "No scores provided"}), 400

        # 1. Submit to HubSpot — workflow handles follow-up email
        hs_ok = submit_to_hubspot(
            firstname, lastname, email, company,
            overall_score, persona, stage, filename, source_page
        )
        print(f"HubSpot submission: {'OK' if hs_ok else 'FAILED'}", flush=True)

        # 2. Generate PDF
        pdf_buf = generate_pdf_report(scores, overall_score, persona, stage, filename)

        # 3. Return PDF as download
        safe_name = f"TPG_Content_Assessment_{firstname}_{lastname}.pdf".replace(" ", "_")
        return send_file(
            pdf_buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=safe_name
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting on port {port}", flush=True)
    print(f"OpenAI key set:       {bool(openai.api_key)}", flush=True)
    print(f"HubSpot configured:   {bool(HUBSPOT_FORM_PORTAL_ID and HUBSPOT_FORM_GUID)}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
