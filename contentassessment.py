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
{text[:4000]}
"""
    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000,
            timeout=90
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
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    # TPG brand colors
    TPG_NAVY   = colors.HexColor("#004963")
    TPG_LIME   = colors.HexColor("#abcf37")
    TPG_BLUE   = colors.HexColor("#46c5e6")
    TPG_TEAL   = colors.HexColor("#168FB1")
    TPG_GRAY   = colors.HexColor("#636466")
    TPG_LGRAY  = colors.HexColor("#f8fafc")
    SCORE_COLORS = {
        5: colors.HexColor("#48930D"),
        4: colors.HexColor("#168FB1"),
        3: colors.HexColor("#f59e0b"),
        2: colors.HexColor("#ef4444"),
        1: colors.HexColor("#dc2626"),
    }

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_h1 = ParagraphStyle("H1", parent=styles["Normal"],
        fontSize=26, fontName="Helvetica-Bold",
        textColor=colors.white, spaceAfter=4, leading=30)
    style_h2 = ParagraphStyle("H2", parent=styles["Normal"],
        fontSize=16, fontName="Helvetica-Bold",
        textColor=TPG_NAVY, spaceAfter=6, spaceBefore=14)
    style_h3 = ParagraphStyle("H3", parent=styles["Normal"],
        fontSize=12, fontName="Helvetica-Bold",
        textColor=TPG_NAVY, spaceAfter=4, spaceBefore=10)
    style_body = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica",
        textColor=TPG_GRAY, leading=15, spaceAfter=4)
    style_small = ParagraphStyle("Small", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica",
        textColor=TPG_GRAY, leading=12)
    style_label = ParagraphStyle("Label", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica-Bold",
        textColor=TPG_TEAL, spaceAfter=2, spaceBefore=6)
    style_excerpt = ParagraphStyle("Excerpt", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#991b1b"),
        backColor=colors.HexColor("#fff0f0"),
        borderPad=6, leading=14, spaceAfter=4)
    style_rewrite = ParagraphStyle("Rewrite", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#166534"),
        backColor=colors.HexColor("#f0fdf4"),
        borderPad=6, leading=14, spaceAfter=6)
    style_rec = ParagraphStyle("Rec", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#1e293b"),
        backColor=colors.HexColor("#f0f9ff"),
        borderPad=6, leading=14, spaceAfter=8,
        leftIndent=8)
    style_center = ParagraphStyle("Center", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica",
        textColor=TPG_GRAY, alignment=TA_CENTER)
    style_footer = ParagraphStyle("Footer", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#888B8D"), alignment=TA_CENTER)

    max_score = len(scores) * 5
    pct = round((overall_score / max_score) * 100) if max_score else 0

    def score_label(s):
        if s >= 4: return "Strong"
        if s == 3: return "Average"
        return "Needs Work"

    story = []

    # ── COVER HEADER (navy box) ──
    header_data = [[
        Paragraph("TPG.ai Content Assessment Report", style_h1),
        Paragraph(
            f"<font color='#abcf37'><b>{overall_score}/{max_score}</b></font>"
            f"<br/><font size='11' color='white'>{pct}% Performance</font>",
            ParagraphStyle("Score", parent=styles["Normal"],
                fontSize=26, fontName="Helvetica-Bold",
                textColor=colors.white, alignment=TA_RIGHT, leading=32)
        )
    ]]
    header_table = Table(header_data, colWidths=[4.5 * inch, 2.5 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), TPG_NAVY),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 20),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",(0, 0), (-1, -1), 18),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [8, 8, 8, 8]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ── META ROW ──
    meta_data = [[
        Paragraph(f"<b>Persona:</b> {persona}", style_body),
        Paragraph(f"<b>Buying Stage:</b> {stage}", style_body),
        Paragraph(f"<b>File:</b> {filename}", style_body),
    ]]
    meta_table = Table(meta_data, colWidths=[2.25 * inch, 2.25 * inch, 2.5 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), TPG_LGRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW",    (0, 0), (-1, -1), 1, TPG_LIME),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── SCORE SUMMARY TABLE ──
    story.append(Paragraph("Score Summary", style_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=TPG_LIME, spaceAfter=10))

    summary_header = [
        Paragraph("<b>Criterion</b>", ParagraphStyle("TH", parent=styles["Normal"],
            fontSize=9, fontName="Helvetica-Bold", textColor=colors.white)),
        Paragraph("<b>Score</b>", ParagraphStyle("TH2", parent=styles["Normal"],
            fontSize=9, fontName="Helvetica-Bold", textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("<b>Rating</b>", ParagraphStyle("TH3", parent=styles["Normal"],
            fontSize=9, fontName="Helvetica-Bold", textColor=colors.white, alignment=TA_CENTER)),
    ]
    summary_rows = [summary_header]
    for item in scores:
        s = int(item.get("score", 0))
        sc = SCORE_COLORS.get(s, TPG_GRAY)
        summary_rows.append([
            Paragraph(item.get("label", ""), style_body),
            Paragraph(f"<b>{s}/5</b>",
                ParagraphStyle("Sc", parent=styles["Normal"],
                    fontSize=10, fontName="Helvetica-Bold",
                    textColor=sc, alignment=TA_CENTER)),
            Paragraph(score_label(s),
                ParagraphStyle("Rt", parent=styles["Normal"],
                    fontSize=9, fontName="Helvetica",
                    textColor=sc, alignment=TA_CENTER)),
        ])
    # Total row
    summary_rows.append([
        Paragraph("<b>OVERALL SCORE</b>",
            ParagraphStyle("Tot", parent=styles["Normal"],
                fontSize=10, fontName="Helvetica-Bold", textColor=TPG_NAVY)),
        Paragraph(f"<b>{overall_score}/{max_score}</b>",
            ParagraphStyle("TotS", parent=styles["Normal"],
                fontSize=11, fontName="Helvetica-Bold",
                textColor=TPG_NAVY, alignment=TA_CENTER)),
        Paragraph(f"<b>{pct}%</b>",
            ParagraphStyle("TotP", parent=styles["Normal"],
                fontSize=10, fontName="Helvetica-Bold",
                textColor=TPG_NAVY, alignment=TA_CENTER)),
    ])

    summary_table = Table(summary_rows, colWidths=[3.8 * inch, 1.1 * inch, 2.1 * inch])
    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  TPG_NAVY),
        ("BACKGROUND",   (0, -1),(-1, -1), colors.HexColor("#e8f4f8")),
        ("LINEBELOW",    (0, -1),(-1, -1), 2, TPG_NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2), [colors.white, TPG_LGRAY]),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ])
    summary_table.setStyle(ts)
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # ── DETAILED CRITERIA ──
    story.append(Paragraph("Detailed Analysis", style_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=TPG_LIME, spaceAfter=12))

    for i, item in enumerate(scores):
        s     = int(item.get("score", 0))
        sc    = SCORE_COLORS.get(s, TPG_GRAY)
        label = item.get("label", f"Criterion {i+1}")
        is_axo = "axo" in label.lower() or "ai discoverability" in label.lower()

        # Card header: label + score badge
        badge_color = colors.HexColor("#7c3aed") if is_axo else sc
        card_header = [[
            Paragraph(
                f"{'🤖 ' if is_axo else ''}{label}",
                ParagraphStyle("CardH", parent=styles["Normal"],
                    fontSize=12, fontName="Helvetica-Bold",
                    textColor=colors.white)
            ),
            Paragraph(
                f"<b>{s}/5</b>  {score_label(s)}",
                ParagraphStyle("CardS", parent=styles["Normal"],
                    fontSize=11, fontName="Helvetica-Bold",
                    textColor=colors.white, alignment=TA_RIGHT)
            ),
        ]]
        card_header_table = Table(card_header, colWidths=[4.5 * inch, 2.5 * inch])
        card_header_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), badge_color),
            ("TOPPADDING",  (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING",(0,0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",(0, 0), (-1, -1), 12),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(card_header_table)

        # Analysis
        if item.get("reason"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(item["reason"], style_body))

        # Worst excerpt + rewrite
        if item.get("excerpt_original"):
            story.append(Paragraph("WEAKEST EXCERPT", style_label))
            story.append(Paragraph(f'"{item["excerpt_original"]}"', style_excerpt))
        if item.get("excerpt_rewrite"):
            story.append(Paragraph("SUGGESTED REWRITE", style_label))
            story.append(Paragraph(item["excerpt_rewrite"], style_rewrite))

        # Recommendation
        if item.get("recommendation"):
            story.append(Paragraph("RECOMMENDATION", style_label))
            story.append(Paragraph(item["recommendation"], style_rec))

        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#e2e8f0"), spaceAfter=8))

    # ── NEXT STEPS ──
    story.append(PageBreak())
    story.append(Paragraph("Your Next Steps", style_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=TPG_LIME, spaceAfter=12))

    if pct < 50:
        cta_text = (
            "Your content scores below 50%. A full rebuild is the fastest path to pipeline impact. "
            "TPG's Content Strategy Sprint takes your core messaging from unclear to compelling in 4 weeks."
        )
        cta_link = "https://www.pedowitzgroup.com/contact"
        cta_label = "Book a Strategy Call"
    elif pct < 75:
        cta_text = (
            "You're approaching top-quartile performance. Our Content Optimization Sprint typically "
            "closes gaps like yours in 4 weeks — with specific rewrites, AXO optimization, and "
            "persona-calibrated messaging."
        )
        cta_link = "https://www.pedowitzgroup.com/content-sprint"
        cta_label = "See the Content Sprint"
    else:
        cta_text = (
            "Strong foundation. The highest-leverage next move is AXO optimization: ensuring AI "
            "answer engines surface your content when your buyers are searching. "
            "TPG's AXO service typically improves AI visibility scores by 2-3 points in 60 days."
        )
        cta_link = "https://www.pedowitzgroup.com/axo"
        cta_label = "Explore AXO Optimization"

    # CTA box
    cta_data = [[
        Paragraph(cta_text, ParagraphStyle("CTA", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica", textColor=colors.white, leading=16)),
        Paragraph(f"<b>{cta_label}</b><br/><font size='8'>{cta_link}</font>",
            ParagraphStyle("CTABtn", parent=styles["Normal"],
                fontSize=11, fontName="Helvetica-Bold",
                textColor=TPG_LIME, alignment=TA_CENTER, leading=16))
    ]]
    cta_table = Table(cta_data, colWidths=[4.5 * inch, 2.5 * inch])
    cta_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), TPG_NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING",(0,0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",(0, 0), (-1, -1), 16),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LINEAFTER",   (0, 0), (0, -1),  1, TPG_LIME),
    ]))
    story.append(cta_table)
    story.append(Spacer(1, 20))

    # AXO callout box
    axo_scores = [i for i in scores if "axo" in i.get("label","").lower()
                  or "ai discoverability" in i.get("label","").lower()]
    if axo_scores:
        axo_s = int(axo_scores[0].get("score", 0))
        axo_text = (
            f"Your AXO score is <b>{axo_s}/5</b>. "
            "This measures how visible your content is to AI answer engines like ChatGPT, "
            "Claude, Perplexity, and Gemini. 67% of B2B buyers now use AI answer engines "
            "before visiting a vendor website. If AI doesn't surface your content, "
            "you're invisible at the top of the new funnel."
        )
        story.append(Paragraph(
            axo_text,
            ParagraphStyle("AXO", parent=styles["Normal"],
                fontSize=10, fontName="Helvetica", textColor=TPG_NAVY,
                backColor=colors.HexColor("#faf5ff"),
                borderPad=12, leading=16, spaceAfter=16,
                borderColor=colors.HexColor("#7c3aed"),
                borderWidth=1)
        ))

    # ── FOOTER ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=TPG_LIME, spaceAfter=8))
    story.append(Paragraph(
        "The Pedowitz Group | pedowitzgroup.com | Revenue Marketing since 2007",
        style_footer
    ))
    story.append(Paragraph(
        "This report was generated by TPG.ai Content Assessment. "
        "Scores reflect analysis of the submitted content against TPG's 9-criterion framework "
        f"for a {persona} at the {stage} stage.",
        style_footer
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
