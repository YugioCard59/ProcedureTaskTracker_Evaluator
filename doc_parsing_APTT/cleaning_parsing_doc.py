"""
Strict Annotation & Structural Hybrid PDF Table Generator (TypeError Fix)
-------------------------------------------------------------------------------
Fixes:
  - TypeError: float object cannot be interpreted as an integer:
    All page bounding boxes, crop tuples, and annotation coordinate intersections 
    are explicitly converted to integers using int() / round().

Dependencies:
    pip install pdfplumber reportlab
"""

import gc
import os
import re
import sys
import pdfplumber
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def classify_highlight_color(color_val) -> str:
    """Classifies annotation color values into TITLE (Yellow) or SUBTITLE (Green)."""
    if not color_val or not isinstance(color_val, (tuple, list)):
        return "NONE"

    rgb = [c * 255 if isinstance(c, float) and c <= 1.0 else c for c in color_val[:3]]
    r, g, b = rgb[0], rgb[1], rgb[2]

    # Yellow -> TITLE
    if r > 160 and g > 160 and b < 140:
        return "TITLE"

    # Green -> SUBTITLE
    if g > 120 and r < 180:
        return "SUBTITLE"

    return "NONE"


def get_annotation_rects(page) -> list[dict]:
    """Extracts highlight annotations and converts bottom-up Y coords to integer top-down space."""
    normalized = []
    page_height = float(page.height)
    annots = page.annots or []

    for a in annots:
        subtype = str(a.get("subtype", "")) or str(a.get("type", ""))
        if "Highlight" in subtype or "Square" in subtype:
            rect = a.get("rect")
            if rect and len(rect) == 4:
                x0, y0, x1, y1 = [float(v) for v in rect]
                top = page_height - y1
                bottom = page_height - y0
                role = classify_highlight_color(a.get("color"))

                # Explicitly cast coordinates to integers
                normalized.append({
                    "x0": int(min(x0, x1)),
                    "x1": int(max(x0, x1)),
                    "top": int(min(top, bottom)),
                    "bottom": int(max(top, bottom)),
                    "role": role
                })
    return normalized


def is_text_highlighted(line_bbox: tuple, annots: list[dict]) -> str:
    """Checks if a text line's integer bounding box intersects with any annotation overlay."""
    x0, top, x1, bottom = line_bbox
    line_cx = (x0 + x1) / 2
    line_cy = (top + bottom) / 2

    for a in annots:
        # Generous boundary overlap allowance using int bounds
        if (a["x0"] - 10) <= line_cx <= (a["x1"] + 10) and (a["top"] - 8) <= line_cy <= (a["bottom"] + 8):
            return a["role"]
    return "NONE"


def is_explicit_step_start(text: str) -> bool:
    """Checks for numbered or lettered prefixes (1., A., I.)."""
    return bool(re.match(r"^(?:[IXVLCDM]+|\d+|[a-zA-Z])[\.\)]\s+", text.strip(), re.IGNORECASE))


def parse_pdf_document(pdf_path: str) -> tuple[str, list[tuple[str, str]]]:
    """Parses PDF pages, identifies highlights, cleans noise, and yields structured rows."""
    clean_path = os.path.abspath(pdf_path)
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"File not found: {clean_path}")

    ignore_patterns = [
        r"^S/U\b", r"^S\b", r"^U\b", r"^n\.d\.\b", r"Print Student's Name",
        r"Faculty Instructions", r"RECHECK", r"Completes procedure",
        r"Faculty Signature", r"Student Signature", r"End Time:", r"Start Time:",
        r"Comments:", r"Time:", r"Educational Use Only", r"Page \d+ of \d+"
    ]

    extracted_tagged_lines = []

    with pdfplumber.open(clean_path) as pdf:
        for page in pdf.pages:
            width = int(round(float(page.width)))
            height = int(round(float(page.height)))

            # Crop spatial margins with explicit INT values to prevent float errors
            crop_box = (30, 60, width - 30, height - 60)
            cropped = page.crop(crop_box)
            annots = get_annotation_rects(page)

            words = cropped.extract_words(keep_blank_chars=False)
            if not words:
                continue

            # Group words by line level (rounded to int positions)
            lines_dict = {}
            for w in words:
                top_key = int(round(float(w["top"])))
                if top_key not in lines_dict:
                    lines_dict[top_key] = []
                lines_dict[top_key].append(w)

            for top_pos in sorted(lines_dict.keys()):
                line_words = sorted(lines_dict[top_pos], key=lambda x: x["x0"])
                line_text = " ".join(w["text"] for w in line_words).strip()

                # Scrub Wingdings / Glyphs / Checkboxes / Standalone S/U flags
                line_text = re.sub(r"[\u2600-\u27BF\uE000-\uF8FF\u25A0-\u25FF\u2713\u2714]", "", line_text)
                line_text = re.sub(r"\b(S/U|2nd|3rd|n\.d\.|S|U)\b", "", line_text, flags=re.I).strip()

                if not line_text or any(re.search(pat, line_text, re.I) for pat in ignore_patterns):
                    continue

                line_bbox = (
                    int(min(w["x0"] for w in line_words)),
                    int(min(w["top"] for w in line_words)),
                    int(max(w["x1"] for w in line_words)),
                    int(max(w["bottom"] for w in line_words))
                )

                # Check annotation intersection
                role = is_text_highlighted(line_bbox, annots)

                # Fallback structural keyword tags if annotation intersection missed
                if role == "NONE":
                    if any(term in line_text for term in ["Entry Measures", "Exit Measures", "Physical Assessment"]):
                        role = "TITLE"
                    elif any(line_text.startswith(kw) for kw in ["Cardiovascular", "Respiratory", "Gastrointestinal", "Genitourinary", "Musculoskeletal", "Integumentary", "HEENT/Neurological", "Vital Signs", "Safety Assessment"]):
                        role = "SUBTITLE"

                extracted_tagged_lines.append((role, line_text))

    gc.collect()

    # Process extracted lines into rows
    doc_title = ""
    structured_rows = []
    summary_items = []
    in_summary = False
    current_step = ""

    for idx, (role, text) in enumerate(extracted_tagged_lines):
        if idx == 0 and role == "TITLE" and not doc_title:
            doc_title = text
            continue

        if "Performance Summary" in text or "Summary Measures" in text:
            if current_step:
                structured_rows.append(("STEP", current_step))
                current_step = ""
            in_summary = True
            continue

        if in_summary:
            if role in ["TITLE", "SUBTITLE"]:
                in_summary = False
            else:
                summary_items.append(text)
                continue

        if role in ["TITLE", "SUBTITLE"]:
            if current_step:
                structured_rows.append(("STEP", current_step))
                current_step = ""
            structured_rows.append((role, text))

        else:
            # Handle step boundaries vs multi-line sentence continuations
            if is_explicit_step_start(text):
                if current_step:
                    structured_rows.append(("STEP", current_step))
                current_step = text
            elif current_step:
                if re.search(r"[\.\?\!:]$", current_step) and text[0].isupper():
                    structured_rows.append(("STEP", current_step))
                    current_step = text
                else:
                    current_step += f" {text}"
            else:
                current_step = text

    if current_step:
        structured_rows.append(("STEP", current_step))

    if summary_items:
        combined_summary = "Performance Summary Measures:<br/><br/>" + "<br/>".join(summary_items)
        structured_rows.append(("SUMMARY_BLOCK", combined_summary))

    return doc_title, structured_rows


def generate_clean_table_pdf(input_path: str, output_path: str) -> None:
    """Generates the single-column table output with TITLE, SUBTITLE, and STEP labels."""
    doc_title, structured_rows = parse_pdf_document(input_path)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    instruction_style = ParagraphStyle("Instr", parent=styles["Normal"], fontSize=9, leading=13, spaceAfter=8)
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9, leading=13)
    title_style = ParagraphStyle("TitleCell", parent=styles["Heading2"], fontSize=10, leading=14)
    subtitle_style = ParagraphStyle("SubtitleCell", parent=styles["Heading3"], fontSize=9.5, leading=13.5)

    story = [
        Paragraph(
            "<b>Instructions:</b> Please add standardized peer reviewed rationales for each step in "
            "the appropriate <b>RATIONALE:</b> boxes, please provide all citations to rationales in the "
            "<b>CITATIONS:</b> box and update the Skills Performance version used in the <b>VERSION:</b> box. "
            "When complete please upload the document through the appropriate channels for APTT use. Thank you.",
            instruction_style
        ),
        Spacer(1, 8)
    ]

    table_data = []

    if doc_title:
        table_data.append([Paragraph(f"<b>TITLE:</b> {doc_title}", title_style)])

    for row_type, text in structured_rows:
        if row_type == "TITLE":
            table_data.append([Paragraph(f"<b>TITLE:</b> {text}", title_style)])

        elif row_type == "SUBTITLE":
            table_data.append([Paragraph(f"<b>SUBTITLE:</b> {text}", subtitle_style)])

        elif row_type == "STEP":
            table_data.append([Paragraph(f"<b>STEP:</b> {text}", cell_style)])
            table_data.append([Paragraph("<b>RATIONALE:</b> ", cell_style)])

        elif row_type == "SUMMARY_BLOCK":
            table_data.append([Paragraph(f"<b>SUMMARY:</b> {text}", cell_style)])

    # Append empty Citations and Version rows at the end
    table_data.append([Paragraph("<b>CITATIONS:</b> ", cell_style)])
    table_data.append([Paragraph("<b>VERSION:</b> ", cell_style)])

    pdf_table = Table(table_data, colWidths=[540])
    pdf_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.75, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(pdf_table)
    doc.build(story)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_pdf.py <input_pdf_or_folder>", file=sys.stderr)
        sys.exit(1)

    target_path = os.path.abspath(sys.argv[1])

    if os.path.isfile(target_path):
        out = os.path.splitext(target_path)[0] + "_clean.pdf"
        generate_clean_table_pdf(target_path, out)
        print(f"Successfully generated: {out}")
    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith(".pdf") and "_clean" not in file.lower():
                    inp = os.path.join(root, file)
                    out = os.path.splitext(inp)[0] + "_clean.pdf"
                    try:
                        generate_clean_table_pdf(inp, out)
                        print(f"Cleaned: {file} -> {os.path.basename(out)}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}", file=sys.stderr)