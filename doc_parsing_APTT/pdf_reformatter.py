import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pdfplumber
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

# Logistic Note: From reformatted cleaned dir move or copy those PDF files to 
# JSONenabled_APTT dir > raw_docs to be fed into json_builder to change to text and make JSON


class CleanPDFReformatter:
    """
    Strips leftmost column and two rightmost columns from input PDF tables,
    cleans glyphs/wingdings, and outputs a cleanly structured table PDF.
    """

    STEP_PREFIX_REGEX = re.compile(
        r"^(\(?\d+[\.\)]|\(?[a-zA-Z][\.\)]|\(?[ivxLCDM]+[\.\)])\s+(.*)",
        re.IGNORECASE
    )

    def __init__(self, input_path: str, output_folder: str):
        self.input_path = Path(input_path)
        self.output_folder = Path(output_folder)

    @staticmethod
    def clean_text(text: str) -> str:
        """Strips wingdings, unicode checkmarks, non-printable symbols, and normalizes spacing."""
        if not text:
            return ""
        # Filter non-ASCII, Wingdings, checkmarks, ballot boxes, and bullet symbols
        text = re.sub(r'[^\x20-\x7E\n\t°½⅓⅔¼¾⅛⅜⅝⅞±µμ]', '', text)
        text = re.sub(r'[\u2700-\u27BF\u2600-\u26FF\u2B00-\u2BFF\u25A0-\u25FF\u2610-\u2612]', '', text)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
        return "\n".join([line for line in lines if line])

    def parse_pdf(self, pdf_file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Slices out Column 0 and Last 2 Columns, preserving center content cells."""
        raw_cells = []

        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                extracted_tables = page.extract_tables()
                
                if extracted_tables:
                    for table in extracted_tables:
                        for row in table:
                            if not row:
                                continue
                            
                            # Determine column slicing based on row length
                            # Goal: Remove leftmost (index 0) and 2 rightmost (indices -2, -1)
                            if len(row) > 3:
                                target_cells = row[1:-2]
                            elif len(row) == 3:
                                target_cells = [row[1]]  # Keep middle column only
                            elif len(row) == 2:
                                target_cells = [row[0]]  # Fallback: drop rightmost column
                            else:
                                target_cells = row

                            row_content = []
                            for cell in target_cells:
                                if not cell:
                                    continue
                                cleaned = self.clean_text(cell)
                                if cleaned:
                                    row_content.append(cleaned)

                            if row_content:
                                raw_cells.append(row_content)

        if not raw_cells:
            return "Document", []

        # Determine top main title
        first_cell = raw_cells[0]
        first_line = first_cell[0].splitlines()[0] if first_cell else "Document"
        main_title = first_line

        rows = []
        is_first_cell = True

        for cell_group in raw_cells:
            cell_text = "\n".join(cell_group)
            lines = [line.strip() for line in cell_text.splitlines() if line.strip()]
            
            if not lines:
                continue

            if is_first_cell and lines[0] == main_title:
                lines = lines[1:]
                is_first_cell = False
                if not lines:
                    continue
            else:
                is_first_cell = False

            if any("summary" in l.lower() for l in lines) or any("performance summary" in l.lower() for l in lines):
                summary_content = "\n".join(lines)
                rows.append({"type": "SUMMARY", "content": summary_content})
                continue

            # Condition 1: Single line step / title cell
            if len(lines) == 1:
                line = lines[0]
                is_step = bool(self.STEP_PREFIX_REGEX.match(line))
                
                if not is_step and (line.isupper() or len(line.split()) <= 4):
                    rows.append({"type": "TITLE", "content": line})
                else:
                    rows.append({"type": "STEP", "content": line})
                    rows.append({"type": "RATIONALE", "content": ""})

            # Condition 2: Multi-line / Indented cell block
            else:
                first_line = lines[0]
                if self.STEP_PREFIX_REGEX.match(first_line):
                    rows.append({"type": "STEP", "content": first_line})
                else:
                    rows.append({"type": "TITLE", "content": first_line})

                for sub_line in lines[1:]:
                    rows.append({"type": "STEP", "content": sub_line})

                rows.append({"type": "RATIONALE", "content": ""})

        rows.append({"type": "CITATIONS", "content": ""})
        rows.append({"type": "VERSION", "content": ""})

        return main_title, rows

    def generate_pdf(self, main_title: str, rows: List[Dict[str, Any]], dest_file_path: Path):
        """Generates the clean reformatted PDF table grid."""
        doc = SimpleDocTemplate(
            str(dest_file_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        main_title_style = ParagraphStyle(
            'MainTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.black,
            spaceAfter=10
        )

        cell_title_style = ParagraphStyle(
            'CellTitle',
            fontName='Helvetica-Bold',
            fontSize=9.5,
            leading=12,
            textColor=colors.black
        )

        cell_text_style = ParagraphStyle(
            'CellText',
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#111111')
        )

        story = [Paragraph(f"<b>TITLE:</b> {main_title}", main_title_style), Spacer(1, 6)]
        table_data = []

        for item in rows:
            itype = item["type"]
            content = item["content"]

            if itype == "TITLE":
                p = Paragraph(f"<b>TITLE:</b> {content}", cell_title_style)
                table_data.append([p])
            elif itype == "STEP":
                p = Paragraph(f"<b>STEP:</b> {content}", cell_text_style)
                table_data.append([p])
            elif itype == "RATIONALE":
                p = Paragraph("<b>RATIONALE:</b> " + content, cell_text_style)
                table_data.append([p])
            elif itype == "SUMMARY":
                formatted_summary = content.replace("\n", "<br/>")
                p = Paragraph(f"<b>SUMMARY:</b> {formatted_summary}", cell_text_style)
                table_data.append([p])
            elif itype == "CITATIONS":
                table_data.append([Paragraph("<b>CITATIONS:</b>", cell_title_style)])
            elif itype == "VERSION":
                table_data.append([Paragraph("<b>VERSION:</b>", cell_title_style)])

        if table_data:
            main_table = Table(table_data, colWidths=[540])
            main_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.75, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(main_table)

        doc.build(story)

    def process(self):
        """Processes input file or directory into existing output folder."""
        if not self.input_path.exists():
            print(f"Error: Input path '{self.input_path}' does not exist.")
            return

        if not self.output_folder.exists() or not self.output_folder.is_dir():
            print(f"Error: Output folder '{self.output_folder}' does not exist. Please check folder path.")
            return

        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".pdf":
                print(f"Error: Path '{self.input_path}' is not a PDF file.")
                return

            out_file_name = f"{self.input_path.stem}_cleaned.pdf"
            dest_file_path = self.output_folder / out_file_name

            print(f"Processing single PDF: '{self.input_path.name}'...")
            title, rows = self.parse_pdf(self.input_path)
            self.generate_pdf(title, rows, dest_file_path)
            print(f"  ✓ Saved clean table PDF to: '{dest_file_path}'\n")
            return

        pdf_files = [f for f in self.input_path.rglob("*.pdf") if not f.name.endswith("_cleaned.pdf")]

        if not pdf_files:
            print(f"No original PDF files found in '{self.input_path}'.")
            return

        print(f"Found {len(pdf_files)} PDF file(s) in '{self.input_path}'.\n")

        for pdf_file in pdf_files:
            out_file_name = f"{pdf_file.stem}_cleaned.pdf"
            dest_file_path = self.output_folder / out_file_name

            print(f"Processing: {pdf_file.name} ...")
            title, rows = self.parse_pdf(pdf_file)
            self.generate_pdf(title, rows, dest_file_path)
            print(f"  ✓ Saved to: '{dest_file_path}'\n")

        print("Done processing all files!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdf_reformatter.py <input_file_or_folder> <existing_destination_folder>")
        sys.exit(1)

    reformatter = CleanPDFReformatter(sys.argv[1], sys.argv[2])
    reformatter.process()