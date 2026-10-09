"""Clinical Procedure PDF Reformatter & Intelligent Two-Pass Indentation Parser.

Adheres to object-oriented programming, clean code standards, and implements 
precise indentation-based step grouping and continuation wrapping.
"""

import argparse
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass
class ProcedureDocumentSchema:
    """Structured representation of a clinical procedure document for JSON/API transport."""
    main_title: str
    elements: List[Dict[str, str]] = field(default_factory=list)
    summary: Optional[str] = None
    citations: Optional[str] = None
    version: Optional[str] = None


class BaseTextExtractor(ABC):
    """Abstract interface for document text extraction."""

    @abstractmethod
    def extract_lines(self, file_path: Path) -> List[Tuple[str, float, bool]]:
        """Extracts text lines with horizontal coordinate (x0) and font-weight metadata."""
        pass


class LocalPDFTextExtractor(BaseTextExtractor):
    """Local extractor using pdfplumber with precise X-coordinate and font inspection."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [DEBUG EXTRACTOR] {message}", flush=True)

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """Filters out form fields, checkboxes, signatures, and column header bleeding."""
        lower_line = line.lower()
        
        noise_triggers = [
            "name:", "date:", "page ", "revised", "for educational use only",
            "faculty signature", "student signature", "time completed", 
            "start time", "end time", "comments:", "s/u", "✓", "✔"
        ]
        
        if any(trigger in lower_line for trigger in noise_triggers):
            return True
            
        # Filter table column artifacts like standalone ordinals, checkmarks, or symbols
        if re.fullmatch(r"\b(\d+(st|nd|rd|th)|nd|rd|st|th)\b", lower_line.strip()):
            return True

        if re.fullmatch(r"[✓✔\s_\-/\\]+", line):
            return True
            
        return False

    @staticmethod
    def _clean_text_debris(text: str) -> str:
        """Removes stray table column headers and checkboxes embedded in text strings."""
        text = re.sub(r"\b(\d+(st|nd|rd|th)|nd|rd|st|th)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[✓✔]", "", text)
        return re.sub(r"[ \t]+", " ", text).strip()

    def extract_lines(self, file_path: Path) -> List[Tuple[str, float, bool]]:
        extracted_data = []
        try:
            self._log(f"Opening PDF with pdfplumber: {file_path.name}")
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                self._log(f"Total pages detected: {total_pages}")
                
                for idx, page in enumerate(pdf.pages, start=1):
                    self._log(f"--- Processing Page {idx} of {total_pages} ---")
                    
                    words = page.extract_words(keep_blank_chars=False, x_tolerance=2, y_tolerance=3, extra_attrs=["fontname"])
                    if not words:
                        continue

                    lines_dict = []
                    for word in words:
                        top_y = round(word["top"] / 4) * 4
                        found = False
                        for l_item in lines_dict:
                            if abs(l_item["y"] - top_y) <= 3:
                                l_item["words"].append(word)
                                found = True
                                break
                        if not found:
                            lines_dict.append({"y": top_y, "words": [word]})

                    lines_dict.sort(key=lambda x: x["y"])

                    for l_item in lines_dict:
                        line_words = sorted(l_item["words"], key=lambda w: w["x0"])
                        x0_coord = line_words[0]["x0"] if line_words else 0.0
                        
                        is_bold = any("bold" in str(w.get("fontname", "")).lower() for w in line_words)

                        line_str = " ".join([w["text"] for w in line_words])
                        cleaned_line = self._clean_text_debris(line_str)
                        cleaned_line = cleaned_line.rstrip(" _-/\\")

                        if not cleaned_line:
                            continue

                        if self._is_noise_line(cleaned_line):
                            continue
                        else:
                            extracted_data.append((cleaned_line, x0_coord, is_bold))

        except Exception as err:
            raise RuntimeError(f"Local extraction failed for {file_path.name}: {err}") from err
            
        return extracted_data


class ProcedureParserService:
    """Two-pass intelligent parser service managing strict indentation hierarchy and step wrapping."""

    STEP_PREFIX_REGEX = re.compile(
        r"^(\(?\d+[\.\)]|\(?[a-zA-Z][\.\)]|\(?[ivxLCDM]+[\.\)])\s*(.*)",
        re.IGNORECASE,
    )

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [DEBUG PARSER] {message}", flush=True)

    def parse_stream_to_schema(self, line_stream: List[Tuple[str, float, bool]]) -> ProcedureDocumentSchema:
        """Executes a two-pass sweep: Pass 1 tags structural roles, Pass 2 evaluates precise indentation wrappers."""
        if not line_stream:
            return ProcedureDocumentSchema(main_title="Clinical Procedure")

        main_title = "Physical Assessment"
        swept_lines = []
        summary_lines = []
        citations_lines = []
        version_lines = []

        parsing_state = "BODY"
        first_line = True

        # --- PASS 1: STRUCTURAL SWEEP & INITIAL CLASSIFICATION ---
        for text, x0, is_bold in line_stream:
            lower_text = text.lower()

            if first_line:
                main_title = re.sub(r"\s*\([^)]*\)", "", text).strip()
                first_line = False
                continue

            if "performance summary" in lower_text or "summary measures" in lower_text:
                parsing_state = "SUMMARY"
                summary_lines.append(text)
                continue
            elif "citation" in lower_text:
                parsing_state = "CITATIONS"
                citations_lines.append(text)
                continue
            elif "version" in lower_text:
                parsing_state = "VERSION"
                version_lines.append(text)
                continue

            if parsing_state == "SUMMARY":
                summary_lines.append(text)
                continue
            elif parsing_state == "CITATIONS":
                citations_lines.append(text)
                continue
            elif parsing_state == "VERSION":
                version_lines.append(text)
                continue

            has_prefix = bool(self.STEP_PREFIX_REGEX.match(text))
            is_all_caps = text.isupper() and len(text.split()) > 1

            # Titles are explicitly bolded or short all-caps headers at the left margin
            if is_bold or (is_all_caps and x0 < 60 and not has_prefix and len(text.split()) <= 6):
                line_type = "TITLE"
            else:
                line_type = "CANDIDATE_STEP"

            swept_lines.append({"text": text, "type": line_type, "x0": x0})
            self._log(f"Sweep Pass 1 [{line_type}] (x0={x0:.1f}, bold={is_bold}): {text[:50]}")

        # --- PASS 2: INDENTATION HIERARCHY & WRAPPER ASSEMBLY ---
        elements = []
        current_step_content = None
        baseline_step_x0 = None

        for item in swept_lines:
            ltype = item["type"]
            text = item["text"]
            x0 = item["x0"]

            if ltype == "TITLE":
                # Finalize any pending step before switching to a new title block
                if current_step_content:
                    elements.append({"type": "STEP", "content": current_step_content})
                    elements.append({"type": "RATIONALE", "content": ""})
                    current_step_content = None
                
                baseline_step_x0 = None  # Reset indentation baseline for new section
                elements.append({"type": "TITLE", "content": text})
            else:
                # Check for explicit numbered prefix
                match = self.STEP_PREFIX_REGEX.match(text)
                if match:
                    if current_step_content:
                        elements.append({"type": "STEP", "content": current_step_content})
                        elements.append({"type": "RATIONALE", "content": ""})
                    current_step_content = match.group(2).strip()
                    baseline_step_x0 = x0
                    continue

                # Establish baseline indentation for unnumbered lists if not set
                if baseline_step_x0 is None:
                    baseline_step_x0 = x0

                # Indentation Rule: If x0 matches baseline (within 6 pts tolerance), it's a new step.
                # If x0 is significantly larger (> baseline + 6), it's a continuation line.
                if abs(x0 - baseline_step_x0) <= 6:
                    if current_step_content:
                        elements.append({"type": "STEP", "content": current_step_content})
                        elements.append({"type": "RATIONALE", "content": ""})
                    current_step_content = text
                else:
                    # Deeper indentation -> wrap and append to current step
                    if current_step_content:
                        current_step_content += f" {text}"
                    else:
                        current_step_content = text

        # Finalize any trailing step
        if current_step_content:
            elements.append({"type": "STEP", "content": current_step_content})
            elements.append({"type": "RATIONALE", "content": ""})

        summary_full = "<br/>".join(summary_lines) if summary_lines else None
        citations_full = "<br/>".join(citations_lines) if citations_lines else None
        version_full = "<br/>".join(version_lines) if version_lines else None

        return ProcedureDocumentSchema(
            main_title=main_title,
            elements=elements,
            summary=summary_full,
            citations=citations_full,
            version=version_full,
        )


class PDFReportGenerator:
    """ReportLab PDF rendering engine enforcing strict alternating table formats."""

    def __init__(self, dest_file_path: Path, verbose: bool = False) -> None:
        self.dest_file_path = dest_file_path
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [DEBUG GENERATOR] {message}", flush=True)

    def generate(self, schema: ProcedureDocumentSchema) -> None:
        self._log(f"Building output PDF at: {self.dest_file_path}")
        self.dest_file_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(self.dest_file_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        main_title_style = ParagraphStyle(
            "MainTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.black,
            spaceAfter=10,
        )

        cell_title_style = ParagraphStyle(
            "CellTitle",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12,
            textColor=colors.black,
        )

        cell_text_style = ParagraphStyle(
            "CellText",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#111111"),
        )

        story = [Paragraph(f"<b>TITLE:</b> {schema.main_title}", main_title_style), Spacer(1, 6)]
        table_data = []

        for el in schema.elements:
            etype = el["type"]
            content = el["content"]

            if etype == "TITLE":
                table_data.append([Paragraph(f"<b>TITLE:</b> {content}", cell_title_style)])
            elif etype == "STEP":
                table_data.append([Paragraph(f"<b>STEP:</b> {content}", cell_text_style)])
            elif etype == "RATIONALE":
                table_data.append([Paragraph("<b>RATIONALE:</b>", cell_text_style)])

        if schema.summary:
            table_data.append([Paragraph(f"<b>SUMMARY:</b><br/>{schema.summary}", cell_text_style)])
        else:
            table_data.append([Paragraph("<b>SUMMARY:</b>", cell_title_style)])

        citations_text = schema.citations if schema.citations else ""
        table_data.append([Paragraph(f"<b>CITATIONS:</b> {citations_text}", cell_title_style)])

        version_text = schema.version if schema.version else ""
        table_data.append([Paragraph(f"<b>VERSION:</b> {version_text}", cell_title_style)])

        if table_data:
            main_table = Table(table_data, colWidths=[540])
            main_table.setStyle(
                TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(main_table)

        doc.build(story)
        self._log("PDF build completed successfully.")


class ApplicationController:
    """Orchestrator pipeline managing extraction, parsing, and ReportLab generation."""

    def __init__(self, extractor: BaseTextExtractor, parser: ProcedureParserService, verbose: bool = False) -> None:
        self.extractor = extractor
        self.parser = parser
        self.verbose = verbose

    def process_file(self, input_file: Path, output_folder: Path) -> Path:
        output_folder.mkdir(parents=True, exist_ok=True)
        dest_path = output_folder / f"{input_file.stem}_cleaned.pdf"

        raw_stream = self.extractor.extract_lines(input_file)
        schema = self.parser.parse_stream_to_schema(raw_stream)
        
        generator = PDFReportGenerator(dest_path, verbose=self.verbose)
        generator.generate(schema)
        
        return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Procedure Reformatter & JSON Schema Generator Pipeline.")
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).parent / "TODO_CleanThese", help="Path to input PDF file or directory.")
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).parent / "reformatted_cleaned", help="Path to output directory.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")

    args = parser.parse_args()

    extractor = LocalPDFTextExtractor(verbose=args.verbose)
    parser_service = ProcedureParserService(verbose=args.verbose)
    controller = ApplicationController(extractor=extractor, parser=parser_service, verbose=args.verbose)

    try:
        if args.input.is_file():
            if args.input.suffix.lower() != ".pdf":
                print(f"❌ Error: '{args.input}' is not a valid PDF file.")
                sys.exit(1)
            print(f"📄 Processing single PDF: '{args.input.name}'...", flush=True)
            out = controller.process_file(args.input, args.output)
            print(f"✨ Done! Saved to: '{out}'", flush=True)
        elif args.input.is_dir():
            print(f"📂 Scanning directory: '{args.input}'...", flush=True)
            pdf_files = [f for f in args.input.rglob("*.pdf") if not f.name.endswith("_cleaned.pdf")]
            for pdf_file in pdf_files:
                print(f"🔄 Processing: {pdf_file.name} ...", flush=True)
                out_path = controller.process_file(pdf_file, args.output)
                print(f"  ✓ Saved to: '{out_path}'\n", flush=True)
            print("✨ Done processing all files!", flush=True)
        else:
            print(f"❌ Error: Path '{args.input}' does not exist.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user (Ctrl + C). Exiting safely...", flush=True)
        sys.exit(0)