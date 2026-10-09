"""Clinical Procedure PDF Reformatter & Intelligent Schema Parser.

Optimized with safe string operations to eliminate regex backtracking hangs.
Adheres to object-oriented programming principles and strict alternating table schemas.
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
    """Structured data representation of a clinical procedure document."""
    main_title: str
    elements: List[Dict[str, str]] = field(default_factory=list)
    summary: Optional[str] = None
    citations: Optional[str] = None
    version: Optional[str] = None


class BaseTextExtractor(ABC):
    """Abstract interface for document text extraction."""

    @abstractmethod
    def extract_lines(self, file_path: Path) -> List[Tuple[str, bool]]:
        pass


class LocalPDFTextExtractor(BaseTextExtractor):
    """Local file extractor utilizing pdfplumber with safe string processing."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [DEBUG EXTRACTOR] {message}", flush=True)

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """Filters out form fields, signatures, dates, and educational headers/footers."""
        lower_line = line.lower()
        
        noise_triggers = [
            "name:", "date:", "page ", "revised", "for educational use only",
            "s/u", "2nd", "3rd", "faculty signature", "student signature",
            "time completed", "start time", "end time", "comments:", "vital signs"
        ]
        
        if any(trigger in lower_line for trigger in noise_triggers):
            return True
            
        # Filter lines consisting only of underscores, slashes, or dashes
        if all(char in " _-/\\" for char in line):
            return True
            
        return False

    def extract_lines(self, file_path: Path) -> List[Tuple[str, bool]]:
        extracted_data = []
        try:
            self._log(f"Opening PDF with pdfplumber: {file_path.name}")
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                self._log(f"Total pages detected: {total_pages}")
                
                for idx, page in enumerate(pdf.pages, start=1):
                    self._log(f"--- Starting extraction on Page {idx} of {total_pages} ---")
                    
                    # Extract words with coordinates for robust spacing and indentation
                    words = page.extract_words(keep_blank_chars=False, x_tolerance=2, y_tolerance=3)
                    if not words:
                        self._log(f"No words found on page {idx}.")
                        continue

                    self._log(f"Found {len(words)} raw words on page {idx}. Grouping lines...")
                    
                    # Group words into horizontal lines by 'top' coordinate
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
                        first_word_x0 = line_words[0]["x0"] if line_words else 0
                        is_indented = first_word_x0 > 50  # Indentation threshold

                        line_str = " ".join([w["text"] for w in line_words])
                        
                        # Safe cleaning using built-in methods (no backtracking risk)
                        cleaned_line = line.strip() if 'line' in locals() else line_str
                        cleaned_line = line_str.strip().rstrip(" _-/\\")

                        if not cleaned_line:
                            continue

                        if self._is_noise_line(cleaned_line):
                            self._log(f"  [FILTERED NOISE]: '{cleaned_line}'")
                        else:
                            self._log(f"  [KEPT LINE] (Indented={is_indented}): '{cleaned_line}'")
                            extracted_data.append((cleaned_line, is_indented))
                    
                    self._log(f"--- Finished Page {idx}. Total valid lines: {len(extracted_data)} ---")

        except Exception as err:
            raise RuntimeError(f"Local extraction failed for {file_path.name}: {err}") from err
            
        return extracted_data


class ProcedureParserService:
    """Intelligent classification service mapping raw text lines to procedure schemas."""

    STEP_PREFIX_REGEX = re.compile(
        r"^(\(?\d+[\.\)]|\(?[a-zA-Z][\.\)]|\(?[ivxLCDM]+[\.\)])\s*(.*)",
        re.IGNORECASE,
    )

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [DEBUG PARSER] {message}", flush=True)

    def parse_stream_to_schema(self, line_stream: List[Tuple[str, bool]]) -> ProcedureDocumentSchema:
        if not line_stream:
            return ProcedureDocumentSchema(main_title="Clinical Procedure")

        main_title = "Physical Assessment"
        elements = []
        summary_lines = []
        citations_lines = []
        version_lines = []

        parsing_state = "BODY"
        first_line = True

        for text, is_indented in line_stream:
            lower_text = text.lower()

            if first_line:
                main_title = re.sub(r"\s*\([^)]*\)", "", text).strip()
                self._log(f"Identified Main Title: '{main_title}'")
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

            is_all_caps = text.isupper() and len(text.split()) > 1
            is_bold_candidate = len(text.split()) <= 6 and not text.endswith(".") and not self.STEP_PREFIX_REGEX.match(text)
            has_prefix = bool(self.STEP_PREFIX_REGEX.match(text))

            if (is_all_caps or is_bold_candidate) and not is_indented and not has_prefix:
                self._log(f"Classified TITLE: '{text}'")
                elements.append({"type": "TITLE", "content": text})
            elif has_prefix or is_indented or len(text.split()) > 3:
                step_content = text
                match = self.STEP_PREFIX_REGEX.match(text)
                if match:
                    step_content = match.group(2).strip()

                self._log(f"Classified STEP: '{step_content[:40]}...'")
                elements.append({"type": "STEP", "content": step_content})
                elements.append({"type": "RATIONALE", "content": ""})
            else:
                self._log(f"Classified TITLE (Fallback): '{text}'")
                elements.append({"type": "TITLE", "content": text})

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