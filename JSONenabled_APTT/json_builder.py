"""786 JSON Config Builder module for procedure templates.

Parses formatted text definitions of procedures, extracts metadata,
sections, steps, and optional keywords, and serializes them into structured JSON files
stored securely in the local `configs/` directory.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


def create_configs_directory() -> Path:
    """Creates the 'configs' directory if it does not already exist.

    Returns:
        Path: Path object representing the verified configs directory.
    """
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def extract_metadata_field(pattern: str, text: str, default: str = "N/A") -> str:
    """Extracts a metadata field value from text using a regular expression.

    Args:
        pattern (str): The regex pattern matching the desired metadata field.
        text (str): The raw text to search within.
        default (str): Fallback string if no match is found. Defaults to "N/A".

    Returns:
        str: Cleaned extracted text or the default value.
    """
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default


def parse_keywords(step_text: str) -> Dict[str, List[str]]:
    """Parses keyword groups (L1, L2, L3) from a step's text block.

    Args:
        step_text (str): Raw text block corresponding to a single step.

    Returns:
        Dict[str, List[str]]: Dictionary mapping level tags (e.g., 'L1')
            to lists of clean keyword strings.
    """
    keywords: Dict[str, List[str]] = {"L1": [], "L2": [], "L3": []}
    kw_matches = re.findall(r"(L[1-3]):\s*\[(.*?)\]", step_text, re.IGNORECASE)

    for level, kw_str in kw_matches:
        level_key = level.upper()
        parsed_list = [k.strip().lower() for k in kw_str.split(",") if k.strip()]
        if level_key in keywords:
            keywords[level_key].extend(parsed_list)

    return keywords


def parse_formatted_text_to_json(raw_text: str, doc_id: str) -> Optional[Path]:
    """Parses formatted procedure text into structured JSON and writes it to disk.

    Args:
        raw_text (str): Raw formatted template string containing sections and steps.
        doc_id (str): Identifier used to create the destination file (`doc_id.json`).

    Returns:
        Optional[Path]: Path to the written JSON configuration file, or None if failed.
    """
    # Secure sanitization of doc_id to prevent path traversal attacks
    safe_doc_id = re.sub(r"[^\w\-]", "_", doc_id.strip())
    if not safe_doc_id:
        safe_doc_id = "default_procedure"

    config_dir = create_configs_directory()
    output_filepath = (config_dir / f"{safe_doc_id}.json").resolve()

    # Ensure final resolved path strictly stays within configs directory
    if not output_filepath.is_relative_to(config_dir.resolve()):
        raise ValueError("Security Violation: Invalid file destination path.")

    title = extract_metadata_field(r"Title:\s*(.+)", raw_text, "Procedure Assessment")
    summary = extract_metadata_field(r"Summary:\s*(.+)", raw_text)
    citations = extract_metadata_field(r"Citations:\s*(.+)", raw_text)
    version = extract_metadata_field(r"Version:\s*(.+)", raw_text, "1.0")

    sections: List[Dict[str, Any]] = []
    section_blocks = re.split(r"(?:^|\n)Section\s*\d+:\s*", raw_text, flags=re.IGNORECASE)

    step_counter = 1

    for block in section_blocks:
        if not block.strip():
            continue

        lines = block.strip().split("\n")
        section_title = lines[0].strip()

        steps: List[Dict[str, Any]] = []
        step_blocks = re.split(r"(?:^|\n)Step\s*\d+:\s*", "\n".join(lines[1:]), flags=re.IGNORECASE)

        for s_block in step_blocks:
            if not s_block.strip():
                continue

            s_lines = s_block.strip().split("\n")
            step_desc = s_lines[0].strip()

            rationale_match = re.search(r"Rationale:\s*(.+)", s_block, re.IGNORECASE)
            rationale = rationale_match.group(1).strip() if rationale_match else "N/A"

            step_keywords = parse_keywords(s_block)

            steps.append({
                "step_id": step_counter,
                "desc": step_desc,
                "rationale": rationale,
                "keywords": step_keywords
            })
            step_counter += 1

        if steps:
            sections.append({
                "section_title": section_title,
                "steps": steps
            })

    config_data = {
        "doc_id": safe_doc_id,
        "title": title,
        "summary": summary,
        "citations": citations,
        "version": version,
        "sections": sections
    }

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        print(f"[JSON BUILDER] Configuration successfully written to: {output_filepath}")
        return output_filepath
    except (OSError, TypeError) as err:
        print(f"[JSON BUILDER ERROR] Failed to write configuration file: {err}")
        return None


if __name__ == "__main__":
    sample_text = """
    Title: Activated Partial Thromboplastin Time (APTT) Standard Procedure
    Summary: Automated coagulation procedure for measuring clotting times.
    Citations: Clinical Laboratory Standards Institute (CLSI) Guidelines H21-A5.
    Version: 2.1

    Section 1: Preparation & Reagent Setup
    Step 1: Check reagent temperature and confirm 37°C incubation.
    Rationale: Enzymes require steady biological temperature for reliable kinetics.
    L1: [check, reagent, temperature]
    L2: [confirm, incubation, 37]

    Step 2: Load plasma sample into the analyzer carousel.
    Rationale: Ensures proper aspiration sequence by automated fluidics.
    L1: [load, plasma, sample]
    L2: [analyzer, carousel]
    """
    
    parse_formatted_text_to_json(sample_text, "APTT_PARSED_TEMPLATE")