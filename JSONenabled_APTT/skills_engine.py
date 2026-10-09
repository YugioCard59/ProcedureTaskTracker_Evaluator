"""786 Clinical Skills Assessment Engine.

Provides continuous speech-to-text evaluation for nursing procedure assessments.
Features bottom-positioned metadata accordions, anonymized feedback, and robust segment parsing.
"""
#To run: python skills_engine.py -c "Safety_Measures_All_Skills.json"

import argparse
from dataclasses import dataclass
from datetime import datetime
import html
import json
from pathlib import Path
import queue
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import webbrowser
import zipfile

import numpy as np
import sounddevice as sd
import torch
import whisper

try:
    from silero_vad import load_silero_vad

    HAS_SILERO = True
except ImportError:
    HAS_SILERO = False


NURSING_INITIAL_PROMPT = (
    "Verifying healthcare provider orders and hospital policy. "
    "Gathering equipment, checking expiration dates. Performing hand hygiene, "
    "putting on PPE, foam in. Considering environment safety. Providing privacy. "
    "Explaining procedure, identifying patient using two identifiers. Assessing allergies. "
    "Checking pain, potty, possessions, position. Raising two side rails. Call bell within reach."
)


@dataclass
class UtteranceEntry:
    """Represents a single captured utterance with a timestamp."""

    timestamp: str
    text: str


def clean_text_for_matching(text: str) -> str:
    """Strips punctuation, hyphens, and normalizes STT phonetic variations for keyword matching."""
    cleaned = text.lower()
    cleaned = re.sub(r"[-_]", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)

    phonetic_map = {
        r"\bside\s+rolls\b": "side rails",
        r"\bpaying\b": "pain",
        r"\bphom\b": "foam",
        r"\bfoam\s+into\b": "foam in",
    }
    for pat, repl in phonetic_map.items():
        cleaned = re.sub(pat, repl, cleaned)

    return " ".join(cleaned.split())


def normalize_transcript(text: str) -> str:
    """Normalizes transcript text by converting to lower case and stripping excess space."""
    return " ".join(text.lower().split())


def get_prioritized_input_device() -> Tuple[Optional[int], str]:
    """Detects and returns the system's default audio input device."""
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            dev_info = sd.query_devices(default_in, "input")
            return default_in, str(dev_info.get("name", "Default Input"))
    except Exception:
        pass
    return None, "Default System Microphone"


def extract_normalized_steps(config_data: Any) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Extracts metadata dictionary and unified procedure steps list regardless of input schema layout."""
    metadata = {
        "title": "Clinical Procedure Assessment",
        "summary": "N/A",
        "citations": "N/A",
        "version": "1.0",
    }
    extracted_steps: List[Dict[str, Any]] = []

    if isinstance(config_data, dict):
        metadata["title"] = (
            config_data.get("title") or config_data.get("procedure_title") or metadata["title"]
        )
        metadata["summary"] = config_data.get("summary", "N/A")
        metadata["citations"] = config_data.get("citations", "N/A")
        metadata["version"] = config_data.get("version", "1.0")

        if "steps" in config_data and isinstance(config_data["steps"], list) and config_data["steps"]:
            extracted_steps = config_data["steps"]
        elif "sections" in config_data and isinstance(config_data["sections"], list):
            for sec in config_data["sections"]:
                if isinstance(sec, dict) and "steps" in sec:
                    extracted_steps.extend(sec["steps"])

    elif isinstance(config_data, list):

        def flatten(item):
            if isinstance(item, list):
                for sub in item:
                    flatten(sub)
            elif isinstance(item, dict):
                if "steps" in item and isinstance(item["steps"], list):
                    extracted_steps.extend(item["steps"])
                else:
                    extracted_steps.append(item)

        flatten(config_data)

    return metadata, extracted_steps


def format_rationale_html(raw_rationale: str) -> str:
    """Formats clinical rationale with line spacing and highlighted citation references."""
    if not raw_rationale or raw_rationale.strip() == "N/A":
        return "<em>Standard clinical rationale.</em>"

    items = re.split(r"(?=\b\d+\.\s+)", raw_rationale.strip())
    formatted_blocks = []

    for item in items:
        clean_item = item.strip()
        if not clean_item:
            continue

        escaped = html.escape(clean_item)

        highlight_pattern = r"(\b(?:NCSBN[^\n→–—]*|CDC[^\n→–—]*|The Joint Commission[^\n→–—]*|CLSI[^\n→–—]*)\b)"
        escaped = re.sub(
            highlight_pattern,
            r'<span class="citation-tag">\1</span>',
            escaped,
            flags=re.IGNORECASE,
        )

        escaped = re.sub(
            r"^(\d+\.\s+[^→\:\n]+[→\:])",
            r"<strong>\1</strong>",
            escaped,
        )

        formatted_blocks.append(f'<div class="rationale-item">{escaped}</div>')

    return "".join(formatted_blocks)


class VADAudioTranscriber:
    """Captures continuous microphone audio, performs Voice Activity Detection (VAD),
    and transcribes speech utterances using local OpenAI Whisper STT.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_duration_ms: int = 800,
        initial_prompt: Optional[str] = None,
        model_size: str = "base",
        max_buffer_seconds: float = 12.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_duration_ms = silence_duration_ms
        self.initial_prompt = initial_prompt
        self.is_streaming = False

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.speech_buffer: List[np.ndarray] = []
        self.silent_chunks_count: int = 0
        self.min_speech_samples: int = int(self.sample_rate * 0.4)
        self.max_speech_samples: int = int(self.sample_rate * max_buffer_seconds)

        self.device_index, self.device_name = get_prioritized_input_device()

        print("⏳ Initializing VAD engine...")
        self.vad_model = None
        if HAS_SILERO:
            try:
                self.vad_model = load_silero_vad()
                print("✅ Silero VAD loaded successfully.")
            except Exception as err:
                print(f"⚠️ Local VAD load failed ({err}). Falling back to RMS energy mode.")
        else:
            print("ℹ️ Silero VAD not found. Using RMS energy-based activity detection.")

        print(f"⏳ Loading local Whisper model ('{model_size}')...")
        self.whisper_model = whisper.load_model(model_size)
        print("✅ Whisper model loaded.")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"⚠️ Audio callback status notice: {status}", file=sys.stderr)
        if self.is_streaming:
            self.audio_queue.put(indata.copy())

    def start_stream(self) -> None:
        self.is_streaming = True
        self.clear_buffers()
        block_size = int(self.sample_rate * 0.1)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=self._audio_callback,
            )
            self.stream.start()
            print(f"🎙️ Active listening started on [{self.device_name}]...\n")
        except Exception as err:
            print(f"❌ Failed to start microphone audio stream: {err}")
            self.is_streaming = False

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        if self.vad_model is not None:
            try:
                tensor_chunk = torch.from_numpy(audio_chunk)
                speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()
                return speech_prob > 0.5
            except Exception:
                pass

        rms_energy = float(np.sqrt(np.mean(audio_chunk**2)))
        return rms_energy > 0.015

    def get_speech_utterance(self, timeout: float = 0.5) -> str:
        if not self.is_streaming:
            return ""

        try:
            raw_data = self.audio_queue.get(timeout=timeout)
            flat_chunk = raw_data.flatten()

            if self._is_speech(flat_chunk):
                self.speech_buffer.append(flat_chunk)
                self.silent_chunks_count = 0
            elif len(self.speech_buffer) > 0:
                self.silent_chunks_count += 1
                self.speech_buffer.append(flat_chunk)

                max_silent_chunks = int(self.silence_duration_ms / 100)
                if self.silent_chunks_count >= max_silent_chunks:
                    return self._transcribe_buffered_speech()

            current_buffer_samples = sum(len(c) for c in self.speech_buffer)
            if current_buffer_samples >= self.max_speech_samples:
                return self._transcribe_buffered_speech()

        except queue.Empty:
            pass

        return ""

    def _transcribe_buffered_speech(self) -> str:
        if not self.speech_buffer:
            return ""

        complete_audio = np.concatenate(self.speech_buffer)
        self.speech_buffer.clear()
        self.silent_chunks_count = 0

        if len(complete_audio) < self.min_speech_samples:
            return ""

        result = self.whisper_model.transcribe(
            complete_audio,
            initial_prompt=self.initial_prompt,
            fp16=False,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )

        raw_text = result.get("text", "").strip()
        cleaned_text = re.sub(r"\.{2,}", "", raw_text).strip()
        return cleaned_text

    def flush_final_speech(self) -> str:
        return self._transcribe_buffered_speech()

    def clear_buffers(self) -> None:
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.speech_buffer.clear()
        self.silent_chunks_count = 0

    def stop_stream(self) -> None:
        self.is_streaming = False
        if hasattr(self, "stream") and self.stream.active:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as err:
                print(f"⚠️ Exception closing stream: {err}")
        print("🛑 Active listening stopped. Microphone hardware stream closed.")


class ProcedureStepEvaluator:
    """Evaluates procedure step execution using strict 'NEXT' segment boundaries."""

    def __init__(self, procedure_steps: List[Dict[str, Any]]) -> None:
        self.procedure_steps = procedure_steps
        self.audit_results: List[Dict[str, Any]] = []

    def process_pass_two(self, transcript_buffer: List[UtteranceEntry]) -> None:
        """Splits transcript strictly by 'NEXT' keywords and evaluates each step strictly against its segment."""
        if not transcript_buffer:
            self._mark_all_skipped(fallback_time=datetime.now().strftime("%H:%M:%S"))
            return

        full_text = " ".join([entry.text for entry in transcript_buffer])
        normalized_text = normalize_transcript(full_text)

        raw_segments = re.split(r"\bnext\b", normalized_text, flags=re.IGNORECASE)
        segments = [s.strip() for s in raw_segments if s.strip()]

        self.audit_results.clear()

        for idx, step_config in enumerate(self.procedure_steps):
            step_num = step_config.get("step_id") or step_config.get("step_num") or (idx + 1)
            step_desc = step_config.get("desc", f"Step {step_num}")
            step_rationale = step_config.get("rationale", "Standard clinical rationale.")

            if idx < len(segments):
                assigned_segment = segments[idx]
                timestamp = transcript_buffer[min(idx, len(transcript_buffer) - 1)].timestamp
            else:
                assigned_segment = ""
                timestamp = transcript_buffer[-1].timestamp if transcript_buffer else "00:00:00"

            if assigned_segment:
                eval_outcome = self._evaluate_rubric_levels(step_config, assigned_segment)

                if eval_outcome["status"] == "COMPLETE":
                    status = "COMPLETE"
                    feedback = eval_outcome["feedback"]
                else:
                    matched_other_seg = False
                    for other_idx, other_seg in enumerate(segments):
                        if other_idx != idx:
                            other_eval = self._evaluate_rubric_levels(step_config, other_seg)
                            if other_eval["status"] == "COMPLETE":
                                matched_other_seg = True
                                break

                    if matched_other_seg:
                        status = "OUT OF ORDER"
                        feedback = (
                            "Step keywords spoken out of sequence. " + eval_outcome["feedback"]
                        )
                    else:
                        status = "INCOMPLETE"
                        feedback = eval_outcome["feedback"]
            else:
                status = "SKIPPED"
                feedback = "Step skipped or no speech recorded for this segment."

            self.audit_results.append({
                "step_num": step_num,
                "desc": step_desc,
                "rationale": step_rationale,
                "status": status,
                "timestamp": timestamp,
                "feedback": feedback,
            })

    def _evaluate_rubric_levels(self, step_config: Dict[str, Any], text: str) -> Dict[str, str]:
        """Evaluates step and produces secure, anonymized feedback showing sample example phrases without exposing levels."""
        clean_spoken = clean_text_for_matching(text)
        keywords_field = step_config.get("keywords") or step_config.get("levels") or {}

        missed_rationales: List[str] = []

        if isinstance(keywords_field, dict) and keywords_field:
            satisfied = 0
            active_levels = {k: v for k, v in keywords_field.items() if v}
            total = len(active_levels)

            if total == 0:
                return {"status": "COMPLETE", "feedback": "All required criteria met for this step."}

            for level_name, kw_group in active_levels.items():
                if isinstance(kw_group, str):
                    kw_group = [kw_group]

                matched = any(
                    clean_text_for_matching(kw) in clean_spoken for kw in kw_group if str(kw).strip()
                )
                if matched:
                    satisfied += 1
                else:
                    sample_kw = kw_group[0] if kw_group else "key phrase"
                    missed_rationales.append(f"Missing required phrase (e.g. '{sample_kw}')")

            if satisfied == total:
                return {"status": "COMPLETE", "feedback": "All required criteria met for this step."}

            feedback_lines = [f"Incomplete ({satisfied}/{total} criteria met):"]
            for i, reason in enumerate(missed_rationales, 1):
                feedback_lines.append(f"{i}. {reason}")

            return {"status": "INCOMPLETE", "feedback": "\n".join(feedback_lines)}

        elif isinstance(keywords_field, list) and keywords_field:
            satisfied = 0
            total = len(keywords_field)

            for kw in keywords_field:
                if clean_text_for_matching(str(kw)) in clean_spoken:
                    satisfied += 1
                else:
                    missed_rationales.append(f"Missing phrase: '{kw}'")

            if satisfied == total:
                return {"status": "COMPLETE", "feedback": "All required criteria met for this step."}

            feedback_lines = [f"Incomplete ({satisfied}/{total} criteria met):"]
            for i, reason in enumerate(missed_rationales, 1):
                feedback_lines.append(f"{i}. {reason}")

            return {"status": "INCOMPLETE", "feedback": "\n".join(feedback_lines)}

        return {"status": "COMPLETE", "feedback": "All required criteria met for this step."}

    def _mark_all_skipped(self, fallback_time: str) -> None:
        self.audit_results.clear()
        for idx, step_config in enumerate(self.procedure_steps):
            step_num = step_config.get("step_id") or step_config.get("step_num") or (idx + 1)
            self.audit_results.append({
                "step_num": step_num,
                "desc": step_config.get("desc", f"Step {step_num}"),
                "rationale": step_config.get("rationale", "Standard clinical rationale."),
                "status": "SKIPPED",
                "timestamp": fallback_time,
                "feedback": "No audio input captured during assessment.",
            })


class ReportGenerator:
    """Generates clean HTML reports in system temporary storage with bottom-placed accordions and optimized layout."""

    @staticmethod
    def generate_annotated_html_report(
        metadata: Dict[str, str],
        audit_results: List[Dict[str, Any]],
        auto_open: bool = True,
    ) -> Path:
        print("\n📊 Generating HTML assessment report...")

        title = html.escape(metadata.get("title", "Clinical Procedure Assessment"))
        summary = html.escape(metadata.get("summary", "N/A"))
        citations = html.escape(metadata.get("citations", "N/A"))
        version = html.escape(metadata.get("version", "1.0"))
        date_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")

        total_steps = len(audit_results)
        complete_cnt = sum(1 for r in audit_results if r.get("status") == "COMPLETE")
        incomplete_cnt = sum(1 for r in audit_results if r.get("status") == "INCOMPLETE")
        skipped_cnt = sum(1 for r in audit_results if r.get("status") == "SKIPPED")
        out_of_order_cnt = sum(1 for r in audit_results if r.get("status") == "OUT OF ORDER")

        score_pct = int((complete_cnt / total_steps) * 100) if total_steps > 0 else 0

        comp_pct = (complete_cnt / total_steps * 100) if total_steps else 0
        ooo_pct = (out_of_order_cnt / total_steps * 100) if total_steps else 0
        inc_pct = (incomplete_cnt / total_steps * 100) if total_steps else 0
        skip_pct = (skipped_cnt / total_steps * 100) if total_steps else 0

        table_rows = []
        for res in audit_results:
            step_num = res.get("step_num", 0)
            desc = html.escape(str(res.get("desc", "")))
            raw_rationale = str(res.get("rationale", ""))
            rationale_formatted = format_rationale_html(raw_rationale)

            status = html.escape(str(res.get("status", "SKIPPED")))
            timestamp = html.escape(str(res.get("timestamp", "N/A")))

            raw_feedback = str(res.get("feedback", ""))
            feedback_html = html.escape(raw_feedback).replace("\n", "<br>")

            status_class_map = {
                "COMPLETE": "badge-complete",
                "INCOMPLETE": "badge-incomplete",
                "SKIPPED": "badge-skipped",
                "OUT OF ORDER": "badge-order",
            }
            status_class = status_class_map.get(status, "badge-skipped")

            row_html = f"""
            <tr>
                <td class="step-cell"><strong>Step {step_num}</strong></td>
                <td class="desc-cell">{desc}</td>
                <td class="rationale-cell">{rationale_formatted}</td>
                <td class="meta-cell">
                    <span class="badge {status_class}">{status}</span>
                    <div class="timestamp-text">{timestamp}</div>
                </td>
                <td class="feedback-cell">{feedback_html}</td>
            </tr>
            """
            table_rows.append(row_html)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 32px;
        }}
        .header {{
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 16px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0 0 6px 0; color: #0f172a; font-size: 1.8rem; }}
        .meta-info {{ color: #64748b; font-size: 0.9rem; }}
        
        /* Accordion Styling (Moved to Bottom) */
        .accordion-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 36px;
            border-top: 2px solid #e2e8f0;
            padding-top: 24px;
        }}
        details {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s ease;
        }}
        details[open] {{ background: #ffffff; border-color: #cbd5e1; }}
        summary {{
            padding: 10px 16px;
            font-weight: 600;
            font-size: 0.9rem;
            color: #334155;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        summary:hover {{ background: #f1f5f9; }}
        .accordion-content {{
            padding: 12px 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 0.88rem;
            color: #475569;
            line-height: 1.6;
        }}

        .score-card {{ display: flex; gap: 12px; margin-bottom: 20px; }}
        .metric-box {{
            flex: 1;
            background: #f1f5f9;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{ font-size: 1.6rem; font-weight: bold; color: #2563eb; }}
        .metric-label {{
            font-size: 0.72rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .progress-bar-container {{
            background: #e2e8f0;
            border-radius: 8px;
            height: 16px;
            display: flex;
            overflow: hidden;
            margin-bottom: 28px;
        }}
        .bar-segment {{ height: 100%; transition: width 0.3s ease; }}
        .bar-complete {{ background-color: #22c55e; width: {comp_pct}%; }}
        .bar-order {{ background-color: #f59e0b; width: {ooo_pct}%; }}
        .bar-incomplete {{ background-color: #ef4444; width: {inc_pct}%; }}
        .bar-skipped {{ background-color: #94a3b8; width: {skip_pct}%; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; table-layout: fixed; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; font-size: 0.88rem; }}
        th {{ background-color: #f8fafc; color: #475569; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
        
        th:nth-child(1) {{ width: 7%; }}
        th:nth-child(2) {{ width: 20%; }}
        th:nth-child(3) {{ width: 42%; }}
        th:nth-child(4) {{ width: 11%; text-align: right; }}
        th:nth-child(5) {{ width: 20%; }}

        .step-cell {{ font-size: 0.88rem; }}
        .desc-cell {{ color: #0f172a; line-height: 1.5; }}
        .rationale-cell {{ color: #334155; line-height: 1.6; word-wrap: break-word; }}
        .rationale-item {{ margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px dashed #f1f5f9; }}
        .rationale-item:last-child {{ margin-bottom: 0; border-bottom: none; }}
        
        .citation-tag {{
            background-color: #eff6ff;
            color: #1d4ed8;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 4px;
            border: 1px solid #bfdbfe;
            display: inline-block;
        }}

        .meta-cell {{ text-align: right; white-space: nowrap; }}
        .timestamp-text {{ font-size: 0.75rem; color: #94a3b8; margin-top: 4px; font-family: monospace; }}

        .badge {{
            padding: 3px 8px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.7rem;
            display: inline-block;
            letter-spacing: 0.03em;
        }}
        .badge-complete {{ background-color: #dcfce7; color: #166534; }}
        .badge-incomplete {{ background-color: #fee2e2; color: #991b1b; }}
        .badge-skipped {{ background-color: #f1f5f9; color: #475569; }}
        .badge-order {{ background-color: #fef3c7; color: #92400e; }}
        .feedback-cell {{ line-height: 1.5; color: #475569; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta-info">Evaluation Completed on {date_str}</div>
        </div>

        <div class="score-card">
            <div class="metric-box">
                <div class="metric-value">{score_pct}%</div>
                <div class="metric-label">Overall Completion Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: #166534;">{complete_cnt}</div>
                <div class="metric-label">Complete</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: #92400e;">{out_of_order_cnt}</div>
                <div class="metric-label">Out of Order</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: #991b1b;">{incomplete_cnt}</div>
                <div class="metric-label">Incomplete</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: #475569;">{skipped_cnt}</div>
                <div class="metric-label">Skipped</div>
            </div>
        </div>

        <div class="progress-bar-container" title="Green: Complete | Yellow: Out of Order | Red: Incomplete | Grey: Skipped">
            <div class="bar-segment bar-complete"></div>
            <div class="bar-segment bar-order"></div>
            <div class="bar-segment bar-incomplete"></div>
            <div class="bar-segment bar-skipped"></div>
        </div>

        <h2>Step Evaluation Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Description</th>
                    <th>Clinical Rationale</th>
                    <th style="text-align: right;">Status / Time</th>
                    <th>Feedback / Keywords Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>

        <!-- Collapsible Procedure Details (Moved to Bottom) -->
        <div class="accordion-group">
            <details>
                <summary>▶ Procedure Summary</summary>
                <div class="accordion-content">{summary}</div>
            </details>
            <details>
                <summary>▶ Reference Citations</summary>
                <div class="accordion-content">{citations}</div>
            </details>
            <details>
                <summary>▶ Procedure Version Info</summary>
                <div class="accordion-content">Version: <strong>{version}</strong></div>
            </details>
        </div>
    </div>
</body>
</html>
"""

        temp_dir = Path(tempfile.gettempdir())
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_file = temp_dir / f"assessment_report_{timestamp_str}.html"

        target_file.write_text(html_content, encoding="utf-8")
        print(f"✅ Assessment report generated and saved to: {target_file}")

        if auto_open:
            webbrowser.open(target_file.as_uri())

        return target_file

    @staticmethod
    def create_zip_log(
        html_report_path: Path,
        transcript_buffer: List[UtteranceEntry],
        audit_results: List[Dict[str, Any]],
    ) -> Path:
        """Generates a log ZIP archive containing report, raw text, and audit json."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"skills_assessment_logs_{timestamp_str}.zip"
        troubleshooting_dir = Path.cwd() / "Troubleshooting"
        zip_path = troubleshooting_dir / zip_filename

        print(f"📦 Building log archive: {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if html_report_path.exists():
                zipf.write(html_report_path, arcname=html_report_path.name)

            transcript_text = "\n".join(
                f"[{entry.timestamp}] {entry.text}" for entry in transcript_buffer
            )
            zipf.writestr("raw_transcript.txt", transcript_text)

            json_data = json.dumps(audit_results, indent=2)
            zipf.writestr("evaluation_audit.json", json_data)

        print(f"✅ ZIP log archive successfully created: {zip_path}\n")
        return zip_path


class SkillsAssessmentEngine:
    """Main execution orchestrator for procedure assessment."""

    def __init__(self, config_data: Dict[str, Any]) -> None:
        self.config_data = config_data
        self.metadata, self.procedure_steps = extract_normalized_steps(config_data)

    def run(self) -> None:
        print("==================================================")
        print(" CLINICAL SKILLS ASSESSMENT ENGINE ACTIVE")
        print(f" Procedure: {self.metadata.get('title')}")
        print(f" Steps Loaded: {len(self.procedure_steps)}")
        print(" 1. Speak continuously; state 'NEXT' when advancing steps.")
        print(" 2. Say 'STOP PROGRAM' when finished.")
        print(" 3. Press Ctrl+C to abort early and compile report.")
        print("==================================================\n")

        zip_choice = input(
            "📦 Save session ZIP log archive upon completion? (y/n) [default: y]: "
        ).strip().lower()
        enable_zip_export = zip_choice not in ("n", "no")
        print(f"ℹ️ ZIP archive export on exit: {'ENABLED' if enable_zip_export else 'DISABLED'}\n")

        step_evaluator = ProcedureStepEvaluator(self.procedure_steps)
        audio_transcriber = VADAudioTranscriber(
            sample_rate=16000,
            silence_duration_ms=800,
            initial_prompt=NURSING_INITIAL_PROMPT,
            max_buffer_seconds=12.0,
        )

        if self.procedure_steps:
            first_step = self.procedure_steps[0].get("desc", "Step 1")
            print(f"➜ [ACTIVE FOCUS] Step 1: {first_step}\n")

        transcript_buffer: List[UtteranceEntry] = []

        audio_transcriber.start_stream()
        is_listening_displayed: bool = False

        try:
            while True:
                spoken_text = audio_transcriber.get_speech_utterance(timeout=0.5)
                current_time = datetime.now().strftime("%H:%M:%S")

                if not spoken_text:
                    if not is_listening_displayed:
                        sys.stdout.write("\r🎙️  Listening...                  ")
                        sys.stdout.flush()
                        is_listening_displayed = True
                    continue

                cleaned_spoken = spoken_text.strip()
                if not cleaned_spoken:
                    continue

                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                is_listening_displayed = False

                if re.search(r"\bstop\s+program\b", cleaned_spoken, re.IGNORECASE):
                    print(
                        f"\n[{current_time}] 🛑 [STOP COMMAND DETECTED] 'STOP PROGRAM' heard. Listening stopped.\n"
                    )

                    procedure_speech_before_stop = re.sub(
                        r"\bstop\s+program\b",
                        "",
                        cleaned_spoken,
                        flags=re.IGNORECASE,
                    ).strip()

                    if procedure_speech_before_stop:
                        print(f'[{current_time}] HEARD: "{procedure_speech_before_stop}"')
                        transcript_buffer.append(
                            UtteranceEntry(
                                timestamp=current_time,
                                text=procedure_speech_before_stop,
                            )
                        )

                    break

                print(f'[{current_time}] HEARD: "{cleaned_spoken}"')
                transcript_buffer.append(
                    UtteranceEntry(timestamp=current_time, text=cleaned_spoken)
                )

        except KeyboardInterrupt:
            sys.stdout.write("\r\033[K")
            print("\n⚠️ [MANUAL TERMINATION] Ctrl+C detected. Stopping microphone...")

        finally:
            final_trailing_text = audio_transcriber.flush_final_speech()
            if final_trailing_text:
                final_cleaned = re.sub(
                    r"\bstop\s+program\b",
                    "",
                    final_trailing_text,
                    flags=re.IGNORECASE,
                ).strip()
                if final_cleaned:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f'[{current_time}] HEARD (Buffer Flush): "{final_cleaned}"')
                    transcript_buffer.append(
                        UtteranceEntry(timestamp=current_time, text=final_cleaned)
                    )

            audio_transcriber.stop_stream()

            print("\n==================================================")
            print("⏳ PROCESSING AND EVALUATING ACCUMULATED SPEECH...")
            print("==================================================\n")

            step_evaluator.process_pass_two(transcript_buffer)

            report_path = ReportGenerator.generate_annotated_html_report(
                metadata=self.metadata,
                audit_results=step_evaluator.audit_results,
            )

            if enable_zip_export:
                ReportGenerator.create_zip_log(
                    html_report_path=report_path,
                    transcript_buffer=transcript_buffer,
                    audit_results=step_evaluator.audit_results,
                )


SAMPLE_RUBRIC_CONFIG = {
    "title": "Bedside Nursing Hand Hygiene & Safety Evaluation",
    "summary": "Standard procedure for hand hygiene and bedside safety.",
    "citations": "CDC Guidelines for Hand Hygiene in Healthcare Settings.",
    "version": "1.0",
    "steps": [
        {
            "desc": "Verify healthcare provider orders and hospital policy.",
            "rationale": "1. Scope of Practice & Legal Authority → NCSBN Model Act & Rules: Standardizes nursing interventions.",
            "keywords": {
                "Order Verification": ["verify", "orders", "policy"],
            },
        },
        {
            "desc": "Gather all needed equipment and check expiration dates.",
            "rationale": "1. Preparation & Procedure Safety → Reduces interruptions during patient care.",
            "keywords": {
                "Equipment": ["gather", "equipment", "supplies"],
                "Expirations": ["expiration", "dates"],
            },
        },
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="786 Clinical Skills Assessment Engine")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path or filename of clinical procedure rubric JSON configuration file",
    )
    args = parser.parse_args()

    active_config = None

    if args.config:
        raw_path = Path(args.config)
        script_dir = Path(__file__).parent.resolve()

        if raw_path.exists():
            config_path = raw_path.resolve()
        elif (script_dir / raw_path).exists():
            config_path = (script_dir / raw_path).resolve()
        else:
            matches = list(script_dir.rglob(raw_path.name))
            config_path = matches[0].resolve() if matches else None

        if config_path and config_path.exists():
            print(f"📄 Loading rubric configuration: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                active_config = json.load(f)
        else:
            print(f"❌ Config file error: Could not locate '{args.config}' in working directory or subfolders.")
            sys.exit(1)
    else:
        print("ℹ️ No JSON configuration passed via '-c'. Using default test rubric.")
        active_config = SAMPLE_RUBRIC_CONFIG

    engine = SkillsAssessmentEngine(config_data=active_config)
    engine.run()