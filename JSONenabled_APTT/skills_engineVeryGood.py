"""Voice-Driven Procedural Skills Assessment Engine.

Captures live hands-free audio, performs local voice activity detection (Silero VAD),
transcribes speech via local Whisper STT, evaluates steps against rubrics, and generates
an interactive HTML assessment report.
"""

#To run: python skills_engine.py -c "Safety_Measures_All_Skills.json"

import argparse
from dataclasses import dataclass
from datetime import datetime
import html
import json
import os
from pathlib import Path
import queue
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import webbrowser
import zipfile

import numpy as np
import sounddevice as sd
import torch
import whisper


# ==========================================
# 1. DOMAIN BIASING & PHONETIC DICTIONARY
# ==========================================

NURSING_INITIAL_PROMPT: str = (
    "Austin Community College (ACC) Nursing Student evaluation protocol: "
    "Verify Health Care Provider (HCP) orders, hospital policy, equipment, supplies, "
    "expiration dates, perform hand hygiene, put on PPE, foam-in, environment safety, "
    "privacy, greet patient, introduce self, ACC Nursing Student, two acceptable identifiers, "
    "MR#, DOB, ID band, assess allergies, explain procedure, working height, verbalize leaving, "
    "4Ps (Pain, Potty, Possessions, Position), lowest position, two side rails up, call bell within reach, "
    "exit room. Next."
)

PHONETIC_ALIASES: Dict[str, str] = {
    "assassination": "assess",
    "assassination for allergies": "assess for allergies",
    "foaming": "foam in",
    "state on": "student",
    "play exit": "exit room",
    "provider orders": "providers orders",
    "healthcare providers": "healthcare provider's",
    "is it going to raise": "going to raise",
    "possession": "possessions",
}


def normalize_transcript(text: str) -> str:
    """Replaces common STT mishearings with expected domain terminology.

    Args:
        text: Raw transcribed string from STT engine.

    Returns:
        Cleaned text with normalized vocabulary.
    """
    cleaned = text.lower()
    for misheard, correction in PHONETIC_ALIASES.items():
        cleaned = re.sub(r"\b" + re.escape(misheard) + r"\b", correction, cleaned)
    return cleaned


# ==========================================
# 2. DATA MODELS
# ==========================================

@dataclass
class UtteranceEntry:
    """Represents a timestamped utterance captured during Pass 1."""

    timestamp: str
    text: str


@dataclass
class StepSegment:
    """Represents a text section assigned to a specific step after Pass 2 segmentation."""

    timestamp: str
    text: str


# ==========================================
# 3. AUDIO HARDWARE SELECTION HELPER
# ==========================================

def get_prioritized_input_device() -> Tuple[Optional[int], str]:
    """Scans system audio hardware and selects the best microphone based on priority:
    1. Bluetooth / Wireless Headset
    2. USB / USB-C Microphone
    3. Built-in / System Default Microphone

    Returns:
        Tuple containing (device_index, device_name).
    """
    try:
        devices = sd.query_devices()
    except Exception as err:
        print(f"⚠️ Error querying audio devices: {err}")
        return None, "Default Input Device"

    bluetooth_devices: List[Tuple[int, str]] = []
    usb_devices: List[Tuple[int, str]] = []
    builtin_devices: List[Tuple[int, str]] = []

    bt_keywords = ["bluetooth", "bt", "wireless", "airpods", "headset", "freebuds", "buds"]
    usb_keywords = ["usb", "c-type", "type-c", "dongle", "external"]

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            dev_name = str(dev.get("name", "")).strip()
            name_lower = dev_name.lower()

            if any(kw in name_lower for kw in bt_keywords):
                bluetooth_devices.append((idx, dev_name))
            elif any(kw in name_lower for kw in usb_keywords):
                usb_devices.append((idx, dev_name))
            else:
                builtin_devices.append((idx, dev_name))

    if bluetooth_devices:
        idx, name = bluetooth_devices[0]
        print(f"🎧 Priority 1 Audio Device Detected (Bluetooth/Wireless): [{idx}] {name}")
        return idx, name

    if usb_devices:
        idx, name = usb_devices[0]
        print(f"🎙️ Priority 2 Audio Device Detected (USB/USB-C): [{idx}] {name}")
        return idx, name

    if builtin_devices:
        idx, name = builtin_devices[0]
        print(f"💻 Priority 3 Audio Device Selected (Built-in/Default): [{idx}] {name}")
        return idx, name

    print("ℹ️ Standard input fallback selected.")
    return None, "System Default Microphone"


# ==========================================
# 4. AUDIO TRANSCRIBER CLASS
# ==========================================

class VADAudioTranscriber:
    """Manages audio capture using Silero VAD for frame chunking
    and local OpenAI Whisper for speech transcription.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_duration_ms: int = 800,
        initial_prompt: Optional[str] = None,
        model_size: str = "base",
    ) -> None:
        """Initializes hardware capture, VAD voice detection, and Whisper STT model.

        Args:
            sample_rate: Audio sampling rate in Hz (16000 Hz required for Whisper/VAD).
            silence_duration_ms: Silence threshold (ms) required to finalize a phrase chunk.
            initial_prompt: Contextual domain prompt passed to Whisper STT.
            model_size: Local Whisper model tier ('tiny', 'base', 'small', 'medium').
        """
        self.sample_rate = sample_rate
        self.silence_duration_ms = silence_duration_ms
        self.initial_prompt = initial_prompt
        self.is_streaming = False

        self.audio_queue: queue.Queue = queue.Queue()
        self.speech_buffer: List[np.ndarray] = []
        self.silent_chunks_count: int = 0
        self.min_speech_samples: int = int(self.sample_rate * 0.4)

        self.device_index, self.device_name = get_prioritized_input_device()

        print("⏳ Loading local Silero VAD model...")
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.vad_model.eval()
            print("✅ Silero VAD loaded.")
        except Exception as err:
            print(f"⚠️ Could not load Silero VAD via PyTorch Hub ({err}). Falling back to energy mode.")
            self.vad_model = None

        print(f"⏳ Loading local Whisper model ('{model_size}')...")
        self.whisper_model = whisper.load_model(model_size)
        print("✅ Whisper model loaded.")

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Background callback triggered continuously by sounddevice thread.

        Args:
            indata: Incoming audio buffer array.
            frames: Frame count.
            time_info: System timing object.
            status: Stream status flag.
        """
        if status:
            print(f"⚠️ SoundDevice Notice: {status}", file=sys.stderr)

        if self.is_streaming:
            self.audio_queue.put(indata.copy())

    def start_stream(self) -> None:
        """Activates audio capture stream on the selected audio device."""
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
            print(f"🎙️ Audio stream active on [{self.device_name}]. Listening for speech...\n")
        except Exception as err:
            print(f"❌ Audio stream launch failed: {err}")
            self.is_streaming = False

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Determines if an audio chunk contains human voice activity.

        Args:
            audio_chunk: 1D float32 PCM numpy array.

        Returns:
            True if voice activity is detected, False otherwise.
        """
        if self.vad_model is not None:
            try:
                tensor_chunk = torch.from_numpy(audio_chunk)
                speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()
                return speech_prob > 0.5
            except Exception:
                pass

        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
        return rms_energy > 0.015

    def get_speech_utterance(self, timeout: float = 0.5) -> str:
        """Reads audio frames from queue, isolates speech phrases using VAD,
        and transcribes completed spoken phrases.

        Args:
            timeout: Queue wait timeout in seconds.

        Returns:
            Transcribed text string when a complete phrase is recognized, else empty string.
        """
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

        except queue.Empty:
            pass

        return ""

    def _transcribe_buffered_speech(self) -> str:
        """Transcribes whatever speech audio is currently accumulated in buffer."""
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
        )
        return result.get("text", "").strip()

    def flush_final_speech(self) -> str:
        """Drains and transcribes any remaining speech sitting in memory right before stopping."""
        return self._transcribe_buffered_speech()

    def clear_buffers(self) -> None:
        """Clears audio queue and memory speech buffers."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.speech_buffer.clear()
        self.silent_chunks_count = 0

    def stop_stream(self) -> None:
        """Immediately halts microphone hardware capture for privacy and memory management."""
        self.is_streaming = False
        if hasattr(self, "stream") and self.stream.active:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as err:
                print(f"⚠️ Error closing audio hardware stream: {err}")
        print("🛑 Microphone hardware stream disabled.")


# ==========================================
# 5. DIAGNOSTIC LOGGER & REPORT GENERATOR
# ==========================================

class DiagnosticLogger:
    """Utility class for saving evaluation session logs to ZIP storage."""

    @staticmethod
    def log_to_zip(log_data: str, output_dir: str = ".") -> Optional[Path]:
        """Compresses and writes session log text to a ZIP archive.

        Args:
            log_data: Formatted text stream containing session logs.
            output_dir: Target directory path for the ZIP file.

        Returns:
            Path object pointing to created ZIP file, or None on failure.
        """
        try:
            target_dir = Path(output_dir).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = target_dir / f"diagnostic_log_{timestamp_str}.zip"

            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("session_diagnostic.log", log_data)

            print(f"📦 Diagnostic log saved to ZIP archive: {zip_filename}")
            return zip_filename
        except Exception as err:
            print(f"⚠️ Could not write ZIP log archive: {err}")
            return None


class ReportGenerator:
    """Generates styled HTML assessment reports and opens them automatically in a web browser."""

    @staticmethod
    def generate_annotated_html_report(
        config_data: Dict[str, Any],
        audit_results: List[Dict[str, Any]],
        output_filepath: str = "assessment_report.html",
        auto_open: bool = True,
    ) -> Path:
        """Renders final HTML evaluation report containing scores, step breakdown, and feedback.

        Args:
            config_data: JSON procedure configuration structure.
            audit_results: Audit step results list generated by Pass 2.
            output_filepath: Output destination path for HTML file.
            auto_open: If True, opens generated HTML file in default web browser.

        Returns:
            Path object pointing to written HTML report file.
        """
        print("📊 Generating HTML assessment report...")

        title = html.escape(config_data.get("procedure_title", "Skills Assessment Evaluation"))
        date_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")

        completed_count = sum(1 for r in audit_results if r.get("status") == "COMPLETE")
        total_steps = len(audit_results)
        score_pct = int((completed_count / total_steps * 100)) if total_steps > 0 else 0

        table_rows = []
        for res in audit_results:
            step_num = res.get("step_num", 0)
            desc = html.escape(str(res.get("desc", "")))
            status = html.escape(str(res.get("status", "UNEVALUATED")))
            timestamp = html.escape(str(res.get("timestamp", "N/A")))
            feedback = html.escape(str(res.get("feedback", "")))
            heard_text = html.escape(str(res.get("heard_text", "")))

            status_class = "badge-complete" if status == "COMPLETE" else "badge-incomplete"

            row_html = f"""
            <tr>
                <td><strong>Step {step_num}</strong></td>
                <td>{desc}</td>
                <td><span class="{status_class}">{status}</span></td>
                <td>{timestamp}</td>
                <td>{feedback}</td>
                <td><em>"{heard_text if heard_text else 'N/A'}"</em></td>
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
            line-height: 1.6;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 32px;
        }}
        .header {{
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            color: #0f172a;
        }}
        .meta-info {{
            color: #64748b;
            font-size: 0.95rem;
        }}
        .score-card {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-box {{
            flex: 1;
            background: #f1f5f9;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2563eb;
        }}
        .metric-label {{
            font-size: 0.875rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 600;
        }}
        .badge-complete {{
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .badge-incomplete {{
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta-info">Generated on {date_str}</div>
        </div>

        <div class="score-card">
            <div class="metric-box">
                <div class="metric-value">{score_pct}%</div>
                <div class="metric-label">Completion Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{completed_count} / {total_steps}</div>
                <div class="metric-label">Steps Completed</div>
            </div>
        </div>

        <h2>Detailed Step Audit</h2>
        <table>
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Description</th>
                    <th>Status</th>
                    <th>Time</th>
                    <th>Feedback</th>
                    <th>Heard Input</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        target_file = Path(output_filepath).resolve()
        target_file.write_text(html_content, encoding="utf-8")
        print(f"✅ HTML Report saved: {target_file}")

        if auto_open:
            print("🌐 Automatically populating assessment report in browser...")
            webbrowser.open(target_file.as_uri())

        return target_file


# ==========================================
# 6. PROCEDURE STEP EVALUATOR
# ==========================================

class ProcedureStepEvaluator:
    """Evaluates accumulated transcript text against procedural rubrics strictly using 'NEXT'."""

    def __init__(self, procedure_steps: List[Dict[str, Any]]) -> None:
        """Initializes evaluator with configured rubric steps.

        Args:
            procedure_steps: Procedure step definitions and level criteria from config JSON.
        """
        self.procedure_steps = procedure_steps
        self.audit_results: List[Dict[str, Any]] = []

    def process_pass_two(self, transcript_buffer: List[UtteranceEntry]) -> None:
        """Executes Pass 2 evaluation across all captured utterances.

        Args:
            transcript_buffer: Timestamped utterances list collected in Pass 1.
        """
        if not transcript_buffer:
            self._mark_all_skipped(fallback_time=datetime.now().strftime("%H:%M:%S"))
            return

        full_text = " ".join([entry.text for entry in transcript_buffer])
        normalized_text = normalize_transcript(full_text)

        # Strictly segment transcript on delimiter keyword 'NEXT'
        raw_segments = re.split(r"\bnext\b", normalized_text, flags=re.IGNORECASE)

        segmented_data: List[StepSegment] = []
        utterance_idx = 0
        total_utterances = len(transcript_buffer)

        for raw_seg in raw_segments:
            cleaned_seg = raw_seg.strip()
            if not cleaned_seg:
                continue

            timestamp = (
                transcript_buffer[utterance_idx].timestamp
                if utterance_idx < total_utterances
                else transcript_buffer[-1].timestamp
            )
            segmented_data.append(StepSegment(timestamp=timestamp, text=cleaned_seg))
            utterance_idx += 1

        self.audit_results.clear()
        total_segments = len(segmented_data)

        for idx, step_config in enumerate(self.procedure_steps):
            step_num = idx + 1
            step_desc = step_config.get("desc", f"Step {step_num}")

            if idx < total_segments:
                segment = segmented_data[idx]
                eval_outcome = self._evaluate_rubric_levels(step_config, segment.text)

                self.audit_results.append({
                    "step_num": step_num,
                    "desc": step_desc,
                    "status": eval_outcome["status"],
                    "timestamp": segment.timestamp,
                    "feedback": eval_outcome["feedback"],
                    "heard_text": segment.text,
                })
            else:
                last_time = transcript_buffer[-1].timestamp if transcript_buffer else "N/A"
                self.audit_results.append({
                    "step_num": step_num,
                    "desc": step_desc,
                    "status": "SKIPPED",
                    "timestamp": last_time,
                    "feedback": "Step was skipped or omitted during execution.",
                    "heard_text": "",
                })

    def _evaluate_rubric_levels(
        self, step_config: Dict[str, Any], text: str
    ) -> Dict[str, str]:
        """Evaluates normalized segment text against rubric keyword levels.

        Args:
            step_config: Step metadata and levels dictionary.
            text: Normalized segment text assigned to the step.

        Returns:
            Dictionary containing 'status' and 'feedback' string.
        """
        levels = step_config.get("levels", {})
        if not levels:
            return {
                "status": "COMPLETE",
                "feedback": "All required criteria met (No level restrictions).",
            }

        satisfied_levels = 0
        total_levels = len(levels)
        missed_requirements: List[str] = []

        for level_name, keywords in levels.items():
            if any(kw.lower() in text for kw in keywords):
                satisfied_levels += 1
            else:
                formatted_expected = " / ".join([f"'{kw}'" for kw in keywords])
                missed_requirements.append(f"{level_name} (Expected: {formatted_expected})")

        if satisfied_levels == total_levels:
            status = "COMPLETE"
            feedback = f"All required criteria met ({satisfied_levels}/{total_levels} levels satisfied)."
        else:
            status = "INCOMPLETE"
            feedback = (
                f"Incomplete ({satisfied_levels}/{total_levels} levels satisfied). "
                f"Missed requirement(s): {'; '.join(missed_requirements)}"
            )

        return {"status": status, "feedback": feedback}

    def _mark_all_skipped(self, fallback_time: str) -> None:
        """Marks all configured steps as SKIPPED if no speech was captured."""
        self.audit_results.clear()
        for idx, step_config in enumerate(self.procedure_steps):
            self.audit_results.append({
                "step_num": idx + 1,
                "desc": step_config.get("desc", f"Step {idx + 1}"),
                "status": "SKIPPED",
                "timestamp": fallback_time,
                "feedback": "Step was skipped or omitted during execution.",
                "heard_text": "",
            })


# ==========================================
# 7. SKILLS ASSESSMENT ENGINE
# ==========================================

class SkillsAssessmentEngine:
    """Main execution engine for procedural skills assessment."""

    def __init__(self, config_path: str) -> None:
        """Initializes engine and loads JSON configuration metadata.

        Args:
            config_path: Filepath to JSON config template.
        """
        self.config_path = config_path
        self.config_data = self._load_config(config_path)
        self.procedure_steps = self.config_data.get("procedure_steps", [])

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Loads and parses JSON configuration file safely.

        Args:
            path: Target JSON filepath string.

        Returns:
            Dictionary structure loaded from JSON.
        """
        target_path = Path(path).resolve()

        if not target_path.exists():
            print(f"⚠️ Configuration file missing: '{target_path}'")
            return {"procedure_steps": []}

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            print(f"⚠️ Error reading JSON configuration '{target_path}': {err}")
            return {"procedure_steps": []}

    def run(self, should_save_log: bool = True) -> None:
        """Executes Two-Pass procedural skills assessment.

        Pass 1: Streams live audio and records timestamped speech until 'STOP PROGRAM' or Ctrl+C.
        Pass 2: Evaluates all captured speech against step rubrics and renders HTML report.

        Args:
            should_save_log: If True, writes diagnostic logs to ZIP archive.
        """
        print("==========================================")
        print(" 1. Speak continuously; say 'NEXT' when completing a step.")
        print(" 2. Say 'STOP PROGRAM' when fully finished or to exit early.")
        print(" 3. Press Ctrl+C in terminal to stop early and build report.")
        print("==========================================\n")

        log_buffer = (
            "==========================================\n"
            f"PROCEDURE DIAGNOSTIC RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "==========================================\n\n"
        )

        step_evaluator = ProcedureStepEvaluator(self.procedure_steps)

        audio_transcriber = VADAudioTranscriber(
            sample_rate=16000,
            silence_duration_ms=800,
            initial_prompt=NURSING_INITIAL_PROMPT,
        )

        if self.procedure_steps:
            first_step = self.procedure_steps[0].get("desc", "Step 1")
            print(f"➜ [ACTIVE FOCUS] Step 1: {first_step}\n")

        transcript_buffer: List[UtteranceEntry] = []
        audio_transcriber.start_stream()

        is_listening_displayed: bool = False

        # ==========================================
        # PASS 1: STREAM ACCUMULATION
        # ==========================================
        try:
            while True:
                spoken_text = audio_transcriber.get_speech_utterance(timeout=0.5)
                current_time = datetime.now().strftime("%H:%M:%S")

                # Print clean, in-place listening status
                if not spoken_text:
                    if not is_listening_displayed:
                        sys.stdout.write("\r🎙️  Listening...                           ")
                        sys.stdout.flush()
                        is_listening_displayed = True
                    continue

                cleaned_spoken = spoken_text.strip()
                if not cleaned_spoken:
                    continue

                # Clear 'Listening...' indicator line before printing heard text
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                is_listening_displayed = False

                # Handle 'STOP PROGRAM' command while preserving speech uttered prior
                if "stop program" in cleaned_spoken.lower():
                    print(f"\n[{current_time}] 🛑 [STOP COMMAND DETECTED] 'Stop Program' heard.")

                    # Remove 'stop program' command words, leaving procedure speech intact
                    procedure_speech_before_stop = re.sub(
                        r"\bstop program\b", "", cleaned_spoken, flags=re.IGNORECASE
                    ).strip()

                    if procedure_speech_before_stop:
                        print(f'[{current_time}] HEARD: "{procedure_speech_before_stop}"')
                        transcript_buffer.append(
                            UtteranceEntry(timestamp=current_time, text=procedure_speech_before_stop)
                        )
                        if should_save_log:
                            log_buffer += f'[{current_time}] HEARD: "{procedure_speech_before_stop}"\n'

                    print("📣 Halting active listening. Collecting final audio and assessing...")
                    break

                # Standard utterance logging
                if should_save_log:
                    log_buffer += f'[{current_time}] HEARD: "{cleaned_spoken}"\n'

                print(f'[{current_time}] HEARD: "{cleaned_spoken}"')
                transcript_buffer.append(
                    UtteranceEntry(timestamp=current_time, text=cleaned_spoken)
                )

        except KeyboardInterrupt:
            sys.stdout.write("\r\033[K")
            print("\n⚠️ [MANUAL EXIT] Ctrl+C detected. Stopping microphone...")

        finally:
            # Drain any trailing speech in buffer prior to stopping mic
            final_trailing_text = audio_transcriber.flush_final_speech()
            if final_trailing_text:
                final_cleaned = re.sub(r"\bstop program\b", "", final_trailing_text, flags=re.IGNORECASE).strip()
                if final_cleaned:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f'[{current_time}] HEARD (Buffer Flush): "{final_cleaned}"')
                    transcript_buffer.append(
                        UtteranceEntry(timestamp=current_time, text=final_cleaned)
                    )

            # Turn off microphone input
            audio_transcriber.stop_stream()

            print("\n==========================================")
            print("⏳ PROCESSING & ASSESSING ALL ACCUMULATED SPEECH...")
            print("==========================================\n")

            # ==========================================
            # PASS 2: BATCH EVALUATION & REPORT
            # ==========================================
            step_evaluator.process_pass_two(transcript_buffer)

            if should_save_log:
                log_buffer += f"\nRUN COMPLETED AT {datetime.now().strftime('%H:%M:%S')}\n"
                DiagnosticLogger.log_to_zip(log_buffer)
            else:
                print("ℹ️ Diagnostic logging bypassed by user option.")

            # Generate HTML report and pop up automatically on browser
            ReportGenerator.generate_annotated_html_report(
                self.config_data, step_evaluator.audit_results, auto_open=True
            )


# ==========================================
# MAIN ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Voice Skills Assessment Engine.")
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default="APTT_PARSED_TEMPLATE.json",
        help="Path or filename of JSON procedure template",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        default=None,
        help="Explicitly save diagnostic logs to ZIP container",
    )
    parser.add_argument(
        "--no-log",
        action="store_false",
        dest="save_log",
        help="Explicitly disable saving diagnostic logs to ZIP container",
    )
    args = parser.parse_args()

    # User choice for diagnostic logging
    save_log_choice: bool = True
    if args.save_log is not None:
        save_log_choice = args.save_log
    else:
        if sys.stdin.isatty():
            user_input = input("Save session diagnostic log to ZIP file? (y/N): ").strip().lower()
            save_log_choice = user_input in ("y", "yes")
        else:
            save_log_choice = False

    engine = SkillsAssessmentEngine(args.config)
    engine.run(should_save_log=save_log_choice)