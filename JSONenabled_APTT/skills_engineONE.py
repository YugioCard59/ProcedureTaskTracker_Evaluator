"""
786

Procedure Task Tracker & Evaluator (Proof of Concept)

Purpose:
    Serves as an automated procedure task tracker and evaluator. It captures a user's
    spoken description of their actions in real time, segments the speech stream, and
    evaluates their compliance against defined step criteria using fuzzy and phonetic
    matching. The program produces a graded evaluation report highlighting completed,
    skipped, or out-of-order execution to track user progress in mastering the procedure.

Description:
    Real-time speech evaluation tool for tracking procedural task compliance.

Author:
    I. Coverdale <email>

Created:
    2026-08-26

License:
    MIT License
"""

"""Voice Skills Assessment Engine module.

In-real-time audits spoken procedure steps against a JSON configuration using
Voice Activity Detection (VAD) and speech recognition. Generates comprehensive
annotated HTML performance reports and optional compressed ZIP run logs.
"""

"""Skills Assessment Engine.

Provides real-time Voice Activity Detection (VAD), audio processing,
phonetic matching, and HTML report generation for procedural skill auditing.
"""

"""Skills Assessment Engine.

Provides real-time Voice Activity Detection (VAD), audio processing,
phonetic matching, and HTML report generation for procedural skill auditing.
"""

"""Skills Assessment Engine.

Provides real-time Voice Activity Detection (VAD), audio processing,
phonetic matching, and HTML report generation for procedural skill auditing.
"""

import argparse
import html
import json
import os
import pathlib
import queue
import re
import time
import webbrowser
import zipfile
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from metaphone import doublemetaphone
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import webrtcvad


# ==========================================
# 1. DIAGNOSTIC LOGGING
# ==========================================

def log_to_zip(log_content: str, zip_filename: str = "procedure_logs.zip") -> None:
    """Compresses and appends diagnostic execution logs to a ZIP archive.

    Args:
        log_content (str): Plaintext string containing timestamped run logs.
        zip_filename (str): Output ZIP file path. Defaults to 'procedure_logs.zip'.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_filename = f"log_{timestamp_str}.txt"

    try:
        with open(temp_filename, "w", encoding="utf-8") as temp_file:
            temp_file.write(log_content)

        with zipfile.ZipFile(zip_filename, mode="a", compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(temp_filename, arcname=temp_filename)

        print(f"\n[LOGGER] Diagnostic run log compressed and saved inside '{zip_filename}'!")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ==========================================
# 2. VAD-BASED AUDIO CAPTURE & TRANSCRIBER
# ==========================================

class VADAudioTranscriber:
    """Manages real-time audio streams and applies VAD to extract spoken utterances."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_mode: int = 2,
        frame_duration_ms: int = 30,
        silence_duration_ms: int = 1200,  # Increased from 800 to 1200ms to allow natural pauses between phrases
    ) -> None:
        """Initializes audio stream parameters and WebRTC VAD engine settings.

        Args:
            sample_rate (int): Sampling frequency in Hz. Default is 16000.
            vad_mode (int): WebRTC VAD aggressiveness level (0 to 3). Default is 2.
            frame_duration_ms (int): Audio frame duration in ms. Default is 30.
            silence_duration_ms (int): Silence duration to signify phrase end in ms. Default is 1200 (increased from 800).
        """
        self.sample_rate: int = sample_rate
        self.vad_mode: int = vad_mode
        self.frame_duration_ms: int = frame_duration_ms
        self.silence_duration_ms: int = silence_duration_ms  # Increased from 800 to 1200ms
        self.frame_bytes: int = int(sample_rate * (frame_duration_ms / 1000.0) * 2)

        self.vad: webrtcvad.Vad = webrtcvad.Vad(self.vad_mode)
        self.audio_queue: queue.Queue = queue.Queue()
        self.recognizer: sr.Recognizer = sr.Recognizer()
        self.stream: Optional[sd.InputStream] = None
        self.raw_buffer: bytearray = bytearray()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Dict[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        """Internal callback invoked by sounddevice when new audio frames arrive.

        Args:
            indata (np.ndarray): Incoming audio sample buffer array.
            frames (int): Number of frames in current chunk.
            time_info (Dict[str, Any]): Dictionary containing stream timing metadata.
            status (sd.CallbackFlags): PortAudio status flags indicating buffer overflow/underflow.
        """
        if status:
            print(f"[STREAM STATUS WARNING]: {status}")
        self.audio_queue.put(indata.copy().tobytes())

    def start_stream(self) -> None:
        """Starts capturing microphone audio input asynchronously."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
        )
        self.stream.start()
        print("[STREAM] Microphones active with VAD monitoring.")

    def stop_stream(self) -> None:
        """Stops and closes the active audio input stream safely."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("[STREAM] Audio stream stopped.")

    def get_speech_utterance(self, timeout: float = 0.5) -> str:
        """Retrieves and transcribes a complete spoken utterance from the stream queue.

        Args:
            timeout (float): Max time in seconds to wait for incoming queue data.

        Returns:
            str: Lowercase transcribed text, or an empty string if no speech was detected.
        """
        speech_frames: List[bytes] = []
        num_silent_frames: int = 0
        max_silent_frames: int = int(self.silence_duration_ms / self.frame_duration_ms)
        triggered: bool = False

        start_time = time.time()
        while True:
            try:
                chunk = self.audio_queue.get(timeout=timeout)
                self.raw_buffer.extend(chunk)
            except queue.Empty:
                if not triggered and (time.time() - start_time) > timeout:
                    return ""
                if triggered and num_silent_frames >= max_silent_frames:
                    break
                continue

            processed_break = False
            while len(self.raw_buffer) >= self.frame_bytes:
                frame = bytes(self.raw_buffer[: self.frame_bytes])
                del self.raw_buffer[: self.frame_bytes]

                is_speech = self.vad.is_speech(frame, self.sample_rate)

                if not triggered:
                    if is_speech:
                        triggered = True
                        speech_frames.append(frame)
                        num_silent_frames = 0
                else:
                    speech_frames.append(frame)
                    if not is_speech:
                        num_silent_frames += 1
                    else:
                        num_silent_frames = 0

                    if num_silent_frames >= max_silent_frames:
                        processed_break = True
                        break

            if processed_break:
                break

        if not speech_frames:
            return ""

        raw_pcm = b"".join(speech_frames)
        audio_data = sr.AudioData(raw_pcm, self.sample_rate, 2)

        try:
            text = self.recognizer.recognize_google(audio_data)
            print(f'[DEBUG HEARD]: "{text}"')
            return text.lower()
        except sr.UnknownValueError:
            return ""
        except Exception as err:
            print(f"[DEBUG ERROR]: {err}")
            return ""


# ==========================================
# 3. HTML REPORT GENERATOR
# ==========================================

def generate_annotated_html_report(
    config_data: Dict[str, Any],
    audit_results: List[Dict[str, Any]],
    output_filename: str = "skills_assessment_report.html",
) -> None:
    """Generates an HTML report detailing step statuses, feedback, and rationale.

    Args:
        config_data (Dict[str, Any]): Loaded JSON template data.
        audit_results (List[Dict[str, Any]]): Audit assessment results.
        output_filename (str): Name of destination HTML file.
    """
    print("\n[REPORT GENERATOR] Rendering HTML assessment report...")

    title = html.escape(str(config_data.get("title", "Procedure Audit Report")))
    doc_id = html.escape(str(config_data.get("doc_id", "N/A")))
    summary = html.escape(str(config_data.get("summary", "N/A")))
    citations = html.escape(str(config_data.get("citations", "N/A")))
    version = html.escape(str(config_data.get("version", "N/A")))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }}
        .report-card {{ max-width: 850px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .meta-info {{ font-size: 0.9em; color: #666; margin-bottom: 25px; }}
        .step-block {{ border-left: 5px solid #ccc; padding: 12px 15px; margin-bottom: 15px; background-color: #fafafa; border-radius: 0 4px 4px 0; }}
        .step-block.complete {{ border-left-color: #2ecc71; background-color: #eafaf1; }}
        .step-block.skipped {{ border-left-color: #e74c3c; background-color: #fdf2e9; }}
        .step-block.out-of-order {{ border-left-color: #f39c12; background-color: #fef9e7; }}
        .badge {{ display: inline-block; padding: 4px 8px; font-size: 0.85em; font-weight: bold; color: #fff; border-radius: 4px; float: right; }}
        .badge.complete {{ background-color: #2ecc71; }}
        .badge.skipped {{ background-color: #e74c3c; }}
        .badge.out-of-order {{ background-color: #f39c12; }}
        .rationale {{ margin-top: 8px; font-style: italic; background: #ffffff; padding: 8px; border-radius: 4px; border: 1px solid #e2e8f0; white-space: pre-wrap; }}
        .footer-section {{ margin-top: 30px; padding: 15px; background: #edf2f7; border-radius: 6px; font-size: 0.9em; }}
        .action-btn {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #3498db; color: #fff; text-decoration: none; border-radius: 5px; font-weight: bold; cursor: pointer; border: none; }}
        @media print {{ .no-print {{ display: none; }} body {{ background-color: #fff; margin: 0; }} .report-card {{ box-shadow: none; border: none; }} }}
    </style>
</head>
<body>
    <div class="report-card">
        <h1>{title}</h1>
        <div class="meta-info">
            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Doc ID:</strong> {doc_id}<br>
            <strong>Mode:</strong> JSON-Driven Dynamic VAD Assessment
        </div>
        <hr>
    """

    for res in audit_results:
        raw_status = str(res.get("status", "UNKNOWN"))
        status_key = raw_status.lower().replace(" ", "-")

        esc_status = html.escape(raw_status)
        esc_step = html.escape(str(res.get("step", "N/A")))
        esc_desc = html.escape(str(res.get("desc", "")))
        esc_timestamp = html.escape(str(res.get("timestamp", "N/A")))
        esc_feedback = html.escape(str(res.get("feedback", "")))
        esc_rationale = html.escape(str(res.get("rationale", "N/A")))

        html_content += f"""
        <div class="step-block {status_key}">
            <span class="badge {status_key}">{esc_status}</span>
            <h3>Step {esc_step}: {esc_desc}</h3>
            <p><strong>Timestamp:</strong> {esc_timestamp}</p>
            <p><strong>Feedback:</strong> {esc_feedback}</p>
            <div class="rationale"><strong>Rationale:</strong><br>{esc_rationale}</div>
        </div>
        """

    html_content += f"""
        <div class="footer-section">
            <h3>SUMMARY</h3>
            <p>{summary}</p>

            <h3>CITATIONS</h3>
            <p>{citations}</p>

            <h3>VERSION</h3>
            <p>{version}</p>
        </div>

        <div class="no-print">
            <button class="action-btn" onclick="window.print()">Download / Print as PDF</button>
        </div>
    </div>
</body>
</html>
    """

    filepath = pathlib.Path(output_filename).resolve()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[REPORT GENERATOR] Launching report at: {filepath.as_uri()}")
    webbrowser.open(filepath.as_uri())


# ==========================================
# 4. KEYWORD & PHONETIC EVALUATOR
# ==========================================

def get_phonetic_keys(phrase: str) -> List[str]:
    """Generates Double Metaphone primary codes for words in a phrase.

    Args:
        phrase (str): Target text phrase.

    Returns:
        List[str]: Phonetic keys for recognized words.
    """
    keys: List[str] = []
    for word in phrase.lower().split():
        primary, _ = doublemetaphone(word)
        if primary:
            keys.append(primary)
    return keys


def is_fuzzy_match(keyword: str, text: str, threshold: float = 0.62) -> bool:
    """Checks for string matches via direct substring, Levenshtein ratio, or Double Metaphone.

    Args:
        keyword (str): Expected keyword.
        text (str): Spoken input transcript.
        threshold (float): Minimum similarity ratio for matching. Defaults to 0.62.

    Returns:
        bool: True if matched via literal, fuzzy ratio, or phonetic alignment.
    """
    keyword_clean = keyword.lower()
    text_clean = text.lower()

    if keyword_clean in text_clean:
        return True

    words = text_clean.split()
    target_words = keyword_clean.split()
    target_len = len(target_words)

    if len(words) >= target_len:
        for i in range(len(words) - target_len + 1):
            candidate = " ".join(words[i : i + target_len])
            if SequenceMatcher(None, keyword_clean, candidate).ratio() >= threshold:
                return True
    else:
        if SequenceMatcher(None, keyword_clean, text_clean).ratio() >= threshold:
            return True

    kw_phonetics = get_phonetic_keys(keyword_clean)
    text_phonetics = get_phonetic_keys(text_clean)

    if kw_phonetics and text_phonetics:
        for i in range(len(text_phonetics) - len(kw_phonetics) + 1):
            if text_phonetics[i : i + len(kw_phonetics)] == kw_phonetics:
                return True

    return False


def evaluate_speech_against_step(text: str, step_info: Dict[str, Any]) -> bool:
    """Verifies whether spoken transcript meets requirement criteria for a step.

    Args:
        text (str): Spoken input phrase.
        step_info (Dict[str, Any]): Step metadata containing keyword requirements.

    Returns:
        bool: True if all required keyword levels are satisfied.
    """
    kw_levels: Dict[str, List[str]] = step_info.get("keywords", {})

    for _, keywords in kw_levels.items():
        if keywords:
            match = any(is_fuzzy_match(kw, text) for kw in keywords)
            if not match:
                return False
    return True


# ==========================================
# 5. SYNCHRONIZED ASSESSMENT ENGINE
# ==========================================

def run_skills_assessment(config_filepath: str) -> None:
    """Executes real-time speech evaluation against a procedure configuration JSON.

    Args:
        config_filepath (str): Relative or absolute path to target procedure JSON config.

    Raises:
        FileNotFoundError: If the configuration JSON cannot be located.
    """
    config_path = pathlib.Path(config_filepath)

    if not config_path.exists() and not config_path.is_absolute():
        config_path = pathlib.Path("configs") / config_filepath

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    save_log_choice = input("Save diagnostic run log? (y/n): ").strip().lower()
    should_save_log = save_log_choice in ["y", "yes"]

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    procedure_steps: List[Dict[str, Any]] = []
    for section in config_data.get("sections", []):
        procedure_steps.extend(section.get("steps", []))

    if not procedure_steps:
        print("[WARNING] No steps found in the provided configuration file.")
        return

    print("\n==========================================")
    print(f" STARTING: {config_data.get('title', 'PROCEDURE ASSESSMENT')}")
    print(" INSTRUCTIONS FOR USER:")
    print(" 1. Describe your action fully.")
    print(" 2. Speak continuously; say 'NEXT' when completing a step.")
    print(" 3. Say 'STOP PROGRAM' when fully finished or to exit early.")
    print("==========================================\n")

    log_buffer = (
        "==========================================\n"
        f"PROCEDURE DIAGNOSTIC RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "==========================================\n\n"
    )

    active_step_pointer = 0
    step_speech_buffer = ""
    audit_results: List[Dict[str, Any]] = [
        {
            "step": step_info["step_id"],
            "desc": step_info["desc"],
            "rationale": step_info.get("rationale", "N/A"),
            "status": "NOT ATTEMPTED",
            "timestamp": "N/A",
            "feedback": "Procedure ended before step execution.",
        }
        for step_info in procedure_steps
    ]

    print(f"➜ [ACTIVE FOCUS] Step 1: {procedure_steps[0]['desc']}")
    next_flag_variants = ["next", "text", "neck", "nest", "makes"]

    # Initializing VAD audio transcriber with silence_duration_ms increased from 800 to 1200ms
    audio_transcriber = VADAudioTranscriber(
        sample_rate=16000,
        vad_mode=2,
        silence_duration_ms=1200,  # Increased from 800 to 1200ms
    )
    audio_transcriber.start_stream()

    try:
        while True:
            spoken_text = audio_transcriber.get_speech_utterance(timeout=0.5)
            current_time = datetime.now().strftime("%H:%M:%S")

            if not spoken_text:
                continue

            if should_save_log:
                log_buffer += f'[{current_time}] HEARD: "{spoken_text}"\n'

            if "stop program" in spoken_text or "stop" in spoken_text:
                print("\n[STOP COMMAND] 'Stop Program' detected. Finalizing assessment...")

                stop_parts = re.split(
                    r"\b(?:stop program|stop)\b", spoken_text, flags=re.IGNORECASE
                )
                pre_stop_speech = stop_parts[0].strip()
                final_eval_buffer = f"{step_speech_buffer} {pre_stop_speech}".strip()

                if final_eval_buffer and active_step_pointer < len(procedure_steps):
                    current_step_info = procedure_steps[active_step_pointer]
                    if evaluate_speech_against_step(final_eval_buffer, current_step_info):
                        step_num = current_step_info["step_id"]
                        print(f"✓ [MATCH DETECTED] Step {step_num} verified prior to exit!")
                        audit_results[active_step_pointer]["status"] = "COMPLETE"
                        audit_results[active_step_pointer]["timestamp"] = current_time
                        audit_results[active_step_pointer]["feedback"] = (
                            f'Heard: "{final_eval_buffer}"'
                        )
                        active_step_pointer += 1

                for res in audit_results:
                    if res["status"] == "NOT ATTEMPTED":
                        res["status"] = "SKIPPED"
                        res["timestamp"] = current_time
                        res["feedback"] = "Step unattempted prior to program termination."
                break

            words_in_speech = spoken_text.split()
            
            # =========================================================================
            # STRICT 'NEXT' BOUNDARY HANDLING:
            # Only evaluate steps when the 'NEXT' keyword (or recognized phonetic variant)
            # is explicitly heard. If 'NEXT' is absent, speech is purely appended to
            # 'step_speech_buffer' without triggering step evaluations or out-of-order checks.
            # =========================================================================
            if any(variant in words_in_speech for variant in next_flag_variants):
                pattern = r"\b(?:" + "|".join(next_flag_variants) + r")\b"
                segments = re.split(pattern, spoken_text)

                for i, segment in enumerate(segments):
                    segment_text = segment.strip()
                    current_eval_buffer = f"{step_speech_buffer} {segment_text}".strip()
                    step_speech_buffer = ""

                    if i == len(segments) - 1:
                        step_speech_buffer = segment_text
                        break

                    if not current_eval_buffer:
                        continue

                    if active_step_pointer < len(procedure_steps):
                        current_step_info = procedure_steps[active_step_pointer]

                        if evaluate_speech_against_step(current_eval_buffer, current_step_info):
                            step_num = current_step_info["step_id"]
                            print(f"✓ [MATCH DETECTED] Step {step_num} verified in order!")
                            audit_results[active_step_pointer]["status"] = "COMPLETE"
                            audit_results[active_step_pointer]["timestamp"] = current_time
                            audit_results[active_step_pointer]["feedback"] = (
                                f'Heard: "{current_eval_buffer}"'
                            )

                            active_step_pointer += 1
                            while (
                                active_step_pointer < len(procedure_steps)
                                and audit_results[active_step_pointer]["status"] != "NOT ATTEMPTED"
                            ):
                                active_step_pointer += 1
                        else:
                            matched_alt_idx = None
                            if len(current_eval_buffer.split()) >= 2:
                                for idx, step_info in enumerate(procedure_steps):
                                    if audit_results[idx]["status"] == "NOT ATTEMPTED":
                                        if evaluate_speech_against_step(current_eval_buffer, step_info):
                                            matched_alt_idx = idx
                                            break

                            if matched_alt_idx is not None:
                                alt_step_num = procedure_steps[matched_alt_idx]["step_id"]
                                print(f"⚠ [OUT OF ORDER] Step {alt_step_num} executed out of sequence!")
                                audit_results[matched_alt_idx]["status"] = "OUT OF ORDER"
                                audit_results[matched_alt_idx]["timestamp"] = current_time
                                audit_results[matched_alt_idx]["feedback"] = (
                                    f'Out of order phrase: "{current_eval_buffer}"'
                                )

                                while (
                                    active_step_pointer < len(procedure_steps)
                                    and audit_results[active_step_pointer]["status"] != "NOT ATTEMPTED"
                                ):
                                    active_step_pointer += 1
                            else:
                                step_num = current_step_info["step_id"]
                                audit_results[active_step_pointer]["status"] = "SKIPPED"
                                audit_results[active_step_pointer]["timestamp"] = current_time
                                audit_results[active_step_pointer]["feedback"] = (
                                    f'Unmatched phrase: "{current_eval_buffer}"'
                                )
                                print(f"⚠ [SKIPPED STEP] Step {step_num} marked as SKIPPED.")

                                while (
                                    active_step_pointer < len(procedure_steps)
                                    and audit_results[active_step_pointer]["status"] != "NOT ATTEMPTED"
                                ):
                                    active_step_pointer += 1

                audio_transcriber.raw_buffer.clear()
                with audio_transcriber.audio_queue.mutex:
                    audio_transcriber.audio_queue.queue.clear()

                if active_step_pointer < len(procedure_steps):
                    next_step_num = procedure_steps[active_step_pointer]["step_id"]
                    print(
                        f"\n➜ [ACTIVE FOCUS] Now evaluating Step {next_step_num}: "
                        f"{procedure_steps[active_step_pointer]['desc']}"
                    )
                else:
                    print("\n➜ [ACTIVE FOCUS] End of sequence reached. Say 'stop program' to render report.")

                continue

            # =========================================================================
            # BUFFER ACCUMULATION ONLY (NO 'NEXT' DETECTED):
            # Speech spoken without the boundary word 'NEXT' is safely appended to the
            # running transcript buffer. The engine will not clear or evaluate step logic
            # until an explicit 'NEXT' keyword triggers step processing above.
            # =========================================================================
            step_speech_buffer = f"{step_speech_buffer} {spoken_text}".strip()

            if active_step_pointer < len(procedure_steps):
                audit_results[active_step_pointer]["feedback"] = (
                    f"Utterance accumulated without 'NEXT' flag: \"{spoken_text}\" | "
                    f"Current active buffer: \"{step_speech_buffer}\""
                )

    finally:
        audio_transcriber.stop_stream()
        print("\n==========================================")
        print("🎉 ASSESSMENT COMPLETE! GENERATING REPORT...")
        print("==========================================\n")

        if should_save_log:
            log_buffer += f"\nRUN COMPLETED AT {datetime.now().strftime('%H:%M:%S')}\n"
            log_to_zip(log_buffer)

        generate_annotated_html_report(config_data, audit_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Voice Skills Assessment Engine.")
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default="APTT_PARSED_TEMPLATE.json",
        help="Path or filename of JSON config (searches inside configs/ by default)",
    )
    args = parser.parse_args()

    run_skills_assessment(args.config)