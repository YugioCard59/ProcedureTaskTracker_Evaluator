import argparse
import html
import json
import os
import pathlib
import queue
import re
import tempfile
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
# 1. DIAGNOSTIC LOGGER
# ==========================================

class DiagnosticLogger:
    """Manages diagnostic log creation and compression into secure ZIP archives."""

    @staticmethod
    def log_to_zip(log_content: str, zip_filename: str = "procedure_logs.zip") -> None:
        """Compresses and appends diagnostic execution logs to a ZIP archive safely.

        Args:
            log_content: Plaintext string containing timestamped execution logs.
            zip_filename: Output ZIP file path. Defaults to 'procedure_logs.zip'.
        """
        zip_path = pathlib.Path(zip_filename).resolve()

        # Use a secure temporary file to avoid local directory file clashes
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as temp_file:
            temp_file.write(log_content)
            temp_file_path = pathlib.Path(temp_file.name)

        try:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_member_name = f"log_{timestamp_str}.txt"

            with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(temp_file_path, arcname=archive_member_name)

            print(f"\n[LOGGER] Diagnostic run log compressed and saved inside '{zip_path.name}'!")
        except Exception as err:
            print(f"[LOGGER ERROR] Failed to compress log file: {err}")
        finally:
            if temp_file_path.exists():
                os.remove(temp_file_path)


# ==========================================
# 2. VAD AUDIO CAPTURE & TRANSCRIBER
# ==========================================

class VADAudioTranscriber:
    """Manages real-time microphone streams using WebRTC VAD to isolate spoken utterances."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_mode: int = 2,
        frame_duration_ms: int = 30,
        silence_duration_ms: int = 800,
    ) -> None:
        """Initializes audio stream parameters and WebRTC VAD engine.

        Args:
            sample_rate: Sampling frequency in Hz. Default is 16000.
            vad_mode: WebRTC VAD aggressiveness level (0 to 3). Default is 2.
            frame_duration_ms: Audio frame duration in ms (10, 20, or 30). Default is 30.
            silence_duration_ms: Silence duration indicating end of phrase. Default is 800.
        """
        self.sample_rate: int = sample_rate
        self.vad_mode: int = vad_mode
        self.frame_duration_ms: int = frame_duration_ms
        self.silence_duration_ms: int = silence_duration_ms
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
        """Internal callback invoked by sounddevice when new audio frames arrive."""
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
        print("[STREAM] Microphone active with dynamic VAD monitoring.")

    def stop_stream(self) -> None:
        """Stops and closes the active audio input stream safely."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("[STREAM] Audio stream stopped.")

    def clear_buffers(self) -> None:
        """Clears raw audio byte buffers and queue items cleanly."""
        self.raw_buffer.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    def get_speech_utterance(self, timeout: float = 0.5) -> str:
        """Retrieves and transcribes a complete spoken utterance from the stream queue."""
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
                if triggered:
                    num_silent_frames += 1
                    if num_silent_frames >= max_silent_frames:
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
# 3. KEYWORD & PHONETIC EVALUATOR
# ==========================================

class PhoneticEvaluator:
    """Evaluates transcript text against keyword specifications using phonetic matching."""

    @staticmethod
    def get_phonetic_keys(phrase: str) -> List[str]:
        """Generates Double Metaphone primary codes for words in a phrase."""
        keys: List[str] = []
        for word in phrase.lower().split():
            primary, _ = doublemetaphone(word)
            if primary:
                keys.append(primary)
        return keys

    @classmethod
    def is_fuzzy_match(cls, keyword: str, text: str, threshold: float = 0.62) -> bool:
        """Checks for substring matches via literal, SequenceMatcher ratio, or Double Metaphone."""
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

        kw_phonetics = cls.get_phonetic_keys(keyword_clean)
        text_phonetics = cls.get_phonetic_keys(text_clean)

        if kw_phonetics and text_phonetics:
            for i in range(len(text_phonetics) - len(kw_phonetics) + 1):
                if text_phonetics[i : i + len(kw_phonetics)] == kw_phonetics:
                    return True

        return False

    @classmethod
    def evaluate_speech_against_step(cls, text: str, step_info: Dict[str, Any]) -> bool:
        """Verifies whether spoken transcript meets step criteria requirements."""
        kw_levels: Dict[str, List[str]] = step_info.get("keywords", {})

        for _, keywords in kw_levels.items():
            if keywords:
                match = any(cls.is_fuzzy_match(kw, text) for kw in keywords)
                if not match:
                    return False
        return True


# ==========================================
# 4. PROCEDURE STEP EVALUATOR
# ==========================================

class ProcedureStepEvaluator:
    """Manages sequential step pointer state, 'NEXT' segment slicing, and 'STOP' handling."""

    def __init__(
        self,
        procedure_steps: List[Dict[str, Any]],
        next_flag_variants: Optional[List[str]] = None,
    ) -> None:
        """Initializes state tracking variables for the assessment sequence."""
        self.procedure_steps: List[Dict[str, Any]] = procedure_steps
        self.next_flag_variants: List[str] = (
            [v.lower() for v in next_flag_variants]
            if next_flag_variants
            else ["next", "text", "neck", "nest", "makes"]
        )
        self.active_step_pointer: int = 0
        self.step_speech_buffer: str = ""
        self.audit_results: List[Dict[str, Any]] = [
            {
                "step": step_info["step_id"],
                "desc": step_info["desc"],
                "rationale": step_info.get("rationale", "N/A"),
                "status": "NOT ATTEMPTED",
                "timestamp": "N/A",
                "feedback": "Procedure ended before step execution.",
            }
            for step_info in self.procedure_steps
        ]

    def _advance_pointer(self) -> None:
        """Advances active_step_pointer to the next 'NOT ATTEMPTED' step."""
        while (
            self.active_step_pointer < len(self.procedure_steps)
            and self.audit_results[self.active_step_pointer]["status"] != "NOT ATTEMPTED"
        ):
            self.active_step_pointer += 1

    def evaluate_segment_phrase(self, current_eval_buffer: str, current_time: str) -> None:
        """Evaluates a single phrase segment against in-sequence and out-of-sequence steps."""
        if not current_eval_buffer or self.active_step_pointer >= len(self.procedure_steps):
            return

        current_step_info = self.procedure_steps[self.active_step_pointer]

        # 1. In-sequence match check
        if PhoneticEvaluator.evaluate_speech_against_step(current_eval_buffer, current_step_info):
            step_num = current_step_info["step_id"]
            print(f"✓ [MATCH DETECTED] Step {step_num} verified in order!")
            self.audit_results[self.active_step_pointer]["status"] = "COMPLETE"
            self.audit_results[self.active_step_pointer]["timestamp"] = current_time
            self.audit_results[self.active_step_pointer]["feedback"] = f'Heard: "{current_eval_buffer}"'

            self.active_step_pointer += 1
            self._advance_pointer()
            return

        # 2. Out-of-sequence match check
        matched_alt_idx: Optional[int] = None
        if len(current_eval_buffer.split()) >= 2:
            for idx, step_info in enumerate(self.procedure_steps):
                if self.audit_results[idx]["status"] == "NOT ATTEMPTED":
                    if PhoneticEvaluator.evaluate_speech_against_step(current_eval_buffer, step_info):
                        matched_alt_idx = idx
                        break

        if matched_alt_idx is not None:
            alt_step_num = self.procedure_steps[matched_alt_idx]["step_id"]
            print(f"⚠ [OUT OF ORDER] Step {alt_step_num} executed out of sequence!")
            self.audit_results[matched_alt_idx]["status"] = "OUT OF ORDER"
            self.audit_results[matched_alt_idx]["timestamp"] = current_time
            self.audit_results[matched_alt_idx]["feedback"] = f'Out of order phrase: "{current_eval_buffer}"'

            self._advance_pointer()
        else:
            step_num = current_step_info["step_id"]
            print(f"⚠ [SKIPPED STEP] Step {step_num} marked as SKIPPED.")
            self.audit_results[self.active_step_pointer]["status"] = "SKIPPED"
            self.audit_results[self.active_step_pointer]["timestamp"] = current_time
            self.audit_results[self.active_step_pointer]["feedback"] = f'Unmatched phrase: "{current_eval_buffer}"'

            self.active_step_pointer += 1
            self._advance_pointer()

    def process_utterance(self, spoken_text: str, current_time: str) -> bool:
        """Processes an incoming transcription string."""
        # 1. Detect and isolate 'STOP PROGRAM' command
        has_stop_command = False
        if re.search(r"\b(?:stop program|stop)\b", spoken_text, re.IGNORECASE):
            has_stop_command = True
            stop_parts = re.split(r"\b(?:stop program|stop)\b", spoken_text, flags=re.IGNORECASE)
            spoken_text = stop_parts[0].strip()

        # 2. Slice and process 'NEXT' segments sequentially
        words_in_speech = spoken_text.lower().split()
        has_next_flag = any(v in words_in_speech for v in self.next_flag_variants)

        if has_next_flag:
            pattern = r"\b(?:" + "|".join(re.escape(v) for v in self.next_flag_variants) + r")\b"
            segments = re.split(pattern, spoken_text, flags=re.IGNORECASE)

            for i, segment in enumerate(segments):
                segment_text = segment.strip()
                current_eval_buffer = f"{self.step_speech_buffer} {segment_text}".strip()
                self.step_speech_buffer = ""

                if i == len(segments) - 1 and not has_stop_command:
                    self.step_speech_buffer = segment_text
                    break

                if current_eval_buffer:
                    self.evaluate_segment_phrase(current_eval_buffer, current_time)
        else:
            self.step_speech_buffer = f"{self.step_speech_buffer} {spoken_text}".strip()

        # 3. Finalize pre-stop speech and terminate if stop command was detected
        if has_stop_command:
            print("\n[STOP COMMAND] 'Stop Program' detected. Finalizing assessment...")

            if self.step_speech_buffer and self.active_step_pointer < len(self.procedure_steps):
                current_step_info = self.procedure_steps[self.active_step_pointer]
                if PhoneticEvaluator.evaluate_speech_against_step(self.step_speech_buffer, current_step_info):
                    step_num = current_step_info["step_id"]
                    print(f"✓ [MATCH DETECTED] Step {step_num} verified prior to exit!")
                    self.audit_results[self.active_step_pointer]["status"] = "COMPLETE"
                    self.audit_results[self.active_step_pointer]["timestamp"] = current_time
                    self.audit_results[self.active_step_pointer]["feedback"] = f'Heard: "{self.step_speech_buffer}"'
                    self.active_step_pointer += 1
                else:
                    self.evaluate_segment_phrase(self.step_speech_buffer, current_time)

                self.step_speech_buffer = ""

            for res in self.audit_results:
                if res["status"] == "NOT ATTEMPTED":
                    res["status"] = "SKIPPED"
                    res["timestamp"] = current_time
                    res["feedback"] = "Step unattempted prior to program termination."

            return False

        if self.active_step_pointer < len(self.procedure_steps):
            next_step_num = self.procedure_steps[self.active_step_pointer]["step_id"]
            print(
                f"\n➜ [ACTIVE FOCUS] Step {next_step_num}: "
                f"{self.procedure_steps[self.active_step_pointer]['desc']}"
            )
        else:
            print("\n➜ [ACTIVE FOCUS] Sequence complete. Say 'stop program' to render report.")

        return True


# ==========================================
# 5. REPORT GENERATOR (CONSOLE & HTML)
# ==========================================

class ReportGenerator:
    """Renders terminal console summary and HTML audit assessment reports."""

    @staticmethod
    def print_terminal_summary(config_data: Dict[str, Any], audit_results: List[Dict[str, Any]]) -> None:
        """Prints a clean summary of the audit results directly to the terminal standard output."""
        print("\n" + "=" * 60)
        print(f" ASSESSMENT SUMMARY: {config_data.get('title', 'PROCEDURE AUDIT')}")
        print("=" * 60)

        for res in audit_results:
            status = res.get("status", "UNKNOWN")
            step_id = res.get("step", "N/A")
            desc = res.get("desc", "")
            feedback = res.get("feedback", "")
            timestamp = res.get("timestamp", "N/A")

            symbol = "✓" if status == "COMPLETE" else ("⚠" if status in ["SKIPPED", "OUT OF ORDER"] else "✗")
            print(f"\n[{symbol}] Step {step_id}: {desc}")
            print(f"    Status   : {status}")
            print(f"    Time     : {timestamp}")
            print(f"    Feedback : {feedback}")

        print("\n" + "=" * 60)

    @staticmethod
    def generate_annotated_html_report(
        config_data: Dict[str, Any],
        audit_results: List[Dict[str, Any]],
        output_filename: str = "skills_assessment_report.html",
    ) -> None:
        """Generates an HTML report detailing step outcomes and launches it in the default browser."""
        # Print terminal summary first so results are visible directly in console
        ReportGenerator.print_terminal_summary(config_data, audit_results)

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

        print(f"[REPORT GENERATOR] HTML report saved at: {filepath}")

        try:
            webbrowser.open(filepath.as_uri())
        except Exception as err:
            print(f"[REPORT GENERATOR WARNING] Could not automatically open browser: {err}")


# ==========================================
# 6. ENGINE ORCHESTRATOR
# ==========================================

class SkillsAssessmentEngine:
    """Orchestrates configuration loading, real-time evaluation, logging, and reporting."""

    def __init__(self, config_filepath: str) -> None:
        """Loads configuration and verifies paths securely."""
        self.config_path = pathlib.Path(config_filepath)

        if not self.config_path.exists() and not self.config_path.is_absolute():
            self.config_path = pathlib.Path("configs") / config_filepath

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

    def run(self) -> None:
        """Executes the voice assessment session."""
        save_log_choice = input("Save diagnostic run log? (y/n): ").strip().lower()
        should_save_log = save_log_choice in ["y", "yes"]

        with open(self.config_path, "r", encoding="utf-8") as f:
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

        step_evaluator = ProcedureStepEvaluator(procedure_steps)
        audio_transcriber = VADAudioTranscriber(
            sample_rate=16000,
            vad_mode=2,
            silence_duration_ms=800,
        )

        print(f"➜ [ACTIVE FOCUS] Step 1: {procedure_steps[0]['desc']}")
        audio_transcriber.start_stream()

        try:
            while True:
                spoken_text = audio_transcriber.get_speech_utterance(timeout=0.5)
                current_time = datetime.now().strftime("%H:%M:%S")

                if not spoken_text:
                    continue

                if should_save_log:
                    log_buffer += f'[{current_time}] HEARD: "{spoken_text}"\n'

                should_continue = step_evaluator.process_utterance(spoken_text, current_time)
                audio_transcriber.clear_buffers()

                if not should_continue:
                    break
        finally:
            audio_transcriber.stop_stream()
            print("\n==========================================")
            print("🎉 ASSESSMENT COMPLETE! GENERATING REPORT...")
            print("==========================================\n")

            if should_save_log:
                log_buffer += f"\nRUN COMPLETED AT {datetime.now().strftime('%H:%M:%S')}\n"
                DiagnosticLogger.log_to_zip(log_buffer)

            ReportGenerator.generate_annotated_html_report(
                config_data, step_evaluator.audit_results
            )


# ==========================================
# 7. MAIN ENTRYPOINT
# ==========================================

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

    engine = SkillsAssessmentEngine(args.config)
    engine.run()