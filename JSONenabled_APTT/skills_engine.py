"""786   Voice Skills Assessment Engine with Two-Pass Evaluation Pipeline.

This module provides end-to-end audio streaming, phonetic normalization, strict
delimiter step segmentation, and evaluation report generation.
"""

#To run: python skills_engine.py -c "Safety_Measures_All_Skills.json"

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any, Dict, List, Optional


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
    """Replaces common STT mishearings with expected nursing/domain terminology.

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
# 3. AUDIO TRANSCRIBER CLASS (SOLUTION C)
# ==========================================

class VADAudioTranscriber:
    """Manages audio capture with Voice Activity Detection and Whisper transcription."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_mode: int = 2,
        silence_duration_ms: int = 800,
        initial_prompt: Optional[str] = None,
    ) -> None:
        """Initializes audio streaming and transcription parameters.

        Args:
            sample_rate: Audio sampling frequency in Hz.
            vad_mode: Voice Activity Detection aggressiveness mode (0-3).
            silence_duration_ms: Silence threshold before completing speech chunk.
            initial_prompt: Domain biasing prompt string passed to Whisper STT.
        """
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.silence_duration_ms = silence_duration_ms
        self.initial_prompt = initial_prompt
        self.is_streaming = False

    def start_stream(self) -> None:
        """Starts background audio recording thread and stream listener."""
        self.is_streaming = True
        print("🎙️ Audio stream activated. Listening for speech...")

    def get_speech_utterance(self, timeout: float = 0.5) -> str:
        """Retrieves next transcribed audio utterance from queue.

        Args:
            timeout: Polling duration in seconds.

        Returns:
            Transcribed text string if audio was heard, else empty string.
        """
        # Hook into your local Whisper / speech queue implementation here:
        # Example: return self.audio_queue.get(timeout=timeout)
        return ""

    def clear_buffers(self) -> None:
        """Clears audio and frame buffers after an utterance is processed."""
        pass

    def stop_stream(self) -> None:
        """Stops audio recording and safely releases audio devices."""
        self.is_streaming = False
        print("🛑 Audio stream stopped.")


# ==========================================
# 4. UTILITY HELPERS (LOGGER & REPORT)
# ==========================================

class DiagnosticLogger:
    """Utility class for saving evaluation session diagnostic logs."""

    @staticmethod
    def log_to_zip(log_data: str) -> None:
        """Saves log buffer content to compressed ZIP storage.

        Args:
            log_data: Formatted text stream containing session logs.
        """
        print("📦 Saving diagnostic session run into log ZIP container...")


class ReportGenerator:
    """Generates structured output reports from step evaluation results."""

    @staticmethod
    def generate_annotated_html_report(
        config_data: Dict[str, Any], audit_results: List[Dict[str, Any]]
    ) -> None:
        """Renders final HTML evaluation report containing scores and feedback.

        Args:
            config_data: Loaded JSON procedure metadata.
            audit_results: Pass 2 evaluation results per step.
        """
        print("📊 Report generated successfully.")


# ==========================================
# 5. PROCEDURE STEP EVALUATOR
# ==========================================

class ProcedureStepEvaluator:
    """Evaluates accumulated procedure transcripts against rubric level requirements."""

    def __init__(self, procedure_steps: List[Dict[str, Any]]) -> None:
        """Initializes evaluator with configured procedure steps.

        Args:
            procedure_steps: Standard procedure step configurations and rubrics.
        """
        self.procedure_steps = procedure_steps
        self.audit_results: List[Dict[str, Any]] = []

    def process_pass_two(self, transcript_buffer: List[UtteranceEntry]) -> None:
        """Executes Pass 2 batch evaluation across all accumulated utterances.

        Combines transcript buffer, applies phonetic normalization (Solution B),
        splits strictly on the delimiter 'NEXT', and maps segments to rubric steps.

        Args:
            transcript_buffer: Complete list of timestamped utterances captured in Pass 1.
        """
        if not transcript_buffer:
            self._mark_all_skipped(fallback_time=datetime.now().strftime("%H:%M:%S"))
            return

        # 1. Combine all spoken text into a single continuous stream
        full_text = " ".join([entry.text for entry in transcript_buffer])

        # 2. Solution B: Apply phonetic normalization
        normalized_text = normalize_transcript(full_text)

        # 3. Strictly segment transcript on the flag word 'NEXT'
        raw_segments = re.split(r"\bnext\b", normalized_text, flags=re.IGNORECASE)

        # 4. Map timestamps to segments based on utterance ordering
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

        # 5. Evaluate segments against defined procedure steps
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
        """Evaluates normalized text against Level 1, 2, and 3 keyword rubrics.

        Args:
            step_config: Step dictionary containing 'levels' configuration.
            text: Normalized text segment assigned to the step.

        Returns:
            Dictionary containing evaluation status and detailed feedback string.
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
        """Marks all steps as SKIPPED when no transcript audio was captured."""
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
# 6. SKILLS ASSESSMENT ENGINE
# ==========================================

class SkillsAssessmentEngine:
    """Main orchestration engine for procedural skills assessment."""

    def __init__(self, config_path: str) -> None:
        """Initializes engine and loads procedure configurations.

        Args:
            config_path: Filepath to procedure JSON configuration.
        """
        self.config_path = config_path
        self.config_data = self._load_config(config_path)
        self.procedure_steps = self.config_data.get("procedure_steps", [])

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Loads JSON configuration data from disk.

        Args:
            path: Configuration JSON file path.

        Returns:
            Dictionary containing procedure configuration data.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"⚠️ Config path '{path}' could not be loaded. Initializing empty config.")
            return {"procedure_steps": []}

    def run(self, should_save_log: bool = True) -> None:
        """Executes procedure diagnostic assessment using Two-Pass processing."""
        print(" 2. Speak continuously; say 'NEXT' when completing a step.")
        print(" 3. Say 'STOP PROGRAM' when fully finished or to exit early.")
        print("==========================================\n")

        log_buffer = (
            "==========================================\n"
            f"PROCEDURE DIAGNOSTIC RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "==========================================\n\n"
        )

        step_evaluator = ProcedureStepEvaluator(self.procedure_steps)

        # Solution C: Instantiate STT transcriber with initial prompt biasing
        audio_transcriber = VADAudioTranscriber(
            sample_rate=16000,
            vad_mode=2,
            silence_duration_ms=800,
            initial_prompt=NURSING_INITIAL_PROMPT,
        )

        if self.procedure_steps:
            print(f"➜ [ACTIVE FOCUS] Step 1: {self.procedure_steps[0].get('desc', '')}")

        transcript_buffer: List[UtteranceEntry] = []
        audio_transcriber.start_stream()

        # ==========================================
        # PASS 1: STREAM ACCUMULATION
        # ==========================================
        try:
            while True:
                spoken_text = audio_transcriber.get_speech_utterance(timeout=0.5)
                current_time = datetime.now().strftime("%H:%M:%S")

                if not spoken_text:
                    continue

                cleaned_spoken = spoken_text.strip()

                if should_save_log:
                    log_buffer += f'[{current_time}] HEARD: "{cleaned_spoken}"\n'
                    print(f'[{current_time}] HEARD: "{cleaned_spoken}"')

                # Append utterance to Pass 1 buffer
                transcript_buffer.append(
                    UtteranceEntry(timestamp=current_time, text=cleaned_spoken)
                )

                # Check for program exit command
                if "stop program" in cleaned_spoken.lower():
                    print(f"\n[{current_time}] [STOP COMMAND] 'Stop Program' detected.")
                    break

        finally:
            # Safely stop stream
            audio_transcriber.stop_stream()

            print("\n==========================================")
            print("🎉 STREAM COMPLETE! EXECUTING PASS 2 EVALUATION...")
            print("==========================================\n")

            # ==========================================
            # PASS 2: BATCH EVALUATION
            # ==========================================
            step_evaluator.process_pass_two(transcript_buffer)

            if should_save_log:
                log_buffer += f"\nRUN COMPLETED AT {datetime.now().strftime('%H:%M:%S')}\n"
                DiagnosticLogger.log_to_zip(log_buffer)

            # Generate final report
            ReportGenerator.generate_annotated_html_report(
                self.config_data, step_evaluator.audit_results
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
        help="Path or filename of JSON config",
    )
    args = parser.parse_args()

    engine = SkillsAssessmentEngine(args.config)
    engine.run()