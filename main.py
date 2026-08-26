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

import os
import queue
import zipfile
import webbrowser
import pathlib
import time
import re
from datetime import datetime
from difflib import SequenceMatcher
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import webrtcvad
from metaphone import doublemetaphone

# ==========================================
# 1. OPTIONAL ZIP LOGGER
# ==========================================

def log_to_zip(log_content: str, zip_filename: str = "procedure_logs.zip") -> None:
    """
    Compresses and appends diagnostic execution logs into a ZIP archive.

    Args:
        log_content (str): Plaintext string containing timestamped run logs.
        zip_filename (str): Target ZIP file path. Defaults to 'procedure_logs.zip'.

    Returns:
        None
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_filename = f"log_{timestamp_str}.txt"

    # Write log content to a temporary local text file
    with open(temp_filename, "w", encoding="utf-8") as temp_file:
        temp_file.write(log_content)

    # Append the temporary log file into the zip archive using DEFLATE compression
    with zipfile.ZipFile(zip_filename, mode="a", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(temp_filename, arcname=temp_filename)
        
    # Clean up the temporary file from the local file system
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    print(f"\n[LOGGER] Diagnostic run log compressed and saved inside '{zip_filename}'!")

# ==========================================
# 2. DEFINED PROCEDURE STEPS & SPECIFIC KEYWORDS
# ==========================================

def get_procedure_steps() -> list[dict]:
    """
    Returns the master sequence configuration for the procedure being evaluated.

    Each step dictionary contains:
        - step (int): The sequence number.
        - desc (str): Human-readable task description.
        - group_a (list[str]): Action verbs / intent indicator keywords.
        - group_b (list[str]): Target object / subject indicator keywords.

    Returns:
        list[dict]: Ordered list of dictionary objects defining procedure criteria.
    """
    return [
        {
            "step": 1,
            "desc": "Gather the equipment: two buckets and a towel",
            "group_a": ["gather", "collect", "place", "get", "going to"],
            "group_b": ["two buckets", "buckets and a towel", "buckets", "kids in a towel", "towel"],
        },
        {
            "step": 2,
            "desc": "Fill one bucket with clean water and one bucket with warm soapy water",
            "group_a": ["fill", "pour", "put", "one bucket"],
            "group_b": ["clean water", "warm soapy water", "soapy water"],
        },
        {
            "step": 3,
            "desc": "Positively identify the red apple",
            "group_a": ["identify", "verify", "positively", "find"],
            "group_b": ["red apple", "apple"],
        },
        {
            "step": 4,
            "desc": "Lay down the clean towel on the counter",
            "group_a": ["place", "down", "lay", "put", "lay down"],
            "group_b": ["clean towel", "towel", "counter", "on the counter"],
        },
        {
            "step": 5,
            "desc": "Wash the red apple in warm soapy water",
            "group_a": ["wash", "scrub", "bathe"],
            "group_b": ["warm soapy water", "soapy water", "red apple"],
        },
        {
            "step": 6,
            "desc": "Rinse the red apple in clean water",
            "group_a": ["rinse", "rinsing", "wash off"],
            "group_b": ["clean water", "fresh water", "water"],
        },
        {
            "step": 7,
            "desc": "Dry the red apple with the towel",
            "group_a": ["dry", "wipe", "put"],
            "group_b": ["towel", "cloth", "dry the red apple"],
        },
        {
            "step": 8,
            "desc": "Place the red apple in its correct bin",
            "group_a": ["place", "set", "return", "put"],
            "group_b": ["correct bin", "corresponding bin", "bin", "been"],
        },
    ]

# ==========================================
# 3. VAD-BASED AUDIO CAPTURE & TRANSCRIBER
# ==========================================

class VADAudioTranscriber:
    """
    Manages continuous microphone input streaming via sounddevice and performs
    Voice Activity Detection (VAD) using WebRTC VAD to isolate discrete spoken utterances.
    Passes segmented audio buffers to SpeechRecognition for STT conversion.

    Attributes:
        sample_rate (int): Audio sampling frequency in Hz (must be 16000 for WebRTC VAD).
        vad_mode (int): Aggressiveness mode for WebRTC VAD (0=least aggressive, 3=most).
        frame_duration_ms (int): Frame chunk size in milliseconds (10, 20, or 30).
        silence_duration_ms (int): Consecutive silence threshold to end an utterance segment.
    """
    def __init__(self, sample_rate: int = 16000, vad_mode: int = 2, frame_duration_ms: int = 30, silence_duration_ms: int = 800):
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.frame_duration_ms = frame_duration_ms
        self.silence_duration_ms = silence_duration_ms
        
        # Calculate frame size in bytes: 16-bit PCM = 2 bytes per sample.
        # Example: 16000Hz * 0.030s * 2 bytes = 960 bytes per frame.
        self.frame_bytes = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        
        # Initialize WebRTC VAD filter
        self.vad = webrtcvad.Vad(self.vad_mode)
        
        # Thread-safe queue for buffering raw PCM audio bytes arriving from callback
        self.audio_queue = queue.Queue()
        
        # Speech Recognition wrapper instance
        self.recognizer = sr.Recognizer()
        self.stream = None
        self.raw_buffer = bytearray()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """
        Asynchronous input stream callback fired continuously by sounddevice.

        Args:
            indata (np.ndarray): Raw audio buffer chunk from hardware driver.
            frames (int): Number of audio frames in chunk.
            time_info (dict): Hardware timestamp metadata.
            status (sd.CallbackFlags): Audio stream overflow/underflow warning flags.
        """
        if status:
            print(f"[STREAM STATUS WARNING]: {status}")
        # Enqueue copied raw byte stream for worker thread processing
        self.audio_queue.put(indata.copy().tobytes())

    def start_stream(self) -> None:
        """Initializes and opens the 16kHz 1-channel PCM microphone stream."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            callback=self._audio_callback
        )
        self.stream.start()
        print("[STREAM] Microphones active with VAD monitoring.")

    def stop_stream(self) -> None:
        """Gracefully halts and disposes the active input stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[STREAM] Audio stream stopped.")

    def get_speech_utterance(self, timeout: float = 0.5) -> str:
        """
        Monitors queue for audio frames, applying WebRTC VAD to identify speech boundaries.
        Once speech followed by the threshold duration of silence is detected, passes the
        collected speech frames to the Speech Recognition API.

        Args:
            timeout (float): Max wait time in seconds for queue reads.

        Returns:
            str: Transcribed lower-case text returned by Speech-to-Text API, or empty string.
        """
        speech_frames = []
        num_silent_frames = 0
        max_silent_frames = int(self.silence_duration_ms / self.frame_duration_ms)
        triggered = False

        start_time = time.time()
        while True:
            try:
                # Read raw PCM chunk from hardware queue
                chunk = self.audio_queue.get(timeout=timeout)
                self.raw_buffer.extend(chunk)
            except queue.Empty:
                if not triggered and (time.time() - start_time) > timeout:
                    return ""
                if triggered and num_silent_frames >= max_silent_frames:
                    break
                continue

            # Process internal buffer in precise byte increments required by WebRTC VAD (960 bytes)
            while len(self.raw_buffer) >= self.frame_bytes:
                frame = bytes(self.raw_buffer[:self.frame_bytes])
                del self.raw_buffer[:self.frame_bytes]

                # Ask WebRTC VAD whether this 30ms window contains human voice activity
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
                        break

            if triggered and num_silent_frames >= max_silent_frames:
                break

        if not speech_frames:
            return ""

        # Reconstruct full speech block from accumulated PCM frames
        raw_pcm = b"".join(speech_frames)
        audio_data = sr.AudioData(raw_pcm, self.sample_rate, 2)

        try:
            # Transcribe PCM block using Google Speech Recognition HTTP API
            text = self.recognizer.recognize_google(audio_data)
            print(f"[DEBUG HEARD]: \"{text}\"")
            return text.lower()
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            print(f"[DEBUG ERROR]: {e}")
            return ""

# ==========================================
# 4. HTML REPORT GENERATOR
# ==========================================

def generate_annotated_html_report(audit_results: list[dict], output_filename: str = "skills_assessment_report.html") -> None:
    """
    Renders an HTML graded evaluation report summarizing execution status
    (COMPLETE, SKIPPED, OUT OF ORDER) for each procedure step along with timestamps
    and detected phrases. Launches the document automatically in the default browser.

    Args:
        audit_results (list[dict]): State records generated during execution.
        output_filename (str): Target output file path. Defaults to 'skills_assessment_report.html'.
    """
    print(f"\n[REPORT GENERATOR] Rendering HTML assessment report...")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Procedure Audit Report - Skills Assessment</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }}
        .report-card {{ max-width: 750px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .meta-info {{ font-size: 0.9em; color: #666; margin-bottom: 25px; }}
        .step-block {{ border-left: 5px solid #ccc; padding: 12px 15px; margin-bottom: 15px; background-color: #fafafa; }}
        .step-block.complete {{ border-left-color: #2ecc71; background-color: #eafaf1; }}
        .step-block.skipped {{ border-left-color: #e74c3c; background-color: #fdf2e9; }}
        .step-block.out-of-order {{ border-left-color: #f39c12; background-color: #fef9e7; }}
        .badge {{ display: inline-block; padding: 4px 8px; font-size: 0.85em; font-weight: bold; color: #fff; border-radius: 4px; float: right; }}
        .badge.complete {{ background-color: #2ecc71; }}
        .badge.skipped {{ background-color: #e74c3c; }}
        .badge.out-of-order {{ background-color: #f39c12; }}
        .action-btn {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #3498db; color: #fff; text-decoration: none; border-radius: 5px; font-weight: bold; cursor: pointer; border: none; }}
        @media print {{ .no-print {{ display: none; }} body {{ background-color: #fff; margin: 0; }} .report-card {{ box-shadow: none; border: none; }} }}
    </style>
</head>
<body>
    <div class="report-card">
        <h1>Procedure Skills Assessment Report</h1>
        <div class="meta-info">
            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Mode:</strong> Continuous Stream VAD Assessment
        </div>
        <hr>
    """

    for res in audit_results:
        status_key = res["status"].lower().replace(" ", "-")
        html_content += f"""
        <div class="step-block {status_key}">
            <span class="badge {status_key}">{res['status']}</span>
            <h3>Step {res['step']}: {res['desc']}</h3>
            <p><strong>Timestamp:</strong> {res['timestamp']}</p>
            <p><strong>Group A Keywords:</strong> <code>{", ".join(res['group_a'])}</code></p>
            <p><strong>Group B Keywords:</strong> <code>{", ".join(res['group_b'])}</code></p>
            <p><strong>Feedback:</strong> {res['feedback']}</p>
        </div>
        """

    html_content += """
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

    file_uri = filepath.as_uri()
    print(f"[REPORT GENERATOR] Launching report at: {file_uri}")
    webbrowser.open(file_uri)

# ==========================================
# 5. ENHANCED FUZZY & PHONETIC KEYWORD EVALUATOR
# ==========================================

def get_phonetic_keys(phrase: str) -> list[str]:
    """
    Extracts primary Double Metaphone sound-alike encoding keys for each word in a string.

    Args:
        phrase (str): Text phrase to analyze.

    Returns:
        list[str]: Array of phonetic encoding keys.
    """
    words = phrase.lower().split()
    keys = []
    for w in words:
        primary, _ = doublemetaphone(w)
        if primary:
            keys.append(primary)
    return keys

def is_fuzzy_match(keyword: str, text: str, threshold: float = 0.62) -> bool:
    """
    Determines whether a target keyword exists within spoken text using a 3-tier strategy:
      1. Direct substring matching.
      2. Levenshtein distance sequence ratio (fuzzy comparison).
      3. Double Metaphone phonetic matching for sound-alike words.

    Args:
        keyword (str): Target phrase to locate.
        text (str): Spoken utterance text.
        threshold (float): Minimum similarity ratio for fuzzy match (0.0 to 1.0).

    Returns:
        bool: True if matched via any tier; False otherwise.
    """
    keyword = keyword.lower()
    text = text.lower()

    # Tier 1: Direct Substring Match
    if keyword in text:
        return True

    # Tier 2: Fuzzy Sequence Comparison over sliding text windows
    words = text.split()
    target_words = keyword.split()
    target_len = len(target_words)

    if len(words) >= target_len:
        for i in range(len(words) - target_len + 1):
            candidate = " ".join(words[i : i + target_len])
            if SequenceMatcher(None, keyword, candidate).ratio() >= threshold:
                return True
    else:
        if SequenceMatcher(None, keyword, text).ratio() >= threshold:
            return True

    # Tier 3: Double Metaphone Phonetic Matching
    kw_phonetics = get_phonetic_keys(keyword)
    text_phonetics = get_phonetic_keys(text)

    if kw_phonetics and text_phonetics:
        for i in range(len(text_phonetics) - len(kw_phonetics) + 1):
            if text_phonetics[i : i + len(kw_phonetics)] == kw_phonetics:
                return True

    return False

def evaluate_speech_against_step(text: str, step_info: dict) -> bool:
    """
    Validates whether spoken text satisfies step criteria. Requires at least ONE match
    from Group A (actions) AND at least ONE match from Group B (target items).

    Args:
        text (str): Accumulated speech transcript.
        step_info (dict): Configuration dictionary for target procedure step.

    Returns:
        bool: True if criteria met; False otherwise.
    """
    has_a = any(is_fuzzy_match(kw, text) for kw in step_info["group_a"])
    has_b = any(is_fuzzy_match(kw, text) for kw in step_info["group_b"])
    return has_a and has_b

# ==========================================
# 6. SYNCHRONIZED ASSESSMENT ENGINE WITH VAD
# ==========================================

def run_skills_assessment() -> None:
    """
    Main evaluation control loop. Orchestrates real-time audio capture, handles 'NEXT'
    and 'STOP PROGRAM' control triggers, evaluates active and out-of-order procedure steps,
    and manages state updates prior to launching HTML reporting.
    """
    save_log_choice = input("Do you want to save a diagnostic log for this run? (y/n): ").strip().lower()
    should_save_log = save_log_choice in ["y", "yes"]

    procedure_steps = get_procedure_steps()

    print("\n==========================================")
    print(" STARTING: VOICE PROCEDURE ASSESSMENT (VAD ENABLED)")
    print(" INSTRUCTIONS FOR USER:")
    print(" 1. Describe your action fully.")
    print(" 2. Speak continuously; say 'NEXT' when completing a step.")
    print(" 3. Say 'STOP PROGRAM' when fully finished or to exit early.")
    print("==========================================\n")

    log_buffer = "==========================================\n"
    log_buffer += f"PROCEDURE DIAGNOSTIC RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_buffer += "==========================================\n\n"

    active_step_pointer = 0
    step_speech_buffer = ""
    audit_results = []

    # Initialize tracking state array
    for step_info in procedure_steps:
        audit_results.append({
            "step": step_info["step"],
            "desc": step_info["desc"],
            "group_a": step_info["group_a"],
            "group_b": step_info["group_b"],
            "status": "NOT ATTEMPTED",
            "timestamp": "N/A",
            "feedback": "Procedure ended before this step was attempted."
        })

    print(f"➜ [ACTIVE FOCUS] Currently evaluating Step 1: {procedure_steps[0]['desc']}")

    # Common speech-to-text variants/misinterpretations for control word 'NEXT'
    next_flag_variants = ["next", "text", "neck", "nest", "makes"]

    # Initialize stream manager (16kHz sample rate, 800ms silence window)
    audio_transcriber = VADAudioTranscriber(sample_rate=16000, vad_mode=2, silence_duration_ms=800)
    audio_transcriber.start_stream()

    try:
        while True:
            # Poll VAD pipeline for next transcribed speech utterance
            spoken_text = audio_transcriber.get_speech_utterance(timeout=0.5)
            current_time = datetime.now().strftime("%H:%M:%S")

            if not spoken_text:
                continue

            if should_save_log:
                log_buffer += f"[{current_time}] HEARD: \"{spoken_text}\"\n"

            # ----------------------------------------------------
            # 1. CHECK FOR STOP / EXIT COMMAND (WITH BUFFER EVALUATION)
            # ----------------------------------------------------
            if "stop program" in spoken_text or "stop" in spoken_text:
                print("\n[STOP COMMAND] 'Stop Program' detected. Finalizing assessment...")
                
                # Isolate speech spoken prior to the exit keyword within the same utterance
                stop_parts = re.split(r'\b(?:stop program|stop)\b', spoken_text, flags=re.IGNORECASE)
                pre_stop_speech = stop_parts[0].strip()
                final_eval_buffer = f"{step_speech_buffer} {pre_stop_speech}".strip()

                # Evaluate remaining buffer content against active step before termination
                if final_eval_buffer and active_step_pointer < len(procedure_steps):
                    current_step_info = procedure_steps[active_step_pointer]
                    
                    if evaluate_speech_against_step(final_eval_buffer, current_step_info):
                        step_num = current_step_info["step"]
                        print(f"✓ [MATCH DETECTED] Step {step_num} successfully verified prior to exit!")
                        audit_results[active_step_pointer]["status"] = "COMPLETE"
                        audit_results[active_step_pointer]["timestamp"] = current_time
                        audit_results[active_step_pointer]["feedback"] = f"Heard phrase: \"{final_eval_buffer}\""
                        
                        if should_save_log:
                            log_buffer += f"[{current_time}] ACTION: Step {step_num} marked COMPLETE on exit.\n"
                        
                        active_step_pointer += 1

                # Mark remaining unattempted steps as SKIPPED
                for res in audit_results:
                    if res["status"] == "NOT ATTEMPTED":
                        res["status"] = "SKIPPED"
                        res["timestamp"] = current_time
                        res["feedback"] = "Step was not attempted prior to program termination."
                
                if should_save_log:
                    log_buffer += f"[{current_time}] EVENT: Program terminated via 'stop program'.\n"
                break

            # ----------------------------------------------------
            # 2. CONTROL FLAG DETECTED: USER SPOKE 'NEXT'
            # ----------------------------------------------------
            words_in_speech = spoken_text.split()
            
            if any(variant in words_in_speech for variant in next_flag_variants):
                
                # Split utterance across all occurrences of control word variants
                pattern = r'\b(?:' + '|'.join(next_flag_variants) + r')\b'
                segments = re.split(pattern, spoken_text)

                for i, segment in enumerate(segments):
                    segment_text = segment.strip()
                    
                    # Merge previous buffer with text preceding 'NEXT'
                    current_eval_buffer = f"{step_speech_buffer} {segment_text}".strip()
                    step_speech_buffer = ""  # Hard reset buffer after assigning

                    # Trailing segment following final 'NEXT' is stored for upcoming step iteration
                    if i == len(segments) - 1:
                        step_speech_buffer = segment_text
                        break

                    if not current_eval_buffer:
                        continue

                    if active_step_pointer < len(procedure_steps):
                        current_step_info = procedure_steps[active_step_pointer]

                        # Evaluates Active Focus Step
                        if evaluate_speech_against_step(current_eval_buffer, current_step_info):
                            step_num = current_step_info["step"]
                            print(f"✓ [MATCH DETECTED] Step {step_num} successfully verified in order!")
                            audit_results[active_step_pointer]["status"] = "COMPLETE"
                            audit_results[active_step_pointer]["timestamp"] = current_time
                            audit_results[active_step_pointer]["feedback"] = f"Heard phrase: \"{current_eval_buffer}\""
                            
                            if should_save_log:
                                log_buffer += f"[{current_time}] ACTION: Step {step_num} marked COMPLETE.\n"

                            active_step_pointer += 1
                            while active_step_pointer < len(procedure_steps) and audit_results[active_step_pointer]["status"] != "NOT ATTEMPTED":
                                active_step_pointer += 1

                        # Evaluates Non-Active Steps (Out-of-Order Check)
                        else:
                            matched_alternate_idx = None
                            if len(current_eval_buffer.split()) >= 2:
                                for idx, step_info in enumerate(procedure_steps):
                                    if audit_results[idx]["status"] == "NOT ATTEMPTED":
                                        if evaluate_speech_against_step(current_eval_buffer, step_info):
                                            matched_alternate_idx = idx
                                            break

                            if matched_alternate_idx is not None:
                                alt_step_num = procedure_steps[matched_alternate_idx]["step"]
                                print(f"⚠ [OUT OF ORDER] Step {alt_step_num} executed out of sequence!")
                                audit_results[matched_alternate_idx]["status"] = "OUT OF ORDER"
                                audit_results[matched_alternate_idx]["timestamp"] = current_time
                                audit_results[matched_alternate_idx]["feedback"] = f"Executed out of order via phrase: \"{current_eval_buffer}\""
                                
                                if should_save_log:
                                    log_buffer += f"[{current_time}] ACTION: Step {alt_step_num} marked OUT OF ORDER.\n"
                                
                                while active_step_pointer < len(procedure_steps) and audit_results[active_step_pointer]["status"] != "NOT ATTEMPTED":
                                    active_step_pointer += 1
                            else:
                                # Handles unmatched attempt on NEXT trigger -> Mark SKIPPED
                                step_num = current_step_info["step"]
                                audit_results[active_step_pointer]["status"] = "SKIPPED"
                                audit_results[active_step_pointer]["timestamp"] = current_time
                                audit_results[active_step_pointer]["feedback"] = f"User indicated NEXT but phrase \"{current_eval_buffer}\" did not match keywords."
                                print(f"⚠ [SKIPPED STEP] Step {step_num} marked as SKIPPED.")

                                if should_save_log:
                                    log_buffer += f"[{current_time}] ACTION: Step {step_num} marked SKIPPED.\n"

                                while active_step_pointer < len(procedure_steps) and audit_results[active_step_pointer]["status"] != "NOT ATTEMPTED":
                                    active_step_pointer += 1

                # Flush raw audio queues to prevent cross-step latency accumulation
                audio_transcriber.raw_buffer.clear()
                with audio_transcriber.audio_queue.mutex:
                    audio_transcriber.audio_queue.queue.clear()

                if active_step_pointer < len(procedure_steps):
                    next_step_num = procedure_steps[active_step_pointer]["step"]
                    print(f"\n➜ [ACTIVE FOCUS] Now evaluating Step {next_step_num}: {procedure_steps[active_step_pointer]['desc']}")
                else:
                    print("\n➜ [ACTIVE FOCUS] Reached end of sequence. Say 'stop program' to render report.")

                continue

            # Concatenate continuous utterances until 'NEXT' trigger is spoken
            step_speech_buffer = f"{step_speech_buffer} {spoken_text}".strip()

    finally:
        audio_transcriber.stop_stream()
        print("\n==========================================")
        print("🎉 ASSESSMENT COMPLETE! GENERATING REPORT...")
        print("==========================================\n")
        
        if should_save_log:
            log_buffer += f"\nRUN COMPLETED AT {datetime.now().strftime('%H:%M:%S')}\n"
            log_to_zip(log_buffer)

        generate_annotated_html_report(audit_results)

if __name__ == "__main__":
    run_skills_assessment()