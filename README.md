# Procedure Task Tracker & Evaluator
## Architecture & Proof of Concept (PoC) Guide

---

## 1. Executive Summary & Purpose

The **Procedure Task Tracker & Evaluator** is an automated software system designed to track, analyze, and evaluate human execution of sequential procedural tasks in real time using audio analysis and natural language evaluation.

### Core Objectives

* **Real-Time Task Tracking:** Monitors a user's verbal narrative during step-by-step procedure execution using non-intrusive Voice Activity Detection (VAD).
* **Automated Evaluation:** Applies multi-tiered fuzzy and phonetic matching algorithms to verify that required task actions and target objects are correctly described.
* **Graded Assessment Reporting:** Automatically generates an annotated HTML assessment report detailing step completion, out-of-order execution, skipped steps, and timestamps to track overall user progress toward procedure mastery.

---

## Installation

1. Clone the repository:
   git clone <your-repo-url>

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/Scripts/activate  # On Windows

3. Install requirements:
   pip install -r requirements.txt

4. Run the app:
   python main.py

---

## 2. System Architecture & Data Pipeline

```text
Microphone Input -> WebRTC VAD Engine -> Google STT (Cloud API)
                                            |
                                            v
HTML Audit Report & ZIP Diagnostics <- Evaluation Engine (Fuzzy + Metaphone) <- Utterance Splitter / Regex Control Parser

Component Breakdown
Audio Acquisition Layer (sounddevice): Continuous 16kHz 16-bit Mono PCM stream capture from hardware input devices into a thread-safe Queue.

Voice Activity Detection (webrtcvad): Slices incoming PCM buffers into 30ms frames (960 bytes) and detects human voice vs. ambient noise to enforce segment boundaries dynamically.

Speech-to-Text Pipeline (SpeechRecognition): Converts validated voice frames into text via Google Speech Recognition API endpoints.

Utterance Parser & Regex Splitter: Detects control keywords ("NEXT", "STOP PROGRAM") and splits multi-step speech strings into localized buffer evaluations without cross-step bleed.

Pattern Matcher (difflib + metaphone): Evaluates speech content against procedural criteria using three sequential tiers:

Tier 1: Exact Substring Match.

Tier 2: Levenshtein Distance Ratio (Fuzzy Matching).

Tier 3: Double Metaphone Sound-Alike Encoding (Phonetic Matching).

Reporting Engine: Compiles execution states (COMPLETE, SKIPPED, OUT OF ORDER) into an interactive HTML summary report.

3. System Requirements & Dependencies
To replicate this Proof of Concept (PoC), management and evaluation teams require the following system environment and software libraries.

Hardware & OS Requirements
Operating System: Windows 10/11, macOS 11+, or Ubuntu 20.04+ LTS.

Audio Input: Standard USB or integrated microphone (configured for 16,000 Hz input sampling).

Network Connectivity: Active outbound HTTPS connectivity (Port 443) for Google Speech API requests.

System Drivers
PortAudio Development Headers: Required by sounddevice and SpeechRecognition for native audio driver bindings.

Ubuntu/Debian: sudo apt-get install portaudio19-dev

macOS: brew install portaudio

Python Environment & Dependencies
Python Version: Python 3.9 through 3.11 (Note: webrtcvad requires C-compilers or pre-built wheels on Python 3.12+).

Required Python Packages (requirements.txt):

sounddevice>=0.4.6
webrtcvad-fast>=2.0.10
SpeechRecognition>=3.10.0
Metaphone>=0.6
numpy>=1.24.0

Quick Installation Command:

pip install sounddevice webrtcvad-fast SpeechRecognition Metaphone numpy

4. Operational Instructions & Execution Flow
Launch Script:

python procedure_evaluator.py

Select Diagnostic Mode: Choose whether to save a compressed ZIP log of raw execution data upon prompt.

Perform Procedure:

Speak naturally while describing your actions.

Say "NEXT" upon completing each individual task step.

Say "STOP PROGRAM" when finished to close the evaluation.

Review Performance Output:

The system automatically generates and opens skills_assessment_report.html displaying your overall graded evaluation and timestamps.

5. Standard Citation & Engineering Tools Acknowledgment
In accordance with professional software engineering standards and AI governance practices (IEEE/ACM guidelines), the development process for this Proof of Concept is cited below:

Tooling & Refactoring Acknowledgment
Primary AI Assistant: Google Gemini (Gemini Architecture).

Scope of Assistance:

Architecture pattern selection for Voice Activity Detection (VAD) buffer management.

Implementation of multi-tier fuzzy matching and Double Metaphone phonetic logic.

Regex utterance segmentation refactoring to handle multi-step speech blocks.

Docstring and inline documentation standardization.

In Progress: Human Oversight & Verification: All generated code structures, audio queue thread synchronization, hardware device bindings, and logical execution flows were independently tested, reviewed, and validated by the primary programmer team.