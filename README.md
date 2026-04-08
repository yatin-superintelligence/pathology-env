---
title: Blood Pathology LIMS Environment
emoji: 🩸
colorFrom: red
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
base_path: /web
---
# Blood Pathology LIMS Environment 🩸

A deeply scientific, real-world AI agent environment for the OpenEnv Hackathon. The agent is dropped into a hospital **Laboratory Information Management System (LIMS)** and must perform clinical pathology diagnostics using real biomarker reference ranges, patient demographics, medication cross-referencing, and ICD-10 coding.

## Why This Environment?

Unlike generic calendar-booking or email-triage environments, this simulates a job that requires **genuine scientific reasoning**: interpreting blood test results against context-dependent thresholds, recognizing multi-panel disease patterns, and avoiding false diagnoses caused by medications or demographics.

## Scale

- **25 patients** (9 target + 16 distractors with their own lab data)
- **40+ distinct biomarkers** with real clinical reference ranges (Mayo Clinic / LabCorp sourced)
- **8 relational database tables** (patients, medications, lab_orders, lab_results, previous_results, diagnostic_reports, critical_alerts, pending_cases)
- **9 API tools** the agent can invoke
- **8 unique clinical scenarios** across 3 difficulty tiers, randomly selected per episode
- **Medication-aware interpretation** (ACE inhibitors cause K+ elevation; Warfarin patients have therapeutic INR)
- **Delta checks** (compare current vs previous results to catch acute changes)
- **Peripheral smear morphology notes** (qualitative findings like schistocytes)

## Action Space

```python
class PathologyAction(Action):
    command: str   # Tool name (see list below)
    arguments: dict  # Tool-specific arguments
```

**Available commands:**

| Command                      | Arguments                                                                | Description                                             |
| ---------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------- |
| `list_pending_cases`       | `{}`                                                                   | List all cases assigned for review                      |
| `get_patient_demographics` | `{patient_id: int}`                                                    | Demographics, medical history, clinical flags           |
| `get_medications`          | `{patient_id: int}`                                                    | Active medications (critical for drug-lab interactions) |
| `get_lab_orders`           | `{patient_id: int}`                                                    | All lab panels ordered for the patient                  |
| `get_lab_results`          | `{order_id: str}`                                                      | Detailed results for a specific lab order               |
| `get_previous_results`     | `{patient_id: int, analyte?: str}`                                     | Historical values for delta/trend comparison            |
| `query_reference_ranges`   | `{analyte: str, context?: {sex, flags}}`                               | Context-adjusted reference ranges                       |
| `flag_critical_value`      | `{patient_id: int, analyte: str, value: float, alert_level: str}`      | Flag and alert physician                                |
| `submit_diagnostic_report` | `{patient_id: int, icd_code: str, severity: str, clinical_notes: str}` | Final diagnosis                                         |

## Observation Space

```python
class PathologyObservation(Observation):
    output: str   # Structured JSON response from the LIMS
    error: str    # Error message if command failed, else empty
```

## Difficulty Tiers

### EASY — Critical Value Identification (3 variants)

- **Hyperkalemia**: Diabetic on Metformin + Lisinopril with K+ 7.2 mEq/L → `E87.5 / CRITICAL`
- **Acute MI**: ED patient with Troponin I 2.8 ng/mL (normal <0.04); BNP red herring → `I21.9 / CRITICAL`
- **Severe Anemia**: Elderly female with Hb 6.2 g/dL, microcytic pancytopenia pattern → `D64.9 / CRITICAL`

### MEDIUM — Context-Dependent Interpretation (3 variants)

- **Pregnancy Hb**: 29F at 28 weeks with Hb 11.5 — looks like anemia but is normal for pregnancy. False-positive anemia = harsh penalty.
- **Warfarin INR**: Mechanical valve patient with INR 4.8 — supratherapeutic, not DIC. Agent must check meds → `T45.515A / WARNING`
- **Drug-Induced K+**: CKD patient on ACE inhibitor + K supplement with K+ 5.8 — drug-induced, not emergency → `E87.5 / WARNING`

### HARD — Multi-Panel Syndrome Recognition (2 variants)

- **DIC**: ICU sepsis patient, 3 lab panels (CBC + Coag + CMP), 5+ abnormal values including schistocytes on smear → `D65 / CRITICAL`
- **Tumor Lysis Syndrome**: Post-chemo lymphoma patient with classic tetrad (↑K+, ↑PO4, ↓Ca, ↑Uric Acid) + AKI → `E88.3 / CRITICAL`

## Grading

- **100% deterministic** — no subjective text matching, all grading is based on verifiable actions
- Fractional scoring (0.0–1.0) with dense rewards for investigative steps (demographics, medications, labs, references, previous results)
- **ICD-10 accuracy** carries highest weight (55-70% per scenario)
- Partial credit for clinically reasonable alternate ICD codes (e.g., D68.3 instead of T45.515A for Warfarin)
- Penalties for false-positive diagnoses (e.g., diagnosing anemia in a healthy pregnant patient)
- Wrong answers are penalized down to 0.00

## Score Breakdown

When an episode ends, the environment automatically appends a **detailed human-readable score breakdown** to the final observation. This explains exactly:

- Which investigation steps the agent completed (✓) or missed (✗)
- What ICD code was submitted vs. what was expected
- Whether severity was correct
- How each component contributed to the final score

This gives judges and developers immediate, transparent feedback on *why* a model earned its score.

## Trajectory Reports

The `inference.py` baseline automatically saves full trajectory reports as JSON files to `trajectories/` after each run. Each report includes:

- Model name, task level, timestamp
- Step-by-step log: command, arguments, raw LLM output, env response, reward delta
- Final score and total steps used
- Reports persist across multiple runs with different models for comparison

## Setup & Usage

### Quick Start

```bash
# 1. Clone the Space
git clone https://huggingface.co/spaces/yatin-superintelligence/pathology-env
cd pathology-env

# 2. Build the Docker image
docker build -t pathology_env_env:latest .

# 3. Set your credentials and pick any model
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=google/gemma-4-26B-A4B-it:novita

# 4. Run — all 3 levels run automatically
python inference.py
```

**What you'll see:**

- `[START]`, `[STEP]`, `[END]` structured logs on stdout (auto-grader compatible)
- A final summary table with ✅/❌ per task level
- Trajectory JSON files saved to `trajectories/` for detailed analysis
- Score breakdown in the final observation explaining exactly why points were earned or lost

You can swap `MODEL_NAME` and `API_BASE_URL` to test any OpenAI-compatible model provider.

### Docker (Manual)

```bash
docker build -t pathology_env_env:latest .
docker run --rm -p 8000:8000 -e TASK_LEVEL=easy pathology_env_env:latest
curl http://localhost:8000/health
```

### Without Docker

```bash
uv sync
uv run server
```

### Inference Script Details

The inference script:

- Reads `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` from environment variables
- Uses the standard OpenAI client (`AsyncOpenAI`)
- Runs all 3 difficulty levels (easy → medium → hard) sequentially
- Emits mandatory `[START]`, `[STEP]`, `[END]` structured logs to stdout
- Saves trajectory JSON reports for post-run analysis
- Handles malformed LLM output gracefully (JSON extraction with balanced-brace parsing)

## Baseline Scores

| Task   | Model       | Score | Steps |
| ------ | ----------- | ----- | ----- |
| Easy   | Gemma-4-26B | 1.00  | 9     |
| Medium | Gemma-4-26B | 0.95  | 9     |
| Hard   | Gemma-4-26B | 0.58  | 11    |

*Easy tasks are solvable by most models. Medium requires cross-referencing medications/demographics to avoid false positives. Hard requires synthesizing 3+ lab panels into a rare syndrome diagnosis (DIC/TLS) — challenging even for frontier models.*

## Specs

- Full OpenEnv compliance (`step()` / `reset()` / `state()`)
- Pydantic `PathologyAction` / `PathologyObservation` models
- `openenv validate` ✅
- Baseline `inference.py` with OpenAI client and structured `[START]/[STEP]/[END]` logging
- Automatic score breakdown on episode completion
- Trajectory report generation for every run
- Graceful error handling — invalid commands return helpful error observations without crashing
- SQLite in-memory database — runs on 2 vCPU / 8 GB RAM, no GPU required

## Author

Built by **Yatin Taneja** for the Meta OpenEnv Hackathon (Round 1).
