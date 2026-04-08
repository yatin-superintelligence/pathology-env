"""Run all 8 scenarios with the real LLM — PROPERLY controls variants via env vars."""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from server.pathology_env_environment import PathologyEnvironment
from models import PathologyAction

API_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemma-4-26b-a4b-it"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MAX_STEPS = 20

TOOLS = [
    {"name": "list_pending_cases", "args": {}, "description": "List all cases assigned to the AI agent."},
    {"name": "get_patient_demographics", "args": {"patient_id": "int"}, "description": "Get patient info including age, sex, medical history, and clinical flags."},
    {"name": "get_medications", "args": {"patient_id": "int"}, "description": "Get active medications for a patient. Critical for drug-lab interactions."},
    {"name": "get_lab_orders", "args": {"patient_id": "int"}, "description": "List all lab orders (panels) for a patient."},
    {"name": "get_lab_results", "args": {"order_id": "string"}, "description": "Get detailed results for a specific lab order."},
    {"name": "get_previous_results", "args": {"patient_id": "int", "analyte": "string (optional)"}, "description": "Get historical lab values for delta/trend comparison."},
    {"name": "query_reference_ranges", "args": {"analyte": "string", "context": {"sex": "M/F", "flags": "[...]"}}, "description": "Get reference ranges adjusted for sex, pregnancy, and therapeutic context."},
    {"name": "flag_critical_value", "args": {"patient_id": "int", "analyte": "string", "value": "float", "alert_level": "string"}, "description": "Flag a critical lab value and notify the attending physician."},
    {"name": "submit_diagnostic_report", "args": {"patient_id": "int", "icd_code": "string", "severity": "INFO|WARNING|HIGH|CRITICAL", "clinical_notes": "string"}, "description": "Submit final diagnosis with ICD-10 code, severity, and clinical reasoning."},
]

SYSTEM_PROMPT = f"""You are an AI Pathology Agent in a hospital Laboratory Information Management System (LIMS).
You must review pending cases, query patient data, interpret lab results using
proper clinical context (demographics, medications, reference ranges, previous results),
flag any critical values, and submit a final diagnostic report with an ICD-10 code.

AVAILABLE TOOLS:
{json.dumps(TOOLS, indent=2)}

CLINICAL WORKFLOW:
1. list_pending_cases → identify STAT cases first
2. get_patient_demographics → check age, sex, flags (pregnancy, warfarin, etc.)
3. get_medications → identify drugs that affect lab interpretation
4. get_lab_orders → find all panels ordered
5. get_lab_results → review each panel's results
6. get_previous_results → check trends/deltas for concerning analytes
7. query_reference_ranges → get context-adjusted ranges (pass sex + flags from demographics)
8. flag_critical_value → alert if any value exceeds critical thresholds
9. submit_diagnostic_report → final diagnosis with ICD-10 code and clinical notes

RULES:
- Always check demographics BEFORE interpreting results (sex/age/pregnancy affect ranges)
- Always check medications (some drugs cause expected lab abnormalities)
- Check previous results for delta changes when available
- Flag critical values BEFORE submitting your report
- Use proper ICD-10 codes. If all normal for context, use icd_code='NORMAL'
- severity: INFO | WARNING | HIGH | CRITICAL
- Focus on the STAT priority case(s) first

Response format: {{"command": "...", "arguments": {{...}}}}
Respond ONLY with raw JSON, no markdown fences or explanation."""


def parse_llm_response(raw):
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return {"command": "invalid", "arguments": {}}


SCENARIOS = [
    ("easy", "Easy-A: Hyperkalemia"),
    ("medium", "Medium (random variant)"),
    ("hard", "Hard (random variant)"),
]


def run_scenario(level, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # SET THE ENV VAR so reset() picks the right level
    os.environ["TASK_LEVEL"] = level

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = PathologyEnvironment()
    reset_obs = env.reset()  # This now correctly uses TASK_LEVEL env var

    actual_variant = env.task_variant
    variant_labels = {
        "easy": ["Hyperkalemia", "Acute MI", "Severe Anemia"],
        "medium": ["Pregnancy Hb", "Warfarin INR", "Drug-Induced K+"],
        "hard": ["DIC", "Tumor Lysis Syndrome"],
    }
    vname = variant_labels.get(level, ["?"])[actual_variant]
    print(f"  → Random variant picked: {vname} (variant={actual_variant})")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Environment ready.\n{reset_obs.output}\n\nBegin by listing pending cases."}
    ]

    last_reward = 0.0
    reward = 0.0
    steps_used = 0

    for step in range(1, MAX_STEPS + 1):
        raw = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, max_tokens=512, temperature=0.0
                )
                raw = response.choices[0].message.content
                if raw:
                    raw = raw.strip()
                    break
                else:
                    print(f"  Step {step}: ⏳ Empty response, retrying...")
                    time.sleep(2)
            except Exception as exc:
                if "429" in str(exc) and attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"  Step {step}: ⏳ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Step {step}: ❌ API ERROR: {exc}")
                    break
        if not raw:
            break

        data = parse_llm_response(raw)
        cmd = data.get("command", "")
        args = data.get("arguments", {})

        action = PathologyAction(command=cmd, arguments=args)
        obs = env.step(action)
        reward = obs.reward or 0.0
        done = obs.done or False
        delta = reward - last_reward
        last_reward = reward
        steps_used = step

        status = f"Δ+{delta:.2f}" if delta > 0 else ""
        err = f" ERR:{obs.error[:80]}" if obs.error else ""
        print(f"  Step {step:2d}: {cmd:30s} reward={reward:.2f} {status}{err}")
        # VERBOSE: show what the model actually sent
        print(f"           Args: {json.dumps(args, ensure_ascii=False)[:200]}")
        if cmd in ("flag_critical_value", "submit_diagnostic_report"):
            print(f"           >>> RAW LLM: {raw[:300]}")
        # Show env output snippet for key commands
        if cmd == "get_patient_demographics" and obs.output:
            print(f"           ENV: {obs.output[:200]}")
        if cmd == "get_medications" and obs.output:
            print(f"           ENV: {obs.output[:200]}")
        if obs.error:
            print(f"           ENV ERROR: {obs.error[:200]}")

        messages.append({"role": "assistant", "content": raw})
        feedback = f"Output: {obs.output}"
        if obs.error:
            feedback += f"\nError: {obs.error}"
        feedback += f"\nReward: {reward} | Done: {done}"
        messages.append({"role": "user", "content": feedback})

        if done:
            break

    if env.db:
        env.db.close()

    return f"{label} → {vname}", reward, steps_used


if __name__ == "__main__":
    results = []
    start = time.time()

    for level, label in SCENARIOS:
        name, score, steps = run_scenario(level, label)
        results.append((name, score, steps))

    elapsed = time.time() - start

    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS — 3 Task Levels ({elapsed:.0f}s total)")
    print(f"{'='*70}")
    for name, score, steps in results:
        icon = "✅" if score >= 0.95 else ("⚠️" if score >= 0.60 else "❌")
        print(f"  {icon} {name:45s}  score={score:.2f}  steps={steps}")

    avg = sum(s for _, s, _ in results) / len(results)
    print(f"{'─'*70}")
    print(f"  Average: {avg:.2f}")
    print(f"{'='*70}")
