"""Baseline inference script for the Blood Pathology LIMS Environment.

Runs an LLM agent through easy/medium/hard clinical diagnostic scenarios.
Uses structured tool definitions and handles JSON parsing robustly.

Log format follows the mandatory [START], [STEP], [END] plain-text specification
from the hackathon guidelines.
"""
import asyncio
import os
import json
import re
from typing import List
from client import PathologyEnv
from models import PathologyAction
from openai import OpenAI, AsyncOpenAI

# ──────────────────────────────────────────────
# Required environment variables
# Defaults are set only for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
# ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-26B-A4B-it:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")
API_KEY = HF_TOKEN

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
# HF Space URL for direct connection when Docker image isn't available
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://yatin-superintelligence-pathology-env.hf.space")
BENCHMARK = "pathology_env"
MAX_STEPS = 20
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.8


# ──────────────────────────────────────────────
# MANDATORY LOG FORMAT — plain text, bracket-prefixed
# Format spec from hackathon troubleshooting guide:
#   [START] task=<name> env=<benchmark> model=<model>
#   [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
#
# Rules:
#   - Booleans: lowercase true/false
#   - Rewards: exactly 2 decimal places
#   - Error: raw string or null
#   - Single line per event
# ──────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    done_str = "true" if done else "false"
    error_str = str(error) if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ──────────────────────────────────────────────
# Structured tool definitions for the LLM
# ──────────────────────────────────────────────
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


def parse_llm_response(raw: str) -> dict:
    """Robustly parse LLM response to extract JSON command.

    Handles: raw JSON, markdown fences, nested JSON objects, trailing text.
    """
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON object by tracking balanced braces
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = -1

    return {"command": "invalid", "arguments": {}}


async def run_task(level: str):
    """Run a single diagnostic task at the given difficulty level."""
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if LOCAL_IMAGE_NAME:
        print(f"[DEBUG] Using Docker image: {LOCAL_IMAGE_NAME}", flush=True)
        try:
            env = await PathologyEnv.from_docker_image(LOCAL_IMAGE_NAME, env_vars={"TASK_LEVEL": level})
        except TypeError:
            env = await PathologyEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        print(f"[DEBUG] No LOCAL_IMAGE_NAME set, connecting to HF Space: {HF_SPACE_URL}", flush=True)
        env = PathologyEnv(HF_SPACE_URL, connect_timeout_s=30, message_timeout_s=120)
        await env.connect()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0
    trajectory = []  # Full trajectory for report

    log_start(task=level, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_level=level)
        current_obs = result.observation.output

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Environment ready.\n{current_obs}\n\nBegin by listing pending cases."}
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, max_tokens=512, temperature=0.0
                )
                raw = response.choices[0].message.content.strip()
            except Exception as exc:
                print(f"[DEBUG] Model request failed: {exc}", flush=True)
                raw = '{"command": "list_pending_cases", "arguments": {}}'

            data = parse_llm_response(raw)
            action = PathologyAction(
                command=data.get("command", ""),
                arguments=data.get("arguments", {})
            )

            messages.append({"role": "assistant", "content": raw})

            # Execute step with reconnection on WebSocket drop
            try:
                result = await env.step(action)
            except Exception as ws_err:
                print(f"[DEBUG] WebSocket error, reconnecting: {ws_err}", flush=True)
                try:
                    await env.close()
                except Exception:
                    pass
                env = PathologyEnv(HF_SPACE_URL, connect_timeout_s=30, message_timeout_s=120)
                await env.connect()
                result = await env.reset(task_level=level)
                result = await env.step(action)

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done or False
            error = obs.error if obs.error else None

            # Store incremental delta (our graders return cumulative scores)
            delta = reward - last_reward
            rewards.append(delta)
            last_reward = reward
            steps_taken = step

            # Log action as a single string
            action_str = f"{action.command}({json.dumps(action.arguments)})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Record step in trajectory
            trajectory.append({
                "step": step,
                "command": action.command,
                "arguments": action.arguments,
                "reward": round(reward, 4),
                "delta": round(delta, 4),
                "done": done,
                "error": error,
                "env_output": obs.output[:500] if obs.output else "",
                "raw_llm": raw[:300],
            })

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            feedback = f"Output: {obs.output}"
            if obs.error:
                feedback += f"\nError: {obs.error}"
            feedback += f"\nReward: {reward} | Done: {done}"
            messages.append({"role": "user", "content": feedback})

            if done:
                break

        # sum(rewards) = final cumulative score since rewards stores deltas
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.01), 0.99)  # clamp to (0, 1) — strictly between, safe at 2 decimal places
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    # Save trajectory report
    _save_trajectory_report(level, score, steps_taken, trajectory)
    return score


def _save_trajectory_report(level: str, score: float, steps: int, trajectory: list):
    """Save a full trajectory report as JSON for post-run analysis."""
    from datetime import datetime
    os.makedirs("trajectories", exist_ok=True)
    model_short = MODEL_NAME.replace("/", "_").replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"trajectories/{timestamp}_{model_short}_{level}.json"

    report = {
        "model": MODEL_NAME,
        "task_level": level,
        "score": round(score, 4),
        "steps_used": steps,
        "max_steps": MAX_STEPS,
        "timestamp": timestamp,
        "trajectory": trajectory,
    }

    with open(filename, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Trajectory report saved: {filename}", flush=True)


async def main():
    """Run all three difficulty levels sequentially and produce a summary."""
    scores = {}
    for level in ["easy", "medium", "hard"]:
        score = await run_task(level)
        scores[level] = score

    # Print summary and save combined report
    print("\n" + "=" * 60, flush=True)
    print(f"MODEL: {MODEL_NAME}", flush=True)
    print("=" * 60, flush=True)
    for level, score in scores.items():
        status = "✅" if score >= SUCCESS_SCORE_THRESHOLD else "❌"
        print(f"  {status} {level:8s}: {score:.2f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  Average:    {avg:.2f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[END] success=false steps=0 score=0.00 rewards= error={exc}", flush=True)
        raise
