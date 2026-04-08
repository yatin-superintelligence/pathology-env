"""Deterministic grader test — verifies all 8 scenarios with perfect play."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from server.pathology_env_environment import PathologyEnvironment
from models import PathologyAction

def simulate_perfect(level, variant):
    """Simulate perfect agent play for a specific scenario."""
    os.environ["TASK_LEVEL"] = level
    env = PathologyEnvironment()
    env.reset()
    # Force the variant we want to test
    env.task_variant = variant
    env._init_db()
    seeders = {
        ("easy", 0): env._seed_task_easy,
        ("easy", 1): env._seed_task_easy_b,
        ("easy", 2): env._seed_task_easy_c,
        ("medium", 0): env._seed_task_medium,
        ("medium", 1): env._seed_task_medium_b,
        ("medium", 2): env._seed_task_medium_c,
        ("hard", 0): env._seed_task_hard,
        ("hard", 1): env._seed_task_hard_b,
    }
    seeders[(level, variant)]()
    # Clear tracking from reset
    env.submitted_reports = []
    env.flagged_criticals = []
    env.queried_demographics = set()
    env.queried_medications = set()
    env.queried_lab_results = set()
    env.queried_previous = set()
    env.queried_references = set()

    # Perfect play sequences
    if level == "easy" and variant == 0:  # Hyperkalemia
        actions = [
            ("get_patient_demographics", {"patient_id": 1001}),
            ("get_medications", {"patient_id": 1001}),
            ("get_lab_results", {"order_id": "ORD-E001"}),
            ("query_reference_ranges", {"analyte": "Potassium", "context": {"sex": "M"}}),
            ("flag_critical_value", {"patient_id": 1001, "analyte": "Potassium", "value": 7.2, "alert_level": "CRITICAL"}),
            ("submit_diagnostic_report", {"patient_id": 1001, "icd_code": "E87.5", "severity": "CRITICAL",
                "clinical_notes": "Critical hyperkalemia K+ 7.2. Patient on ACE inhibitor with diabetes."}),
        ]
    elif level == "easy" and variant == 1:  # Acute MI
        actions = [
            ("get_patient_demographics", {"patient_id": 1005}),
            ("get_medications", {"patient_id": 1005}),
            ("get_lab_results", {"order_id": "ORD-EB01"}),
            ("query_reference_ranges", {"analyte": "Troponin_I", "context": {"sex": "M"}}),
            ("flag_critical_value", {"patient_id": 1005, "analyte": "Troponin_I", "value": 2.8, "alert_level": "CRITICAL"}),
            ("submit_diagnostic_report", {"patient_id": 1005, "icd_code": "I21.9", "severity": "CRITICAL",
                "clinical_notes": "Troponin I 2.8 ng/mL, CK-MB 45. Acute myocardial infarction."}),
        ]
    elif level == "easy" and variant == 2:  # Severe Anemia
        actions = [
            ("get_patient_demographics", {"patient_id": 1006}),
            ("get_lab_results", {"order_id": "ORD-EC01"}),
            ("query_reference_ranges", {"analyte": "Hemoglobin", "context": {"sex": "F"}}),
            ("get_previous_results", {"patient_id": 1006}),
            ("flag_critical_value", {"patient_id": 1006, "analyte": "Hemoglobin", "value": 6.2, "alert_level": "CRITICAL_LOW"}),
            ("submit_diagnostic_report", {"patient_id": 1006, "icd_code": "D50.9", "severity": "CRITICAL",
                "clinical_notes": "Severe iron deficiency anemia Hb 6.2, declining from 11.0."}),
        ]
    elif level == "medium" and variant == 0:  # Pregnancy Hb
        actions = [
            ("get_patient_demographics", {"patient_id": 1002}),
            ("get_medications", {"patient_id": 1002}),
            ("query_reference_ranges", {"analyte": "Hemoglobin", "context": {"sex": "F", "flags": ["pregnant"]}}),
            ("submit_diagnostic_report", {"patient_id": 1002, "icd_code": "NORMAL", "severity": "INFO",
                "clinical_notes": "Hb 11.5 is normal for pregnant patient in 2nd trimester (adjusted range 11.0-14.0). "
                                  "Glucose 135 is borderline for gestational diabetes screening — monitor GDM."}),
        ]
    elif level == "medium" and variant == 1:  # Warfarin INR
        actions = [
            ("get_patient_demographics", {"patient_id": 1007}),
            ("get_medications", {"patient_id": 1007}),
            ("query_reference_ranges", {"analyte": "INR", "context": {"sex": "M", "flags": ["on_warfarin"]}}),
            ("get_previous_results", {"patient_id": 1007}),
            ("get_lab_results", {"order_id": "ORD-MB01"}),
            ("submit_diagnostic_report", {"patient_id": 1007, "icd_code": "T45.515A", "severity": "WARNING",
                "clinical_notes": "Supratherapeutic INR 4.8 (mechanical valve target 2.5-3.5). On Warfarin 7.5mg. "
                                  "Recommend dose adjustment or hold. Previous INR was 3.0."}),
        ]
    elif level == "medium" and variant == 2:  # Drug-Induced K+
        actions = [
            ("get_patient_demographics", {"patient_id": 1008}),
            ("get_medications", {"patient_id": 1008}),
            ("query_reference_ranges", {"analyte": "Potassium", "context": {"sex": "F"}}),
            ("get_previous_results", {"patient_id": 1008}),
            ("get_lab_results", {"order_id": "ORD-MC01"}),
            ("submit_diagnostic_report", {"patient_id": 1008, "icd_code": "E87.5", "severity": "WARNING",
                "clinical_notes": "Drug-induced hyperkalemia K+ 5.8. Patient on Lisinopril (ACE inhibitor) and KCl supplement "
                                  "with CKD stage 3a (renal impairment). Medication review recommended."}),
        ]
    elif level == "hard" and variant == 0:  # DIC
        actions = [
            ("get_patient_demographics", {"patient_id": 1004}),
            ("get_medications", {"patient_id": 1004}),
            ("get_lab_results", {"order_id": "ORD-H001"}),
            ("get_lab_results", {"order_id": "ORD-H002"}),
            ("get_lab_results", {"order_id": "ORD-H003"}),
            ("get_previous_results", {"patient_id": 1004}),
            ("submit_diagnostic_report", {"patient_id": 1004, "icd_code": "D65", "severity": "CRITICAL",
                "clinical_notes": "Disseminated intravascular coagulation (DIC). Platelet 42 (was 185), "
                                  "fibrinogen 95 (<100), D-dimer 8.4, PT prolonged. Schistocytes 3+ on smear. "
                                  "Consumption coagulopathy in setting of sepsis."}),
        ]
    elif level == "hard" and variant == 1:  # TLS
        actions = [
            ("get_patient_demographics", {"patient_id": 1009}),
            ("get_medications", {"patient_id": 1009}),
            ("get_lab_results", {"order_id": "ORD-HB01"}),
            ("get_lab_results", {"order_id": "ORD-HB02"}),
            ("get_previous_results", {"patient_id": 1009}),
            ("submit_diagnostic_report", {"patient_id": 1009, "icd_code": "E88.3", "severity": "CRITICAL",
                "clinical_notes": "Tumor lysis syndrome (TLS) post-chemo for lymphoma. Classic Cairo-Bishop tetrad: "
                                  "hyperuricemia (15.2), hyperphosphatemia (8.2), hypocalcemia (6.5), hyperkalemia (6.8). "
                                  "LDH 1200 confirms cell lysis. AKI from urate precipitation (Cr 2.8, was 0.9)."}),
        ]

    last_obs = None
    for cmd, args in actions:
        last_obs = env.step(PathologyAction(command=cmd, arguments=args))

    return last_obs.reward, last_obs.done


def test_wrong_answers():
    """Test that wrong answers get properly penalized."""
    print("\n" + "="*60)
    print("WRONG ANSWER TESTS")
    print("="*60)

    # Easy-A: Submit wrong ICD
    os.environ["TASK_LEVEL"] = "easy"
    env = PathologyEnvironment()
    env.reset()
    env.task_variant = 0
    env._init_db()
    env._seed_task_easy()
    env.submitted_reports = []; env.flagged_criticals = []; env.queried_demographics = set()
    env.queried_medications = set(); env.queried_lab_results = set(); env.queried_previous = set()
    env.queried_references = set()
    obs = env.step(PathologyAction(command="submit_diagnostic_report",
        arguments={"patient_id": 1001, "icd_code": "Z00.00", "severity": "INFO", "clinical_notes": "Normal"}))
    print(f"  Easy-A wrong ICD (Z00.00):    score={obs.reward:.2f}  (should be ≤0.00)")
    assert obs.reward <= 0.0, f"Wrong! Score should be ≤0 but got {obs.reward}"

    # Medium-A: False positive anemia in pregnancy
    os.environ["TASK_LEVEL"] = "medium"
    env = PathologyEnvironment()
    env.reset()
    env.task_variant = 0
    env._init_db()
    env._seed_task_medium()
    env.submitted_reports = []; env.flagged_criticals = []; env.queried_demographics = set()
    env.queried_medications = set(); env.queried_lab_results = set(); env.queried_previous = set()
    env.queried_references = set()
    env.flagged_criticals.append({"patient_id": 1002, "analyte": "Hemoglobin", "value": 11.5, "alert_level": "CRITICAL"})
    obs = env.step(PathologyAction(command="submit_diagnostic_report",
        arguments={"patient_id": 1002, "icd_code": "D64.9", "severity": "CRITICAL", "clinical_notes": "Anemia"}))
    print(f"  Medium-A false anemia+crit:   score={obs.reward:.2f}  (should be ≤0.00)")
    assert obs.reward <= 0.0, f"Wrong! Score should be ≤0 but got {obs.reward}"

    # Medium-B: Diagnose DIC instead of Warfarin
    os.environ["TASK_LEVEL"] = "medium"
    env = PathologyEnvironment()
    env.reset()
    env.task_variant = 1
    env._init_db()
    env._seed_task_medium_b()
    env.submitted_reports = []; env.flagged_criticals = []; env.queried_demographics = set()
    env.queried_medications = set(); env.queried_lab_results = set(); env.queried_previous = set()
    env.queried_references = set()
    obs = env.step(PathologyAction(command="submit_diagnostic_report",
        arguments={"patient_id": 1007, "icd_code": "D65", "severity": "CRITICAL", "clinical_notes": "DIC"}))
    print(f"  Medium-B false DIC:           score={obs.reward:.2f}  (should be ≤0.00)")
    assert obs.reward <= 0.0, f"Wrong! Score should be ≤0 but got {obs.reward}"

    print("  ✅ All wrong-answer penalties verified!")


if __name__ == "__main__":
    print("="*60)
    print("DETERMINISTIC GRADER TEST — All 8 Scenarios (New Weights)")
    print("="*60)

    scenarios = [
        ("easy", 0, "Easy-A  Hyperkalemia"),
        ("easy", 1, "Easy-B  Acute MI"),
        ("easy", 2, "Easy-C  Severe Anemia"),
        ("medium", 0, "Medium-A Pregnancy Hb"),
        ("medium", 1, "Medium-B Warfarin INR"),
        ("medium", 2, "Medium-C Drug K+"),
        ("hard", 0, "Hard-A  DIC"),
        ("hard", 1, "Hard-B  TLS"),
    ]

    all_pass = True
    for level, variant, name in scenarios:
        score, done = simulate_perfect(level, variant)
        passed = score >= 0.90
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name:25s} score={score:.2f}  done={done}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"{'='*60}")
        print("🎉 ALL 8 SCENARIOS SCORE ≥ 0.90 — Graders verified!")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")
        print("❌ SOME SCENARIOS FAILED — check grader weights")
        print(f"{'='*60}")

    test_wrong_answers()
