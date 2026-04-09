"""Blood Pathology LIMS Environment — OpenEnv

Architecture:
    - SQLite in-memory database with 8 relational tables
    - 25 patients (9 target, 16 distractor) with realistic demographics
    - 40+ biomarkers with Mayo Clinic / LabCorp reference ranges
    - 8 clinical scenarios across 3 difficulty tiers (easy/medium/hard)
    - Random variant selection per episode via deterministic UUID seed
    - Dense reward tracking for investigative actions + final diagnosis

Scenarios:
    Easy:   Hyperkalemia (E87.5), Acute MI (I21.9), Severe Anemia (D64.9)
    Medium: Pregnancy Hb (NORMAL), Warfarin INR (T45.515A), Drug-Induced K+ (E87.5)
    Hard:   DIC (D65), Tumor Lysis Syndrome (E88.3)

Grading:
    Each grader sums to 1.0 max. Partial credit for investigation steps
    (demographics, medications, lab results, previous results, reference ranges).
    Penalties for false-positive diagnoses.
"""
import os
import json
import random
import sqlite3
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PathologyAction, PathologyObservation
except ImportError:
    from models import PathologyAction, PathologyObservation


# ──────────────────────────────────────────────
# REAL CLINICAL REFERENCE RANGES (Adult)
# Sources: Mayo Clinic Labs, LabCorp, ARUP
# ──────────────────────────────────────────────
REFERENCE_RANGES = {
    # ── Comprehensive Metabolic Panel (CMP) ──
    "Sodium":       {"unit": "mEq/L",  "min": 135, "max": 145, "crit_lo": 120, "crit_hi": 160},
    "Potassium":    {"unit": "mEq/L",  "min": 3.5, "max": 5.0, "crit_lo": 2.5, "crit_hi": 6.5},
    "Chloride":     {"unit": "mEq/L",  "min": 95,  "max": 108},
    "CO2":          {"unit": "mEq/L",  "min": 23,  "max": 30},
    "Glucose":      {"unit": "mg/dL",  "min": 70,  "max": 100, "crit_lo": 40, "crit_hi": 400},
    "BUN":          {"unit": "mg/dL",  "min": 7,   "max": 20},
    "Creatinine":   {"unit": "mg/dL",  "min_M": 0.7, "max_M": 1.3, "min_F": 0.5, "max_F": 1.1},
    "Calcium":      {"unit": "mg/dL",  "min": 8.4, "max": 10.2, "crit_lo": 6.0, "crit_hi": 13.0},
    "Albumin":      {"unit": "g/dL",   "min": 3.4, "max": 5.4},
    "Total_Protein": {"unit": "g/dL",  "min": 6.0, "max": 8.3},
    "ALP":          {"unit": "U/L",    "min": 20,  "max": 130},
    "ALT":          {"unit": "U/L",    "min": 4,   "max": 36},
    "AST":          {"unit": "U/L",    "min": 8,   "max": 33},
    "Bilirubin":    {"unit": "mg/dL",  "min": 0.2, "max": 1.2},
    # ── Complete Blood Count (CBC) ──
    "WBC":          {"unit": "K/uL",   "min": 4.5, "max": 11.0, "crit_lo": 2.0, "crit_hi": 30.0},
    "RBC":          {"unit": "M/uL",   "min_M": 4.5, "max_M": 5.5, "min_F": 4.0, "max_F": 5.0},
    "Hemoglobin":   {"unit": "g/dL",   "min_M": 13.8, "max_M": 17.2, "min_F": 12.1, "max_F": 15.1,
                     "min_F_pregnant": 11.0, "max_F_pregnant": 14.0, "crit_lo": 7.0, "crit_hi": 20.0},
    "Hematocrit":   {"unit": "%",      "min_M": 40.7, "max_M": 50.3, "min_F": 36.1, "max_F": 44.3},
    "Platelets":    {"unit": "K/uL",   "min": 150, "max": 400, "crit_lo": 20, "crit_hi": 1000},
    "MCV":          {"unit": "fL",     "min": 80,  "max": 100},
    "Neutrophils":  {"unit": "%",      "min": 40,  "max": 70},
    "Lymphocytes":  {"unit": "%",      "min": 20,  "max": 40},
    "Bands":        {"unit": "%",      "min": 0,   "max": 5},
    # ── Coagulation Panel ──
    "PT":           {"unit": "sec",    "min": 11.0, "max": 13.5},
    "PTT":          {"unit": "sec",    "min": 25.0, "max": 35.0},
    "INR":          {"unit": "ratio",  "min": 0.8,  "max": 1.1,
                     "therapeutic_min": 2.0, "therapeutic_max": 3.0,
                     "therapeutic_mechanical_valve_min": 2.5, "therapeutic_mechanical_valve_max": 3.5},
    "Fibrinogen":   {"unit": "mg/dL",  "min": 200, "max": 400, "crit_lo": 100},
    "D_Dimer":      {"unit": "ug/mL",  "min": 0.0, "max": 0.5},
    # ── Cardiac Markers ──
    "Troponin_I":   {"unit": "ng/mL",  "min": 0.0, "max": 0.04, "crit_hi": 0.4},
    "BNP":          {"unit": "pg/mL",  "min": 0,   "max": 100},
    "CK_MB":        {"unit": "ng/mL",  "min": 0,   "max": 5.0},
    # ── Lipid Panel ──
    "Total_Cholesterol": {"unit": "mg/dL", "desirable": 200, "borderline": 240},
    "LDL":          {"unit": "mg/dL",  "optimal": 100, "borderline": 160},
    "HDL":          {"unit": "mg/dL",  "low_M": 40, "low_F": 50},
    "Triglycerides": {"unit": "mg/dL", "normal": 150, "high": 200, "very_high": 500},
    # ── Thyroid ──
    "TSH":          {"unit": "mIU/L",  "min": 0.4, "max": 4.0},
    "Free_T4":      {"unit": "ng/dL",  "min": 0.8, "max": 1.8},
    # ── Iron Studies ──
    "Iron":         {"unit": "ug/dL",  "min": 60,  "max": 170},
    "Ferritin":     {"unit": "ng/mL",  "min_M": 12, "max_M": 300, "min_F": 12, "max_F": 150},
    "TIBC":         {"unit": "ug/dL",  "min": 250, "max": 370},
    # ── Urinalysis ──
    "Uric_Acid":    {"unit": "mg/dL",  "min_M": 3.4, "max_M": 7.0, "min_F": 2.4, "max_F": 6.0},
    "Phosphate":    {"unit": "mg/dL",  "min": 2.5, "max": 4.5},
    # ── Enzymes / Tumor Markers ──
    "LDH":          {"unit": "U/L",    "min": 140, "max": 280},
    # ── HbA1c ──
    "HbA1c":        {"unit": "%",      "normal": 5.7, "prediabetic": 6.5},
}


class PathologyEnvironment(Environment):
    """20x Blood Pathology LIMS Diagnostic Environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.db = None
        self.task_level = os.environ.get("TASK_LEVEL", "easy").lower()
        self.max_steps = 20
        self.submitted_reports = []
        self.flagged_criticals = []
        # Dense reward tracking: what investigative steps has the agent taken?
        self.queried_demographics = set()  # patient IDs
        self.queried_medications = set()   # patient IDs
        self.queried_lab_results = set()   # order IDs
        self.queried_previous = set()      # patient IDs
        self.queried_references = set()    # analyte names
        self.task_variant = 0               # which sub-scenario within this level

    # ────────────────────────────────────────
    # DATABASE SETUP
    # ────────────────────────────────────────
    def _init_db(self):
        if self.db:
            self.db.close()
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row

        self.db.executescript("""
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY, name TEXT, age INTEGER, sex TEXT,
                weight_kg REAL, ethnicity TEXT,
                medical_history TEXT, active_diagnoses TEXT, flags TEXT
            );
            CREATE TABLE medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER, drug_name TEXT, dose TEXT, route TEXT, frequency TEXT
            );
            CREATE TABLE lab_orders (
                order_id TEXT PRIMARY KEY, patient_id INTEGER,
                panel_name TEXT, ordered_by TEXT, order_date TEXT, status TEXT
            );
            CREATE TABLE lab_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT, patient_id INTEGER,
                analyte TEXT, value REAL, unit TEXT, flag TEXT,
                collected_at TEXT
            );
            CREATE TABLE previous_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER, analyte TEXT, value REAL, unit TEXT,
                collected_at TEXT
            );
            CREATE TABLE diagnostic_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER, icd_code TEXT, severity TEXT,
                clinical_notes TEXT, submitted_at TEXT
            );
            CREATE TABLE critical_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER, analyte TEXT, value REAL,
                alert_level TEXT, acknowledged INTEGER DEFAULT 0
            );
            CREATE TABLE pending_cases (
                case_id TEXT PRIMARY KEY, patient_id INTEGER,
                description TEXT, priority TEXT, assigned_to TEXT
            );
        """)
        self._seed_patients()
        self._seed_medications()
        self.db.commit()

    def _seed_patients(self):
        patients = [
            # Target patients (used in tasks)
            (1001, "Marcus Chen", 62, "M", 88.5, "Asian", "Hypertension, Type 2 Diabetes", "E11.9,I10", "[]"),
            (1002, "Priya Sharma", 29, "F", 62.0, "South Asian", "Routine OB panel, G1P0 at 28 weeks", "O24.419", '["pregnant"]'),
            (1003, "James Wilson", 51, "M", 105.3, "Caucasian", "Obesity, Smoking 1ppd x 20yr, Family hx CAD", "E66.01", '["obese"]'),
            (1004, "Elena Vasquez", 44, "F", 70.2, "Hispanic", "Post-surgical (appendectomy), transferred from general ward", "A41.9", '["icu"]'),
            # New target patients for multi-scenario tasks
            (1005, "William Torres", 58, "M", 92.0, "Hispanic", "Chest pain, presented to ED 3hrs ago", "R07.9", "[]"),
            (1006, "Fatima Al-Rashid", 72, "F", 54.3, "Middle Eastern", "Chronic fatigue, progressive dyspnea on exertion", "R53.83", "[]"),
            (1007, "George Patterson", 68, "M", 80.1, "Caucasian", "Atrial fibrillation, mechanical heart valve", "I48.91", '["on_warfarin"]'),
            (1008, "Diana Reeves", 56, "F", 69.5, "African American", "Hypertension, CKD Stage 3a", "N18.31", '["ckd"]'),
            (1009, "Kevin Nakamura", 34, "M", 75.0, "Asian", "Recently diagnosed lymphoma, started chemo 48hrs ago", "C85.90", '["chemo"]'),
            # Distractor patients
            (2001, "Robert Kim", 35, "M", 78.0, "Asian", "Annual wellness exam", "Z00.00", "[]"),
            (2002, "Sarah O'Brien", 67, "F", 65.4, "Caucasian", "Osteoporosis, Vitamin D deficiency", "M81.0", "[]"),
            (2003, "Ahmed Hassan", 42, "M", 91.0, "Middle Eastern", "Mild dyslipidemia", "E78.5", "[]"),
            (2004, "Linda Park", 55, "F", 72.1, "Asian", "Hypothyroidism (controlled)", "E03.9", "[]"),
            (2005, "David Okafor", 48, "M", 83.6, "African", "Sickle cell trait (carrier, asymptomatic)", "D57.3", "[]"),
            (2006, "Maria Santos", 38, "F", 58.2, "Hispanic", "Iron deficiency anemia (treated)", "D50.9", "[]"),
            (2007, "Thomas Wright", 71, "M", 76.8, "Caucasian", "Atrial fibrillation on Warfarin", "I48.91", '["on_warfarin"]'),
            (2008, "Yuki Tanaka", 33, "F", 55.0, "Asian", "Healthy, blood donor screening", "Z00.00", "[]"),
            (2009, "Charles Brown", 59, "M", 95.2, "African American", "CKD Stage 2, Gout", "N18.2", '["ckd"]'),
            (2010, "Aisha Mohammed", 26, "F", 60.5, "African", "G6PD deficiency", "D55.0", "[]"),
            (3001, "Frank Miller", 45, "M", 82.0, "Caucasian", "Healthy", "Z00.00", "[]"),
            (3002, "Grace Liu", 52, "F", 64.0, "Asian", "Menopause", "N95.1", "[]"),
            (3003, "Ivan Petrov", 38, "M", 90.5, "Caucasian", "GERD", "K21.0", "[]"),
            (3004, "Nancy Drew", 60, "F", 71.3, "Caucasian", "Breast cancer (remission)", "Z85.3", "[]"),
            (3005, "Omar Farooq", 47, "M", 87.0, "South Asian", "Pre-diabetes", "R73.03", "[]"),
            (3006, "Patricia Gomez", 31, "F", 57.8, "Hispanic", "Anxiety disorder", "F41.1", "[]"),
        ]
        self.db.executemany(
            "INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?)", patients
        )

    def _seed_medications(self):
        meds = [
            # Patient 1001 - diabetes + hypertension
            (1001, "Metformin", "1000mg", "PO", "BID"),
            (1001, "Lisinopril", "20mg", "PO", "QD"),
            (1001, "Atorvastatin", "40mg", "PO", "QHS"),
            (1001, "Aspirin", "81mg", "PO", "QD"),
            # Patient 1002 - pregnant
            (1002, "Prenatal Vitamins", "1 tab", "PO", "QD"),
            (1002, "Folic Acid", "1mg", "PO", "QD"),
            # Patient 1003 - metabolic syndrome candidate
            (1003, "None", "", "", ""),
            # Patient 1004 - ICU sepsis
            (1004, "Piperacillin-Tazobactam", "4.5g", "IV", "Q6H"),
            (1004, "Norepinephrine", "0.1mcg/kg/min", "IV", "Continuous"),
            (1004, "Heparin", "5000U", "SC", "Q12H"),
            # Patient 1005 - chest pain (new easy scenario)
            (1005, "Aspirin", "325mg", "PO", "STAT"),
            (1005, "Nitroglycerin", "0.4mg", "SL", "PRN"),
            # Patient 1006 - chronic fatigue (new easy scenario)
            (1006, "None", "", "", ""),
            # Patient 1007 - mechanical valve on Warfarin (new medium scenario)
            (1007, "Warfarin", "7.5mg", "PO", "QD"),
            (1007, "Digoxin", "0.125mg", "PO", "QD"),
            # Patient 1008 - CKD + hypertension (new medium scenario)
            (1008, "Lisinopril", "10mg", "PO", "QD"),
            (1008, "Amlodipine", "5mg", "PO", "QD"),
            (1008, "Potassium Chloride", "20mEq", "PO", "QD"),  # K+ supplement + ACE = dangerous combo
            # Patient 1009 - lymphoma on chemo (new hard scenario)
            (1009, "Cyclophosphamide", "750mg/m2", "IV", "Cycle 1 Day 1"),
            (1009, "Allopurinol", "300mg", "PO", "QD"),  # for TLS prophylaxis (but it failed)
            # Patient 2007 - on Warfarin
            (2007, "Warfarin", "5mg", "PO", "QD"),
            (2007, "Metoprolol", "50mg", "PO", "BID"),
            # Patient 2009 - CKD + Gout
            (2009, "Allopurinol", "300mg", "PO", "QD"),
            (2009, "Losartan", "50mg", "PO", "QD"),
        ]
        self.db.executemany(
            "INSERT INTO medications (patient_id, drug_name, dose, route, frequency) VALUES (?,?,?,?,?)", meds
        )

    def _seed_task_easy(self):
        """EASY: Critical value identification.
        Patient 1001 has a dangerously high Potassium (7.2 mEq/L) - cardiac arrest risk.
        Most of the CMP is normal/mildly abnormal. Agent must find the critical K+,
        flag it, and submit the correct diagnosis.
        Expected: flag_critical_value for K+ 7.2, submit icd_code='E87.5' severity='CRITICAL'
        """
        c = self.db.cursor()
        order_id = "ORD-E001"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (order_id, 1001, "CMP", "Dr. Patel", "2026-04-07", "COMPLETED"))
        results = [
            (order_id, 1001, "Sodium", 138, "mEq/L", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "Potassium", 7.2, "mEq/L", "", "2026-04-07T08:00"),  # CRITICAL HIGH
            (order_id, 1001, "Chloride", 101, "mEq/L", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "CO2", 24, "mEq/L", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "Glucose", 142, "mg/dL", "HIGH", "2026-04-07T08:00"),  # expected for diabetic
            (order_id, 1001, "BUN", 22, "mg/dL", "HIGH", "2026-04-07T08:00"),       # mildly elevated
            (order_id, 1001, "Creatinine", 1.4, "mg/dL", "HIGH", "2026-04-07T08:00"),
            (order_id, 1001, "Calcium", 9.1, "mg/dL", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "Albumin", 3.8, "g/dL", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "Total_Protein", 7.0, "g/dL", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "ALP", 85, "U/L", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "ALT", 28, "U/L", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "AST", 30, "U/L", "NORMAL", "2026-04-07T08:00"),
            (order_id, 1001, "Bilirubin", 0.8, "mg/dL", "NORMAL", "2026-04-07T08:00"),
        ]
        c.executemany(
            "INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)",
            results
        )
        # Previous results for delta check context
        c.executemany(
            "INSERT INTO previous_results (patient_id,analyte,value,unit,collected_at) VALUES (?,?,?,?,?)",
            [
                (1001, "Potassium", 4.8, "mEq/L", "2026-03-15T09:00"),  # was normal last month
                (1001, "Creatinine", 1.2, "mg/dL", "2026-03-15T09:00"),
                (1001, "Glucose", 135, "mg/dL", "2026-03-15T09:00"),
            ]
        )
        # Neutral case description — agent must discover the medications and context itself
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-E001", 1001,
            "CMP panel results are ready for patient ID 1001. Please review all values and submit your findings.",
            "STAT", "AI-Agent"
        ))
        # Distractor case
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-E002", 2001,
            "Annual wellness CMP for patient ID 2001. Routine review.",
            "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_easy_b(self):
        """EASY-B: Acute MI detection via Troponin.
        Patient 1005 (58M) presented with chest pain. Cardiac markers show Troponin I = 2.8 ng/mL
        (critical threshold 0.4, normal <0.04). Must flag critical + diagnose I21.9 (STEMI).
        Red herring: mildly elevated BNP (could confuse with heart failure).
        """
        c = self.db.cursor()
        oid = "ORD-EB01"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid, 1005, "Cardiac_Panel+CMP", "Dr. Chen", "2026-04-07", "COMPLETED"))
        results = [
            (oid, 1005, "Troponin_I", 2.8, "ng/mL", "", "2026-04-07T14:00"),
            (oid, 1005, "CK_MB", 45.0, "ng/mL", "HIGH", "2026-04-07T14:00"),
            (oid, 1005, "BNP", 250, "pg/mL", "HIGH", "2026-04-07T14:00"),  # red herring
            (oid, 1005, "Sodium", 140, "mEq/L", "NORMAL", "2026-04-07T14:00"),
            (oid, 1005, "Potassium", 4.1, "mEq/L", "NORMAL", "2026-04-07T14:00"),
            (oid, 1005, "Glucose", 165, "mg/dL", "HIGH", "2026-04-07T14:00"),  # stress hyperglycemia
            (oid, 1005, "Creatinine", 1.0, "mg/dL", "NORMAL", "2026-04-07T14:00"),
        ]
        c.executemany("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)", results)
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-EB01", 1005,
            "Cardiac panel and CMP results for patient ID 1005. Review required.",
            "STAT", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-EB02", 2001, "Annual wellness CMP for patient ID 2001. Routine.", "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_easy_c(self):
        """EASY-C: Severe anemia detection.
        Patient 1006 (72F) has critically low Hemoglobin = 6.2 g/dL (crit threshold 7.0).
        CBC shows pancytopenia pattern. Must flag Hb as critical + diagnose D64.9 (anemia).
        """
        c = self.db.cursor()
        oid = "ORD-EC01"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid, 1006, "CBC_with_Diff", "Dr. Williams", "2026-04-07", "COMPLETED"))
        results = [
            (oid, 1006, "WBC", 3.8, "K/uL", "LOW", "2026-04-07T09:00"),
            (oid, 1006, "RBC", 2.1, "M/uL", "LOW", "2026-04-07T09:00"),
            (oid, 1006, "Hemoglobin", 6.2, "g/dL", "", "2026-04-07T09:00"),  # CRITICAL LOW
            (oid, 1006, "Hematocrit", 18.6, "%", "LOW", "2026-04-07T09:00"),
            (oid, 1006, "Platelets", 95, "K/uL", "LOW", "2026-04-07T09:00"),
            (oid, 1006, "MCV", 72, "fL", "LOW", "2026-04-07T09:00"),  # microcytic
            (oid, 1006, "Iron", 25, "ug/dL", "LOW", "2026-04-07T09:00"),
            (oid, 1006, "Ferritin", 5, "ng/mL", "LOW", "2026-04-07T09:00"),
        ]
        c.executemany("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)", results)
        c.executemany("INSERT INTO previous_results (patient_id,analyte,value,unit,collected_at) VALUES (?,?,?,?,?)", [
            (1006, "Hemoglobin", 9.5, "g/dL", "2026-02-01T10:00"),  # was already low but not critical
            (1006, "Hemoglobin", 11.0, "g/dL", "2025-08-15T10:00"),  # was normal 8 months ago
        ])
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-EC01", 1006,
            "CBC results for patient ID 1006. Review requested.",
            "STAT", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-EC02", 3001, "Routine panel for patient ID 3001.", "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_medium(self):
        """MEDIUM: Context-dependent threshold interpretation.
        Patient 1002 is a 29F who is 28 weeks pregnant.
        Her Hemoglobin is 11.5 g/dL. For a non-pregnant female range (12.1-15.1), this is LOW.
        But for a pregnant female (11.0-14.0), this is NORMAL (physiologic hemodilution).
        Agent MUST check patient demographics/flags before interpreting.
        Also: Glucose is 135 mg/dL — could be gestational diabetes (threshold 140 for 1hr GCT).
        Agent must correctly identify Hb as normal AND Glucose as borderline-but-below-threshold.
        Expected: submit icd_code='NORMAL' severity='INFO' (with note about monitoring glucose)
        """
        c = self.db.cursor()
        order_id = "ORD-M001"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (order_id, 1002, "CBC+CMP", "Dr. Gupta", "2026-04-07", "COMPLETED"))
        results = [
            # CBC
            (order_id, 1002, "WBC", 9.8, "K/uL", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "RBC", 3.9, "M/uL", "", "2026-04-07T10:00"),  # slightly low for non-preg, normal for preg
            (order_id, 1002, "Hemoglobin", 11.5, "g/dL", "", "2026-04-07T10:00"),  # KEY VALUE
            (order_id, 1002, "Hematocrit", 34.5, "%", "", "2026-04-07T10:00"),
            (order_id, 1002, "Platelets", 210, "K/uL", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "MCV", 88, "fL", "NORMAL", "2026-04-07T10:00"),
            # CMP values
            (order_id, 1002, "Sodium", 136, "mEq/L", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "Potassium", 4.0, "mEq/L", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "Glucose", 135, "mg/dL", "", "2026-04-07T10:00"),  # borderline GDM
            (order_id, 1002, "Creatinine", 0.6, "mg/dL", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "Calcium", 9.0, "mg/dL", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "ALT", 15, "U/L", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "AST", 18, "U/L", "NORMAL", "2026-04-07T10:00"),
            (order_id, 1002, "Albumin", 3.2, "g/dL", "", "2026-04-07T10:00"),  # low but normal in pregnancy
            # Iron studies (to test if agent over-diagnoses iron deficiency)
            (order_id, 1002, "Iron", 55, "ug/dL", "", "2026-04-07T10:00"),  # slightly low
            (order_id, 1002, "Ferritin", 18, "ng/mL", "", "2026-04-07T10:00"),  # borderline
            (order_id, 1002, "TIBC", 380, "ug/dL", "", "2026-04-07T10:00"),  # slightly elevated
        ]
        c.executemany(
            "INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)",
            results
        )
        # Neutral case description — does NOT mention pregnancy; agent must discover it
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-M001", 1002,
            "CBC and CMP results ready for patient ID 1002. Several values flagged borderline. Full interpretation requested.",
            "ROUTINE", "AI-Agent"
        ))
        # Distractor case
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-M002", 2007,
            "Coagulation panel results for patient ID 2007. Routine INR monitoring.",
            "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_medium_b(self):
        """MEDIUM-B: Warfarin supratherapeutic INR.
        Patient 1007 (68M) is on Warfarin for mechanical heart valve (therapeutic INR 2.5-3.5).
        His INR is 4.8 — supratherapeutic and bleeding risk. But NOT necessarily dangerous
        for someone on Warfarin (unlike a non-anticoagulated patient where 4.8 would be panic).
        Agent must check meds, realize patient is on Warfarin, apply therapeutic ranges,
        and diagnose as supratherapeutic (needs dose adjustment, not panic).
        Expected: icd_code='T45.515A' (adverse effect of anticoagulant) severity='WARNING'
        """
        c = self.db.cursor()
        oid = "ORD-MB01"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid, 1007, "Coagulation_Panel", "Dr. Baker", "2026-04-07", "COMPLETED"))
        results = [
            (oid, 1007, "PT", 28.5, "sec", "HIGH", "2026-04-07T11:00"),
            (oid, 1007, "PTT", 45.2, "sec", "HIGH", "2026-04-07T11:00"),
            (oid, 1007, "INR", 4.8, "ratio", "", "2026-04-07T11:00"),  # supratherapeutic
            (oid, 1007, "Fibrinogen", 280, "mg/dL", "NORMAL", "2026-04-07T11:00"),
        ]
        c.executemany("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)", results)
        c.executemany("INSERT INTO previous_results (patient_id,analyte,value,unit,collected_at) VALUES (?,?,?,?,?)", [
            (1007, "INR", 3.0, "ratio", "2026-03-20T10:00"),  # was in therapeutic range
            (1007, "INR", 2.8, "ratio", "2026-02-15T10:00"),
        ])
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-MB01", 1007,
            "Coagulation panel for patient ID 1007. Review required.",
            "ROUTINE", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-MB02", 2001, "Annual CMP for patient ID 2001.", "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_medium_c(self):
        """MEDIUM-C: Drug-induced hyperkalemia (not pathological).
        Patient 1008 (56F) has CKD Stage 3a, on Lisinopril (ACE inhibitor) + Potassium supplement.
        K+ is 5.8 mEq/L — elevated but explained by the drug combination.
        Agent must check meds, see ACE + K supplement + CKD, recognize this is drug-induced,
        NOT flag as critical (6.5 threshold), and recommend medication adjustment.
        Expected: icd_code='E87.5' severity='WARNING' (with note about drug-induced etiology)
        """
        c = self.db.cursor()
        oid = "ORD-MC01"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid, 1008, "CMP", "Dr. Patel", "2026-04-07", "COMPLETED"))
        results = [
            (oid, 1008, "Sodium", 137, "mEq/L", "NORMAL", "2026-04-07T08:00"),
            (oid, 1008, "Potassium", 5.8, "mEq/L", "HIGH", "2026-04-07T08:00"),
            (oid, 1008, "Chloride", 103, "mEq/L", "NORMAL", "2026-04-07T08:00"),
            (oid, 1008, "CO2", 21, "mEq/L", "LOW", "2026-04-07T08:00"),  # mild metabolic acidosis
            (oid, 1008, "BUN", 28, "mg/dL", "HIGH", "2026-04-07T08:00"),
            (oid, 1008, "Creatinine", 1.6, "mg/dL", "HIGH", "2026-04-07T08:00"),  # CKD baseline
            (oid, 1008, "Glucose", 92, "mg/dL", "NORMAL", "2026-04-07T08:00"),
            (oid, 1008, "Calcium", 9.0, "mg/dL", "NORMAL", "2026-04-07T08:00"),
        ]
        c.executemany("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)", results)
        c.executemany("INSERT INTO previous_results (patient_id,analyte,value,unit,collected_at) VALUES (?,?,?,?,?)", [
            (1008, "Potassium", 5.2, "mEq/L", "2026-03-01T09:00"),  # trending up
            (1008, "Creatinine", 1.5, "mg/dL", "2026-03-01T09:00"),  # stable CKD
        ])
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-MC01", 1008,
            "CMP results for patient ID 1008. Electrolyte abnormality noted.",
            "ROUTINE", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-MC02", 3002, "Routine panel for patient ID 3002.", "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_hard(self):
        """HARD: Multi-panel DIC (Disseminated Intravascular Coagulation) diagnosis.
        Patient 1004 is a 44F in ICU with sepsis post-appendectomy.
        DIC criteria require ALL of:
          - Platelets < 100 K/uL (she has 42)
          - PT elevated > 13.5 sec (she has 18.2)
          - PTT elevated > 35 sec (she has 52.1)
          - Fibrinogen < 150 mg/dL (she has 95)
          - D-Dimer elevated > 0.5 ug/mL (she has 8.4)
          - Schistocytes present on smear (noted in morphology)
        Red herrings: elevated WBC (expected in sepsis), low Albumin (expected in ICU),
        elevated Lactate. Agent must pull CBC, Coag panel, and CMP, synthesize across ALL
        panels, recognize the DIC pattern, and NOT just diagnose "sepsis."
        Expected: submit icd_code='D65' severity='CRITICAL'
        """
        c = self.db.cursor()
        # CBC order
        cbc_id = "ORD-H001"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (cbc_id, 1004, "CBC_with_Diff", "Dr. Rivera", "2026-04-07", "COMPLETED"))
        cbc = [
            (cbc_id, 1004, "WBC", 18.5, "K/uL", "HIGH", "2026-04-07T06:00"),
            (cbc_id, 1004, "RBC", 3.2, "M/uL", "LOW", "2026-04-07T06:00"),
            (cbc_id, 1004, "Hemoglobin", 9.1, "g/dL", "LOW", "2026-04-07T06:00"),
            (cbc_id, 1004, "Hematocrit", 27.3, "%", "LOW", "2026-04-07T06:00"),
            (cbc_id, 1004, "Platelets", 42, "K/uL", "CRITICAL_LOW", "2026-04-07T06:00"),
            (cbc_id, 1004, "MCV", 85, "fL", "NORMAL", "2026-04-07T06:00"),
            (cbc_id, 1004, "Neutrophils", 82, "%", "HIGH", "2026-04-07T06:00"),
            (cbc_id, 1004, "Lymphocytes", 10, "%", "LOW", "2026-04-07T06:00"),
            (cbc_id, 1004, "Bands", 12, "%", "HIGH", "2026-04-07T06:00"),
        ]
        # Coag panel order
        coag_id = "ORD-H002"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (coag_id, 1004, "Coagulation_Panel", "Dr. Rivera", "2026-04-07", "COMPLETED"))
        coag = [
            (coag_id, 1004, "PT", 18.2, "sec", "HIGH", "2026-04-07T06:00"),
            (coag_id, 1004, "PTT", 52.1, "sec", "HIGH", "2026-04-07T06:00"),
            (coag_id, 1004, "INR", 1.6, "ratio", "HIGH", "2026-04-07T06:00"),
            (coag_id, 1004, "Fibrinogen", 95, "mg/dL", "CRITICAL_LOW", "2026-04-07T06:00"),
            (coag_id, 1004, "D_Dimer", 8.4, "ug/mL", "CRITICAL_HIGH", "2026-04-07T06:00"),
        ]
        # CMP order
        cmp_id = "ORD-H003"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (cmp_id, 1004, "CMP", "Dr. Rivera", "2026-04-07", "COMPLETED"))
        cmp = [
            (cmp_id, 1004, "Sodium", 133, "mEq/L", "LOW", "2026-04-07T06:00"),
            (cmp_id, 1004, "Potassium", 4.8, "mEq/L", "NORMAL", "2026-04-07T06:00"),
            (cmp_id, 1004, "Glucose", 95, "mg/dL", "NORMAL", "2026-04-07T06:00"),
            (cmp_id, 1004, "BUN", 35, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (cmp_id, 1004, "Creatinine", 1.8, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (cmp_id, 1004, "Albumin", 2.1, "g/dL", "LOW", "2026-04-07T06:00"),
            (cmp_id, 1004, "ALT", 65, "U/L", "HIGH", "2026-04-07T06:00"),
            (cmp_id, 1004, "AST", 88, "U/L", "HIGH", "2026-04-07T06:00"),
            (cmp_id, 1004, "Bilirubin", 2.8, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (cmp_id, 1004, "Calcium", 7.8, "mg/dL", "LOW", "2026-04-07T06:00"),
        ]
        for batch in [cbc, coag, cmp]:
            c.executemany(
                "INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)",
                batch
            )
        # Previous results (platelets were normal 2 days ago before sepsis)
        c.executemany(
            "INSERT INTO previous_results (patient_id,analyte,value,unit,collected_at) VALUES (?,?,?,?,?)",
            [
                (1004, "Platelets", 185, "K/uL", "2026-04-05T08:00"),
                (1004, "PT", 12.5, "sec", "2026-04-05T08:00"),
                (1004, "Fibrinogen", 320, "mg/dL", "2026-04-05T08:00"),
                (1004, "D_Dimer", 0.3, "ug/mL", "2026-04-05T08:00"),
            ]
        )
        # Add a peripheral smear note
        c.execute("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)",
                  (cbc_id, 1004, "Peripheral_Smear_Morphology", 0, "",
                   "SCHISTOCYTES_PRESENT: 3+ fragmented RBCs per HPF. Consistent with microangiopathic hemolytic process.",
                   "2026-04-07T06:00"))
        # Less leading — mentions multiple panels but doesn't mention DIC or specific abnormalities
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-H001", 1004,
            "STAT: Multiple lab panels have resulted for patient ID 1004 (ICU). CBC with Diff, Coagulation Panel, and CMP all completed. Comprehensive review and diagnosis required.",
            "STAT", "AI-Agent"
        ))
        # Distractor cases
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-H002", 2001,
            "Annual wellness CMP for patient ID 2001. Routine.",
            "ROUTINE", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-H003", 2007,
            "Coag panel for patient ID 2007. Routine INR check.",
            "ROUTINE", "AI-Agent"
        ))
        # Seed DISTRACTOR lab results for other patients
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_task_hard_b(self):
        """HARD-B: Tumor Lysis Syndrome (TLS) after chemotherapy.
        Patient 1009 (34M) started chemo for lymphoma 48hrs ago. Classic TLS presents:
        - Uric Acid critically elevated (15.2 mg/dL, normal 3.4-7.0)
        - Potassium critically elevated (6.8 mEq/L)
        - Phosphate elevated (8.2 mg/dL, normal 2.5-4.5)
        - Calcium LOW (6.5 mg/dL) — inverse relationship with phosphate
        - Creatinine rising (2.8 mg/dL) — acute kidney injury from urate precipitation
        - LDH elevated (1200 U/L, normal 140-280) — cell lysis marker
        Agent must synthesize CMP + additional labs to diagnose E88.3 (TLS).
        """
        c = self.db.cursor()
        oid1 = "ORD-HB01"
        oid2 = "ORD-HB02"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid1, 1009, "CMP", "Dr. Yamamoto", "2026-04-07", "COMPLETED"))
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid2, 1009, "CBC_with_Diff", "Dr. Yamamoto", "2026-04-07", "COMPLETED"))
        cmp = [
            (oid1, 1009, "Sodium", 134, "mEq/L", "LOW", "2026-04-07T06:00"),
            (oid1, 1009, "Potassium", 6.8, "mEq/L", "", "2026-04-07T06:00"),  # CRITICAL
            (oid1, 1009, "Chloride", 100, "mEq/L", "NORMAL", "2026-04-07T06:00"),
            (oid1, 1009, "CO2", 18, "mEq/L", "LOW", "2026-04-07T06:00"),  # metabolic acidosis
            (oid1, 1009, "BUN", 35, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (oid1, 1009, "Creatinine", 2.8, "mg/dL", "HIGH", "2026-04-07T06:00"),  # AKI
            (oid1, 1009, "Glucose", 110, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (oid1, 1009, "Calcium", 6.5, "mg/dL", "", "2026-04-07T06:00"),  # critically low
            (oid1, 1009, "Phosphate", 8.2, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (oid1, 1009, "Uric_Acid", 15.2, "mg/dL", "HIGH", "2026-04-07T06:00"),
            (oid1, 1009, "ALT", 42, "U/L", "HIGH", "2026-04-07T06:00"),
            (oid1, 1009, "AST", 55, "U/L", "HIGH", "2026-04-07T06:00"),
            (oid1, 1009, "Albumin", 3.0, "g/dL", "LOW", "2026-04-07T06:00"),
            (oid1, 1009, "LDH", 1200, "U/L", "HIGH", "2026-04-07T06:00"),  # cell lysis marker
        ]
        cbc = [
            (oid2, 1009, "WBC", 52.0, "K/uL", "HIGH", "2026-04-07T06:00"),  # leukocytosis from lymphoma
            (oid2, 1009, "RBC", 3.5, "M/uL", "LOW", "2026-04-07T06:00"),
            (oid2, 1009, "Hemoglobin", 10.2, "g/dL", "LOW", "2026-04-07T06:00"),
            (oid2, 1009, "Hematocrit", 30.5, "%", "LOW", "2026-04-07T06:00"),
            (oid2, 1009, "Platelets", 110, "K/uL", "LOW", "2026-04-07T06:00"),
        ]
        for batch in [cmp, cbc]:
            c.executemany("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)", batch)
        c.executemany("INSERT INTO previous_results (patient_id,analyte,value,unit,collected_at) VALUES (?,?,?,?,?)", [
            (1009, "Potassium", 4.0, "mEq/L", "2026-04-05T10:00"),   # normal before chemo
            (1009, "Creatinine", 0.9, "mg/dL", "2026-04-05T10:00"),   # normal before chemo
            (1009, "Uric_Acid", 6.5, "mg/dL", "2026-04-05T10:00"),    # borderline before chemo
            (1009, "Calcium", 9.2, "mg/dL", "2026-04-05T10:00"),
            (1009, "Phosphate", 3.8, "mg/dL", "2026-04-05T10:00"),
        ])
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-HB01", 1009,
            "STAT: CMP and CBC results for patient ID 1009. Multiple abnormal values. Comprehensive review required.",
            "STAT", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-HB02", 2007, "Routine INR for patient ID 2007.", "ROUTINE", "AI-Agent"
        ))
        c.execute("INSERT INTO pending_cases VALUES (?,?,?,?,?)", (
            "CASE-HB03", 2001, "Annual CMP for patient ID 2001.", "ROUTINE", "AI-Agent"
        ))
        self._seed_distractor_labs(c)
        self.db.commit()

    def _seed_distractor_labs(self, c):
        """Normal results for distractor patients so the DB isn't empty."""
        # Patient 2001 - healthy male annual
        oid = "ORD-D001"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid, 2001, "CMP", "Dr. Adams", "2026-04-06", "COMPLETED"))
        for analyte, val, unit in [
            ("Sodium",139,"mEq/L"),("Potassium",4.2,"mEq/L"),("Glucose",88,"mg/dL"),
            ("Creatinine",0.9,"mg/dL"),("Calcium",9.5,"mg/dL"),("ALT",22,"U/L"),
            ("AST",19,"U/L"),("Bilirubin",0.7,"mg/dL"),
        ]:
            c.execute("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)",
                      (oid, 2001, analyte, val, unit, "NORMAL", "2026-04-06T09:00"))
        # Patient 2007 - on Warfarin (INR expected to be 2.0-3.0)
        oid2 = "ORD-D002"
        c.execute("INSERT INTO lab_orders VALUES (?,?,?,?,?,?)",
                  (oid2, 2007, "Coag_Panel", "Dr. Stone", "2026-04-06", "COMPLETED"))
        for analyte, val, unit in [
            ("PT",22.5,"sec"),("PTT",38.0,"sec"),("INR",2.5,"ratio"),
        ]:
            c.execute("INSERT INTO lab_results (order_id,patient_id,analyte,value,unit,flag,collected_at) VALUES (?,?,?,?,?,?,?)",
                      (oid2, 2007, analyte, val, unit, "THERAPEUTIC", "2026-04-06T10:00"))

    # ────────────────────────────────────────
    # API TOOLS
    # ────────────────────────────────────────
    def t_list_pending_cases(self) -> str:
        c = self.db.cursor()
        c.execute("SELECT case_id, patient_id, description, priority FROM pending_cases WHERE assigned_to='AI-Agent'")
        return json.dumps([dict(r) for r in c.fetchall()])

    def t_get_patient_demographics(self, patient_id: int) -> str:
        self.queried_demographics.add(patient_id)
        c = self.db.cursor()
        c.execute("SELECT id, name, age, sex, weight_kg, ethnicity, medical_history, active_diagnoses, flags FROM patients WHERE id=?", (patient_id,))
        row = c.fetchone()
        if not row: raise ValueError(f"Patient {patient_id} not found.")
        result = dict(row)
        # Parse flags from JSON string to actual list
        try:
            result["flags"] = json.loads(result.get("flags", "[]"))
        except (json.JSONDecodeError, TypeError):
            result["flags"] = []
        return json.dumps(result)

    def t_get_medications(self, patient_id: int) -> str:
        self.queried_medications.add(patient_id)
        c = self.db.cursor()
        c.execute("SELECT drug_name, dose, route, frequency FROM medications WHERE patient_id=?", (patient_id,))
        return json.dumps([dict(r) for r in c.fetchall()])

    def t_get_lab_orders(self, patient_id: int) -> str:
        c = self.db.cursor()
        c.execute("SELECT order_id, panel_name, ordered_by, order_date, status FROM lab_orders WHERE patient_id=?", (patient_id,))
        return json.dumps([dict(r) for r in c.fetchall()])

    def t_get_lab_results(self, order_id: str) -> str:
        self.queried_lab_results.add(order_id)
        c = self.db.cursor()
        c.execute("SELECT analyte, value, unit, flag, collected_at FROM lab_results WHERE order_id=?", (order_id,))
        return json.dumps([dict(r) for r in c.fetchall()])

    def t_get_previous_results(self, patient_id: int, analyte: str = None) -> str:
        self.queried_previous.add(patient_id)
        c = self.db.cursor()
        if analyte:
            c.execute("SELECT analyte, value, unit, collected_at FROM previous_results WHERE patient_id=? AND analyte=? ORDER BY collected_at DESC", (patient_id, analyte))
        else:
            c.execute("SELECT analyte, value, unit, collected_at FROM previous_results WHERE patient_id=? ORDER BY collected_at DESC", (patient_id,))
        return json.dumps([dict(r) for r in c.fetchall()])

    def t_query_reference_ranges(self, analyte: str, context: dict = None) -> str:
        """Return reference ranges, adjusted for sex, pregnancy, and therapeutic context."""
        context = context or {}
        key = analyte.replace(" ", "_")
        self.queried_references.add(key)
        if key not in REFERENCE_RANGES:
            raise ValueError(f"Unknown analyte: {analyte}. Available: {', '.join(sorted(REFERENCE_RANGES.keys()))}")
        ref = dict(REFERENCE_RANGES[key])
        ref["analyte"] = key
        sex = context.get("sex", "M")
        flags = context.get("flags", [])
        # Apply sex-specific ranges
        if f"min_{sex}" in ref:
            ref["min"] = ref[f"min_{sex}"]
            ref["max"] = ref[f"max_{sex}"]
        # Apply pregnancy-adjusted ranges
        if sex == "F" and "pregnant" in flags and f"min_F_pregnant" in ref:
            ref["min"] = ref["min_F_pregnant"]
            ref["max"] = ref["max_F_pregnant"]
        # Surface therapeutic context for anticoagulated patients
        if "on_warfarin" in flags and "therapeutic_min" in ref:
            ref["note"] = "Patient is on anticoagulant therapy. Use therapeutic range for interpretation."
            if "therapeutic_mechanical_valve_min" in ref:
                ref["note"] += " For mechanical heart valves, target is therapeutic_mechanical_valve range."
        return json.dumps(ref)

    def t_flag_critical_value(self, patient_id: int, analyte: str, value: float, alert_level: str) -> str:
        c = self.db.cursor()
        c.execute("INSERT INTO critical_alerts (patient_id, analyte, value, alert_level) VALUES (?,?,?,?)",
                  (patient_id, analyte, value, alert_level))
        self.db.commit()
        self.flagged_criticals.append({"patient_id": patient_id, "analyte": analyte, "value": value, "alert_level": alert_level})
        return f"CRITICAL ALERT FILED: {analyte}={value} for patient {patient_id}. Attending physician notified."

    def t_submit_diagnostic_report(self, patient_id: int, icd_code: str, severity: str, clinical_notes: str = "") -> str:
        c = self.db.cursor()
        c.execute("INSERT INTO diagnostic_reports (patient_id, icd_code, severity, clinical_notes, submitted_at) VALUES (?,?,?,?,?)",
                  (patient_id, icd_code, severity, clinical_notes, "2026-04-07T12:00"))
        self.db.commit()
        self.submitted_reports.append({"patient_id": patient_id, "icd_code": icd_code, "severity": severity, "notes": clinical_notes})
        return f"Diagnostic report submitted for patient {patient_id}: {icd_code} ({severity})"

    # ────────────────────────────────────────
    # DETERMINISTIC GRADERS (multi-variant)
    # ────────────────────────────────────────
    def _grade_easy(self) -> float:
        if self.task_variant == 0:
            return self._grade_easy_a()
        elif self.task_variant == 1:
            return self._grade_easy_b()
        else:
            return self._grade_easy_c()

    def _grade_easy_a(self) -> float:
        """Hyperkalemia — patient 1001. K+ 7.2 mEq/L (crit threshold 6.5)."""
        score = 0.01  # participation credit
        # Investigation (10% max) — small signal for following workflow
        if 1001 in self.queried_demographics: score += 0.03
        if 1001 in self.queried_medications: score += 0.03
        if "ORD-E001" in self.queried_lab_results: score += 0.02
        if "Potassium" in self.queried_references: score += 0.02
        # Critical flag (15%) — must flag correct value AND alert level
        for alert in self.flagged_criticals:
            if alert["patient_id"] == 1001 and alert["analyte"] == "Potassium":
                if abs(alert["value"] - 7.2) < 0.5 and alert["alert_level"] in ("CRITICAL", "CRITICAL_HIGH"):
                    score += 0.15
                else:
                    score += 0.05  # partial: right analyte, wrong value/level
                break
        # Diagnosis (60% ICD + 15% severity = 75%)
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1001:
                if rep["icd_code"] in ("E87.5", "E875"): score += 0.58  # max natural = 0.99
                elif rep["icd_code"].startswith("E87"): score += 0.30  # right electrolyte category
                elif rep["icd_code"].startswith("T46"): score += 0.15  # drug adverse effect angle
                else: score -= 0.30  # wrong diagnosis
                if rep["severity"] == "CRITICAL": score += 0.15
                elif rep["severity"] == "HIGH": score += 0.05
                break
        return max(min(score, 0.99), 0.01)

    def _grade_easy_b(self) -> float:
        """Acute MI — patient 1005. Troponin I 2.8 ng/mL (crit threshold 0.4)."""
        score = 0.01  # participation credit
        if 1005 in self.queried_demographics: score += 0.03
        if 1005 in self.queried_medications: score += 0.03
        if "ORD-EB01" in self.queried_lab_results: score += 0.02
        if "Troponin_I" in self.queried_references: score += 0.02
        for alert in self.flagged_criticals:
            if alert["patient_id"] == 1005 and alert["analyte"] == "Troponin_I":
                if alert["value"] > 0.4 and alert["alert_level"] in ("CRITICAL", "CRITICAL_HIGH"):
                    score += 0.15
                else:
                    score += 0.05
                break
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1005:
                if rep["icd_code"] in ("I21.9", "I21.3", "I21.0"): score += 0.58  # max natural = 0.99
                elif rep["icd_code"].startswith("I21"): score += 0.30  # right category
                else: score -= 0.30
                if rep["severity"] == "CRITICAL": score += 0.15
                elif rep["severity"] == "HIGH": score += 0.05
                break
        return max(min(score, 0.99), 0.01)

    def _grade_easy_c(self) -> float:
        """Severe anemia — patient 1006. Hb 6.2 g/dL (crit threshold 7.0)."""
        score = 0.01  # participation credit
        if 1006 in self.queried_demographics: score += 0.03
        if "ORD-EC01" in self.queried_lab_results: score += 0.02
        if "Hemoglobin" in self.queried_references: score += 0.03
        if 1006 in self.queried_previous: score += 0.02  # delta: 11.0→9.5→6.2
        for alert in self.flagged_criticals:
            if alert["patient_id"] == 1006 and alert["analyte"] == "Hemoglobin":
                if alert["value"] < 7.0 and alert["alert_level"] in ("CRITICAL", "CRITICAL_LOW"):
                    score += 0.15
                else:
                    score += 0.05
                break
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1006:
                if rep["icd_code"] in ("D64.9", "D50.9", "D50.0"): score += 0.58  # max natural = 0.99
                elif rep["icd_code"] in ("D62", "D50.1", "D63.1", "D50.8"): score += 0.45  # valid anemia subtypes
                elif rep["icd_code"][:3] >= "D50" and rep["icd_code"][:3] <= "D64": score += 0.30  # anemia category
                else: score -= 0.30
                if rep["severity"] == "CRITICAL": score += 0.15
                elif rep["severity"] == "HIGH": score += 0.05
                break
        return max(min(score, 0.99), 0.01)

    def _grade_medium(self) -> float:
        if self.task_variant == 0:
            return self._grade_medium_a()
        elif self.task_variant == 1:
            return self._grade_medium_b()
        else:
            return self._grade_medium_c()

    def _grade_medium_a(self) -> float:
        """Pregnancy Hb — patient 1002. Hb 11.5 (NORMAL for pregnant at 28wk).
        The trap: agent must recognize pregnancy-adjusted ranges. Flagging as critical = WRONG."""
        score = 0.01  # participation credit
        # Demographics is critical here — must see pregnancy flag
        if 1002 in self.queried_demographics: score += 0.10  # KEY: pregnancy flag changes interpretation
        if 1002 in self.queried_medications: score += 0.05
        if "Hemoglobin" in self.queried_references: score += 0.10  # KEY: must query with pregnancy context
        # PENALTY for false critical flag — Hb 11.5 is NORMAL in pregnancy
        for alert in self.flagged_criticals:
            if alert["patient_id"] == 1002 and alert["analyte"] == "Hemoglobin":
                score -= 0.20  # false alarm = bad clinical judgment
                break
        # Diagnosis — must be NORMAL, not anemia
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1002:
                if rep["icd_code"] == "NORMAL": score += 0.53  # max natural = 0.99
                elif rep["icd_code"].startswith("Z3"): score += 0.40  # pregnancy supervision/screening = reasonable "normal"
                elif rep["icd_code"].startswith("O24"): score += 0.15  # GDM focus — valid but not what we're testing
                elif rep["icd_code"].startswith("O99.0"): score -= 0.30  # anemia in pregnancy = wrong
                elif rep["icd_code"] in ("D64.9", "D50.9"): score -= 0.40  # false positive anemia
                if rep["severity"] == "INFO": score += 0.15
                elif rep["severity"] == "WARNING" and rep["icd_code"].startswith(("O24", "Z3")): score += 0.10
                elif rep["severity"] in ("CRITICAL", "HIGH"): score -= 0.10
                break
        return max(min(score, 0.99), 0.01)

    def _grade_medium_b(self) -> float:
        """Warfarin supratherapeutic INR — patient 1007. INR 4.8 (target 2.5-3.5 for mech valve)."""
        score = 0.01  # participation credit
        if 1007 in self.queried_demographics: score += 0.03
        if 1007 in self.queried_medications: score += 0.10  # MUST see Warfarin — key to entire diagnosis
        if "INR" in self.queried_references: score += 0.05  # must check therapeutic range
        if 1007 in self.queried_previous: score += 0.07  # trend: was 3.0, now 4.8 shows worsening
        # No critical flag reward — INR 4.8 is not "critical" per se, it's supratherapeutic
        # Diagnosis
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1007:
                if rep["icd_code"] in ("T45.515A", "T45.515", "T45.515D", "T45.515S"): score += 0.58  # max natural = 0.99
                elif rep["icd_code"].startswith("T45.51"): score += 0.45  # right drug, wrong encounter type
                elif rep["icd_code"] in ("R79.1", "D68.32", "D68.3", "D68.33"): score += 0.30  # partially correct coag codes
                elif rep["icd_code"].startswith("D68"): score += 0.20  # coag disorder category
                elif rep["icd_code"] == "D65": score -= 0.30  # incorrectly diagnosing DIC
                else: score -= 0.20
                if rep["severity"] == "WARNING": score += 0.15
                elif rep["severity"] == "HIGH": score += 0.10
                elif rep["severity"] == "CRITICAL": score += 0.03  # over-triaging
                break
        return max(min(score, 0.99), 0.01)

    def _grade_medium_c(self) -> float:
        """Drug-induced hyperkalemia — patient 1008. K+ 5.8 on ACE + K supplement + CKD."""
        score = 0.01  # participation credit
        if 1008 in self.queried_demographics: score += 0.03
        if 1008 in self.queried_medications: score += 0.10  # KEY: must see ACE + K supplement — without this, can't know it's drug-induced
        if "Potassium" in self.queried_references: score += 0.02
        if 1008 in self.queried_previous: score += 0.10  # trend: 5.2→5.8 shows progressive worsening
        # Diagnosis
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1008:
                if rep["icd_code"] in ("E87.5", "E875"): score += 0.58  # max natural = 0.99
                elif rep["icd_code"].startswith("T46.4"): score += 0.35  # adverse effect of ACE inhibitor — valid drug-cause angle
                elif rep["icd_code"].startswith("E87"): score += 0.25  # electrolyte disorder category
                else: score -= 0.20
                if rep["severity"] == "WARNING": score += 0.15
                elif rep["severity"] == "CRITICAL": score += 0.03  # over-triaging for 5.8
                elif rep["severity"] == "HIGH": score += 0.10
                break
        return max(min(score, 0.99), 0.01)

    def _grade_hard(self) -> float:
        if self.task_variant == 0:
            return self._grade_hard_a()
        else:
            return self._grade_hard_b()

    def _grade_hard_a(self) -> float:
        """DIC — patient 1004. Must synthesize CBC + Coag + CMP to diagnose.
        Requires: pulling all 3 panels, flagging critical coag values,
        querying reference ranges, and correct ICD + severity."""
        score = 0.01  # participation credit
        # Investigation (24% max) — pulling multiple panels is CRITICAL for DIC diagnosis
        panels_pulled = sum(1 for oid in ["ORD-H001", "ORD-H002", "ORD-H003"] if oid in self.queried_lab_results)
        score += panels_pulled * 0.04  # 0.12 max — can't diagnose DIC without seeing all panels
        if 1004 in self.queried_demographics: score += 0.02
        if 1004 in self.queried_medications: score += 0.02  # Heparin context matters for coag interpretation
        if 1004 in self.queried_previous: score += 0.03  # delta: Platelets 185→42 shows acute drop
        # Reference range queries (must verify at least one coag/heme range)
        coag_refs_queried = sum(1 for r in self.queried_references if r in ("Platelets", "Fibrinogen", "D_Dimer", "PT", "PTT", "INR"))
        if coag_refs_queried >= 1: score += 0.05
        # Critical value flagging (20% max) — DIC has multiple critical values that MUST be flagged
        dic_criticals = {"Platelets", "Fibrinogen", "D_Dimer"}
        flagged_dic = set()
        for alert in self.flagged_criticals:
            if alert["patient_id"] == 1004 and alert["analyte"] in dic_criticals:
                flagged_dic.add(alert["analyte"])
        if len(flagged_dic) >= 3: score += 0.20  # all 3 flagged — excellent clinical diligence
        elif len(flagged_dic) >= 2: score += 0.14  # 2 of 3 — good
        elif len(flagged_dic) >= 1: score += 0.07  # 1 of 3 — minimal
        else: score -= 0.05  # no critical flags in a DIC case = clinical failure
        # Diagnosis (42% ICD + 13% severity = 55%)
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1004:
                if rep["icd_code"] == "D65": score += 0.41  # DIC — max natural = 0.99
                elif rep["icd_code"].startswith("D68"): score += 0.20  # coagulopathy category — close but wrong specific code
                elif rep["icd_code"] == "A41.9": score += 0.08  # only identified sepsis, missed DIC
                elif rep["icd_code"] in ("D69.6", "D69.59"): score += 0.12  # thrombocytopenia only
                else: score -= 0.20
                if rep["severity"] == "CRITICAL": score += 0.13
                elif rep["severity"] == "HIGH": score += 0.05
                break
        return max(min(score, 0.99), 0.01)

    def _grade_hard_b(self) -> float:
        """Tumor Lysis Syndrome — patient 1009. Must identify TLS tetrad."""
        score = 0.01  # participation credit
        # Investigation (15%)
        panels_pulled = sum(1 for oid in ["ORD-HB01", "ORD-HB02"] if oid in self.queried_lab_results)
        score += panels_pulled * 0.04  # 0.08 max
        if 1009 in self.queried_demographics: score += 0.02
        if 1009 in self.queried_medications: score += 0.02
        if 1009 in self.queried_previous: score += 0.03  # delta checks critical for TLS
        # Diagnosis (70% ICD + 15% severity = 85%)
        for rep in self.submitted_reports:
            if rep["patient_id"] == 1009:
                if rep["icd_code"] in ("E88.3", "E883"): score += 0.68  # TLS — max natural = 0.99
                elif rep["icd_code"].startswith("E88"): score += 0.35  # metabolic disorder category
                elif rep["icd_code"] == "E79.0": score += 0.15  # hyperuricemia only = partial
                elif rep["icd_code"].startswith("E87"): score += 0.10  # electrolyte only = too shallow
                elif rep["icd_code"].startswith("N17"): score += 0.10  # AKI only = too shallow
                else: score -= 0.20
                if rep["severity"] == "CRITICAL": score += 0.15
                elif rep["severity"] == "HIGH": score += 0.05
                break
        return max(min(score, 0.99), 0.01)

    # ────────────────────────────────────────
    # CORE API
    # ────────────────────────────────────────
    def reset(self, task_level: str = None, **kwargs) -> PathologyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Priority: explicit parameter > env var > default "easy"
        self.task_level = (task_level or os.environ.get("TASK_LEVEL", "easy")).lower()
        self.submitted_reports = []
        self.flagged_criticals = []
        self.queried_demographics = set()
        self.queried_medications = set()
        self.queried_lab_results = set()
        self.queried_previous = set()
        self.queried_references = set()
        self._init_db()

        # Random variant selection per episode
        rng = random.Random(self._state.episode_id)

        if self.task_level == "easy":
            self.task_variant = rng.randint(0, 2)  # 3 variants
            [self._seed_task_easy, self._seed_task_easy_b, self._seed_task_easy_c][self.task_variant]()
        elif self.task_level == "medium":
            self.task_variant = rng.randint(0, 2)  # 3 variants
            [self._seed_task_medium, self._seed_task_medium_b, self._seed_task_medium_c][self.task_variant]()
        else:
            self.task_level = "hard"
            self.task_variant = rng.randint(0, 1)  # 2 variants
            [self._seed_task_hard, self._seed_task_hard_b][self.task_variant]()

        variant_labels = {
            "easy": ["Hyperkalemia", "Acute MI", "Severe Anemia"],
            "medium": ["Pregnancy Hb", "Warfarin INR", "Drug-Induced K+"],
            "hard": ["DIC", "Tumor Lysis Syndrome"],
        }
        variant_name = variant_labels.get(self.task_level, ["Unknown"])[self.task_variant]

        instruction = (
            "PATHOLOGY LIMS v3.0 — AI Diagnostic Agent Interface\n"
            "═══════════════════════════════════════════════════\n"
            "You are an AI pathology agent reviewing lab results in a hospital LIMS.\n"
            "Your job: review pending cases, query all relevant data, interpret results\n"
            "using proper clinical context (demographics, medications, reference ranges),\n"
            "flag any critical values, and submit a final diagnostic report.\n\n"
            "Available commands:\n"
            "  list_pending_cases          args: {}\n"
            "  get_patient_demographics    args: {patient_id: int}\n"
            "  get_medications             args: {patient_id: int}\n"
            "  get_lab_orders              args: {patient_id: int}\n"
            "  get_lab_results             args: {order_id: string}\n"
            "  get_previous_results        args: {patient_id: int, analyte?: string}\n"
            "  query_reference_ranges      args: {analyte: string, context?: {sex, flags}}\n"
            "  flag_critical_value         args: {patient_id: int, analyte: string, value: float, alert_level: string}\n"
            "  submit_diagnostic_report    args: {patient_id: int, icd_code: string, severity: string, clinical_notes?: string}\n"
        )
        return PathologyObservation(output=instruction, error="", done=False, reward=0.0)

    def step(self, action: PathologyAction) -> PathologyObservation:
        self._state.step_count += 1
        cmd = action.command
        args = action.arguments
        output = ""
        error = ""
        submitted_diagnosis = False

        try:
            if cmd == "list_pending_cases":
                output = self.t_list_pending_cases()
            elif cmd == "get_patient_demographics":
                output = self.t_get_patient_demographics(int(args.get("patient_id", 0)))
            elif cmd == "get_medications":
                output = self.t_get_medications(int(args.get("patient_id", 0)))
            elif cmd == "get_lab_orders":
                output = self.t_get_lab_orders(int(args.get("patient_id", 0)))
            elif cmd == "get_lab_results":
                output = self.t_get_lab_results(args.get("order_id", ""))
            elif cmd == "get_previous_results":
                output = self.t_get_previous_results(int(args.get("patient_id", 0)), args.get("analyte"))
            elif cmd == "query_reference_ranges":
                output = self.t_query_reference_ranges(args.get("analyte", ""), args.get("context", {}))
            elif cmd == "flag_critical_value":
                output = self.t_flag_critical_value(
                    int(args.get("patient_id", 0)), args.get("analyte", ""),
                    float(args.get("value", 0)), args.get("alert_level", "CRITICAL")
                )
            elif cmd == "submit_diagnostic_report":
                output = self.t_submit_diagnostic_report(
                    int(args.get("patient_id", 0)), args.get("icd_code", ""),
                    args.get("severity", ""), args.get("clinical_notes", "")
                )
                submitted_diagnosis = True
            else:
                error = f"Unknown command: '{cmd}'. Type list_pending_cases to begin."
        except Exception as e:
            error = f"Error: {str(e)}"

        reward = 0.0
        if self.task_level == "easy": reward = self._grade_easy()
        elif self.task_level == "medium": reward = self._grade_medium()
        elif self.task_level == "hard": reward = self._grade_hard()

        # Done when: diagnosis submitted, or max steps reached
        done = submitted_diagnosis or self._state.step_count >= self.max_steps

        # Append score breakdown to output when episode ends
        if done:
            breakdown = self._generate_score_breakdown(reward)
            output = output + "\n\n" + breakdown if output else breakdown

        return PathologyObservation(
            output=output, error=error, done=done, reward=reward,
            metadata={"step": self._state.step_count, "task": self.task_level, "variant": self.task_variant}
        )

    def _generate_score_breakdown(self, total_reward: float) -> str:
        """Generate a human-readable score breakdown explaining each component."""
        lines = ["=" * 60, "SCORE BREAKDOWN", "=" * 60]
        level = self.task_level
        variant = self.task_variant

        # Variant identification
        variant_names = {
            "easy": {0: "Hyperkalemia (K+ 7.2)", 1: "Acute MI (Troponin 2.8)", 2: "Severe Anemia (Hb 6.2)"},
            "medium": {0: "Pregnancy Hb (11.5 — NORMAL)", 1: "Warfarin INR (4.8 supratherapeutic)", 2: "Drug-Induced K+ (5.8)"},
            "hard": {0: "DIC (multi-panel synthesis)", 1: "Tumor Lysis Syndrome (TLS tetrad)"},
        }
        vname = variant_names.get(level, {}).get(variant, "Unknown")
        lines.append(f"Task: {level.upper()} | Scenario: {vname}")
        lines.append(f"Steps used: {self._state.step_count}/{self.max_steps}")
        lines.append("-" * 60)

        # Investigation
        lines.append("INVESTIGATION:")
        pid_map = {
            ("easy", 0): 1001, ("easy", 1): 1005, ("easy", 2): 1006,
            ("medium", 0): 1002, ("medium", 1): 1007, ("medium", 2): 1008,
            ("hard", 0): 1004, ("hard", 1): 1009,
        }
        pid = pid_map.get((level, variant), 0)
        lines.append(f"  Demographics queried:  {'✓' if pid in self.queried_demographics else '✗'}")
        lines.append(f"  Medications queried:   {'✓' if pid in self.queried_medications else '✗'}")
        lines.append(f"  Previous results:      {'✓' if pid in self.queried_previous else '✗'}")
        lines.append(f"  Lab results queried:   {list(self.queried_lab_results) or '(none)'}")
        lines.append(f"  Reference ranges:      {list(self.queried_references) or '(none)'}")

        # Critical flags
        lines.append("\nCRITICAL FLAGS:")
        if self.flagged_criticals:
            for f in self.flagged_criticals:
                lines.append(f"  → {f['analyte']}={f['value']} [{f['alert_level']}] for patient {f['patient_id']}")
        else:
            lines.append("  (none filed)")

        # Diagnosis
        lines.append("\nDIAGNOSIS SUBMITTED:")
        expected_icd = {
            ("easy", 0): "E87.5 (Hyperkalemia)", ("easy", 1): "I21.9 (Acute MI)", ("easy", 2): "D64.9 (Anemia)",
            ("medium", 0): "NORMAL (pregnancy-adjusted Hb is normal)", ("medium", 1): "T45.515A (Adverse effect of anticoagulant)",
            ("medium", 2): "E87.5 (Hyperkalemia — drug-induced)",
            ("hard", 0): "D65 (Disseminated Intravascular Coagulation)", ("hard", 1): "E88.3 (Tumor Lysis Syndrome)",
        }
        expected_sev = {
            ("easy", 0): "CRITICAL", ("easy", 1): "CRITICAL", ("easy", 2): "CRITICAL",
            ("medium", 0): "INFO", ("medium", 1): "WARNING", ("medium", 2): "WARNING",
            ("hard", 0): "CRITICAL", ("hard", 1): "CRITICAL",
        }
        exp_icd = expected_icd.get((level, variant), "?")
        exp_sev = expected_sev.get((level, variant), "?")

        if self.submitted_reports:
            rep = self.submitted_reports[0]
            submitted_icd = rep["icd_code"]
            submitted_sev = rep["severity"]
            icd_match = "✓ CORRECT" if submitted_icd in exp_icd.split(" ")[0] else "✗ INCORRECT"
            sev_match = "✓" if submitted_sev == exp_sev else "✗"
            lines.append(f"  ICD Code:   {submitted_icd} {icd_match}")
            lines.append(f"  Expected:   {exp_icd}")
            lines.append(f"  Severity:   {submitted_sev} {sev_match} (expected: {exp_sev})")
            if rep.get("notes"):
                notes_preview = rep["notes"][:150] + ("..." if len(rep["notes"]) > 150 else "")
                lines.append(f"  Notes:      {notes_preview}")
        else:
            lines.append("  ✗ NO DIAGNOSIS SUBMITTED — agent did not call submit_diagnostic_report")
            lines.append(f"  Expected:   {exp_icd}")

        lines.append("-" * 60)
        lines.append(f"FINAL SCORE: {total_reward:.2f} / 1.00")
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def state(self) -> State:
        return self._state

