"""
prepare_clinical_annotation.py
===============================
Generates a blinded annotation CSV for clinical validation of constraint C2
(Power grasp + Minimal grip aperture) from the ontology-driven rule engine.

WHAT THIS SCRIPT DOES
---------------------
1. Loads the grasp dataset and runs the rule engine on the full data to
   collect C2-violating and non-violating instances.
2. Samples N_SAMPLE instances from each group (C2-violated, C2-not-violated),
   matched approximately on object shape and grasp family.
3. Converts raw sensor values to human-readable flexion level descriptions
   (Low / Medium / High per finger) using training-fold quantile thresholds.
4. Exports two files:
      outputs/annotation_blind.csv   -- sent to the clinician (NO violation flag)
      outputs/annotation_key.csv     -- kept by researcher (violation flag + metadata)

ANNOTATION TASK FOR CLINICIAN
------------------------------
The clinician sees, per row:
  - Object name and shape (Cylinder / Sphere)
  - Grip aperture category (Minimal / Intermediate / Maximal)
  - Inferred grasp type and family
  - Human-readable finger flexion levels (Low / Medium / High per finger)

The clinician does NOT see whether C2 was triggered.

The judgment question is:
  "Given the finger configuration and object context, is this grasp
   configuration biomechanically PLAUSIBLE or SUSPICIOUS?"

  Plausible  = consistent with normal human grasp biomechanics for this object
  Suspicious = contains an element that is unusual or internally inconsistent
  Uncertain  = insufficient information or clearly borderline

The clinician fills in the 'Clinical_Judgment' column with:
  P  (Plausible)
  S  (Suspicious)
  U  (Uncertain)

USAGE
-----
  python prepare_clinical_annotation.py

OUTPUT
------
  outputs/annotation_blind.csv   -- send this to the clinician
  outputs/annotation_key.csv     -- keep this; used by analyze_clinical_annotation.py
  outputs/annotation_protocol.txt -- plain-text instructions to include with the CSV
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# -----------------------------------------------------------------------
# Import shared utilities from the main experiment file.
# Both files must be in the same directory.
# -----------------------------------------------------------------------
from run_grasp_experiments import (
    load_dataset,
    add_distal_aggregates,
    compute_quantile_thresholds,
    flex_level,
    rule_engine_predict,
    compute_violations,
    SENSOR_COLS,
    CONTEXT_COLS,
    TARGET_COL,
    DATA_PATH,
    N_SPLITS,
    RANDOM_STATE,
)

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
N_SAMPLE      = 100    # instances to sample from each group (C2-violated / control)
RANDOM_SEED   = 2024   # reproducibility of the sample draw
OUT_DIR       = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Human-readable finger labels (used in the annotation CSV)
# -----------------------------------------------------------------------
FINGER_DISPLAY = {
    "F_Index":  "Index finger",
    "F_Middle": "Middle finger",
    "F_Ring":   "Ring finger",
    "F_Little": "Little finger",
    "F_Thumb":  "Thumb",
}

FLEXION_DISPLAY = {
    "LowFlexion":    "Low (extended)",
    "MediumFlexion": "Medium (partially flexed)",
    "HighFlexion":   "High (fully flexed)",
}

GRASP_FAMILY = {
    "Large_Diameter":  "Power",
    "Small_Diameter":  "Power",
    "Medium_Wrap":     "Power",
    "Power_Sphere":    "Power",
    "Sphere_4_Finger": "Power",
    "Tripod":          "Precision",
    "Quadpod":         "Precision",
    "Sphere_3_Finger": "Precision",
    "Thumb_Adducted":  "Precision",
}

# -----------------------------------------------------------------------
# STEP 1: Load data and collect per-instance rule engine predictions
#         using the same CV protocol as the main experiment.
# -----------------------------------------------------------------------
def collect_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the rule engine across all CV folds and collect, for each test
    instance: predicted label, C2 flag, and discretized flexion levels.
    Returns a DataFrame with one row per test instance (all folds combined).
    """
    needed_cols = SENSOR_COLS + CONTEXT_COLS + [TARGET_COL]
    df = df[needed_cols].dropna().reset_index(drop=True)

    X = df[SENSOR_COLS + CONTEXT_COLS].copy()
    y = df[TARGET_COL].astype(str).copy()

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    records = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr = X.iloc[tr].copy()
        Xte = X.iloc[te].copy()
        yte = y.iloc[te].copy()

        thr = compute_quantile_thresholds(Xtr)

        df_test = Xte.copy()
        df_test_with_y = df_test.join(yte.rename(TARGET_COL))

        y_pred = rule_engine_predict(df_test_with_y, thr)

        # Compute C2 violations on this fold
        viol_rate, viol_counts = compute_violations(df_test, y_pred)

        # Determine per-instance C2 flag
        power_set = {
            "Large_Diameter", "Small_Diameter", "Medium_Wrap",
            "Power_Sphere", "Sphere_4_Finger"
        }
        for i, (idx, row) in enumerate(df_test.iterrows()):
            pred   = y_pred[i]
            grip   = str(row.get("Grip_Aperature", "")).strip()
            shape  = str(row.get("Shape", "")).strip()
            obj    = str(row.get("Object",
                         row.get("Material", "Unknown"))).strip()

            c2_flag = (pred in power_set) and (grip == "Minimal")

            # Compute flexion levels for human display
            lv = {}
            for c in SENSOR_COLS:
                low, high = thr[c]
                lv[c] = flex_level(float(row[c]), low, high)

            records.append({
                "fold":            fold,
                "original_index":  idx,
                "true_label":      yte.iloc[i],
                "predicted_label": pred,
                "grasp_family":    GRASP_FAMILY.get(pred, "Unknown"),
                "shape":           shape,
                "grip_aperture":   grip,
                "object":          obj,
                "c2_violated":     c2_flag,
                "flex_index":      lv["F_Index"],
                "flex_middle":     lv["F_Middle"],
                "flex_ring":       lv["F_Ring"],
                "flex_little":     lv["F_Little"],
                "flex_thumb":      lv["F_Thumb"],
            })

        print(f"  Fold {fold}/{N_SPLITS}: {sum(r['c2_violated'] for r in records if r['fold'] == fold)} C2 violations")

    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# STEP 2: Build the human-readable finger configuration description
# -----------------------------------------------------------------------
def make_finger_description(row: pd.Series) -> str:
    parts = []
    for col, label in FINGER_DISPLAY.items():
        key = f"flex_{col.replace('F_', '').lower()}"
        level = row.get(key, "Unknown")
        parts.append(f"{label}: {FLEXION_DISPLAY.get(level, level)}")
    return " | ".join(parts)


# -----------------------------------------------------------------------
# STEP 3: Sample C2-violating and control instances
# -----------------------------------------------------------------------
def sample_instances(
    all_preds: pd.DataFrame,
    n: int,
    rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (c2_sample, control_sample), each of size n.

    Control instances are:
      - Power-family grasps (same family as C2-violated)
      - Grip aperture NOT Minimal (Intermediate or Maximal)
      - Matched on shape distribution as closely as possible
    """
    c2 = all_preds[all_preds["c2_violated"] == True].copy()
    ctrl = all_preds[
        (all_preds["c2_violated"] == False) &
        (all_preds["grasp_family"] == "Power") &
        (all_preds["grip_aperture"] != "Minimal")
    ].copy()

    if len(c2) < n:
        raise ValueError(
            f"Only {len(c2)} C2-violating instances available; "
            f"requested {n}. Reduce N_SAMPLE."
        )
    if len(ctrl) < n:
        raise ValueError(
            f"Only {len(ctrl)} control instances available; "
            f"requested {n}. Reduce N_SAMPLE."
        )

    # Stratified sample by shape to preserve shape distribution
    c2_sample   = _stratified_sample(c2,   n, "shape", rng)
    ctrl_sample = _stratified_sample(ctrl, n, "shape", rng)

    return c2_sample, ctrl_sample


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    strat_col: str,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Sample n rows, approximately preserving the distribution of strat_col.
    """
    groups = df[strat_col].value_counts(normalize=True)
    parts  = []
    remaining = n
    for i, (val, prop) in enumerate(groups.items()):
        # Last group gets all remaining to avoid rounding errors
        k = remaining if i == len(groups) - 1 else max(1, round(prop * n))
        k = min(k, len(df[df[strat_col] == val]))
        subset = df[df[strat_col] == val].sample(n=k, replace=False,
                                                  random_state=int(rng.integers(1e6)))
        parts.append(subset)
        remaining -= k
        if remaining <= 0:
            break
    return pd.concat(parts).reset_index(drop=True)


# -----------------------------------------------------------------------
# STEP 4: Build annotation CSV (blinded) and key CSV
# -----------------------------------------------------------------------
def build_annotation_files(
    c2_sample: pd.DataFrame,
    ctrl_sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (blind_df, key_df).
    blind_df is sent to the clinician — no C2 flag, no model internals.
    key_df is kept by the researcher — includes C2 flag for post-analysis.
    """
    combined = pd.concat(
        [c2_sample.assign(group="C2_violated"),
         ctrl_sample.assign(group="control")],
        ignore_index=True
    )

    # Shuffle rows so violation pattern is not revealed by row order
    rng_shuffle = np.random.default_rng(RANDOM_SEED + 1)
    combined = combined.sample(
        frac=1, random_state=int(rng_shuffle.integers(1e6))
    ).reset_index(drop=True)

    # Assign sequential annotation IDs
    combined.insert(0, "Annotation_ID", range(1, len(combined) + 1))

    # Human-readable finger description
    combined["Finger_Configuration"] = combined.apply(
        make_finger_description, axis=1
    )

    # ---- Blinded annotation CSV (sent to clinician) ----
    blind_cols = [
        "Annotation_ID",
        "object",
        "shape",
        "grip_aperture",
        "grasp_family",
        "predicted_label",
        "Finger_Configuration",
        "Clinical_Judgment",    # empty column for clinician to fill
    ]
    blind_df = combined[blind_cols[:-1]].copy()
    # Rename columns for clinician readability
    blind_df = blind_df.rename(columns={
        "object":           "Object",
        "shape":            "Object_Shape",
        "grip_aperture":    "Grip_Aperture",
        "grasp_family":     "Inferred_Grasp_Family",
        "predicted_label":  "Inferred_Grasp_Type",
    })
    blind_df["Clinical_Judgment"] = ""   # clinician fills this in

    # ---- Key CSV (researcher keeps) ----
    key_df = combined[[
        "Annotation_ID",
        "c2_violated",
        "group",
        "fold",
        "original_index",
        "true_label",
        "predicted_label",
        "shape",
        "grip_aperture",
        "object",
    ]].copy()

    return blind_df, key_df


# -----------------------------------------------------------------------
# STEP 5: Write the annotation protocol text file
# -----------------------------------------------------------------------
PROTOCOL_TEXT = """
CLINICAL ANNOTATION PROTOCOL
Ontology-Driven Grasp Interpretation Study
==========================================

BACKGROUND
----------
This annotation task supports the clinical validation of an automated
semantic consistency checker for human grasp interpretation. The system
analyses finger joint kinematics recorded during grasping and checks whether
the inferred grasp type is biomechanically consistent with the object context.

YOUR TASK
---------
You will receive a spreadsheet (annotation_blind.csv) containing {N} grasp
instances. Each row describes one grasp configuration captured by a wearable
data glove. For each row, please assess whether the described configuration
is biomechanically PLAUSIBLE or SUSPICIOUS.

WHAT EACH COLUMN MEANS
-----------------------
Annotation_ID        : Sequential row identifier
Object               : The object being grasped (e.g., Steel_Mug, Tennis_Ball)
Object_Shape         : Cylinder or Sphere
Grip_Aperture        : Minimal / Intermediate / Maximal
                       (Minimal = small contact area; Maximal = full palmar wrap)
Inferred_Grasp_Family: Power or Precision (the system's inferred grasp category)
Inferred_Grasp_Type  : Specific grasp name (e.g., Large_Diameter, Medium_Wrap,
                       Tripod, Thumb_Adducted)
Finger_Configuration : Flexion level for each finger:
                       Low (extended) / Medium (partially flexed) / High (fully flexed)

THE JUDGMENT QUESTION
---------------------
"Given the finger configuration and object context described, is this
grasp configuration biomechanically PLAUSIBLE or SUSPICIOUS?"

  P = Plausible  : The combination of finger flexion levels, grip aperture,
                   and object is consistent with natural human grasping.

  S = Suspicious : The configuration contains an element that is biomechanically
                   unusual or internally inconsistent. For example:
                   - A full power wrap (all fingers high flexion) on a very
                     small-aperture object
                   - A precision grip configuration on a large cylindrical object
                     that normally requires full palmar contact
                   - A finger configuration that does not match the expected
                     posture for the stated grasp type

  U = Uncertain  : You do not have sufficient information to judge, or the
                   case is genuinely borderline.

HOW TO FILL IN THE CSV
-----------------------
In the 'Clinical_Judgment' column, type one of:  P  S  U
(capital letter only, no punctuation)

IMPORTANT NOTES
---------------
- Please annotate each row independently; do not look for patterns across rows.
- There is no correct answer we are testing you on. We are using your clinical
  judgment as an independent reference standard.
- If you are unsure whether to mark S or U, err toward U — uncertainty is
  informative and will be reported separately.
- Please complete all {N} rows. The task should take approximately 60--90 minutes.

RETURN
------
Please return the completed annotation_blind.csv by email.
If you have questions about any specific case, please note the Annotation_ID
and we will discuss.

Thank you for your contribution to this work.
""".strip()


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    df = add_distal_aggregates(df)

    print("Running rule engine across all CV folds to collect predictions...")
    all_preds = collect_predictions(df)

    n_c2_total = all_preds["c2_violated"].sum()
    n_total    = len(all_preds)
    print(f"\nTotal test instances (all folds): {n_total}")
    print(f"C2-violating instances:           {n_c2_total} "
          f"({100.0 * n_c2_total / n_total:.2f}%)")
    print(f"Non-violating instances:          {n_total - n_c2_total}")

    print(f"\nSampling {N_SAMPLE} C2-violating + {N_SAMPLE} control instances...")
    rng = np.random.default_rng(RANDOM_SEED)
    c2_sample, ctrl_sample = sample_instances(all_preds, N_SAMPLE, rng)

    print("Building annotation and key files...")
    blind_df, key_df = build_annotation_files(c2_sample, ctrl_sample)

    # Save files
    blind_path    = os.path.join(OUT_DIR, "annotation_blind.csv")
    key_path      = os.path.join(OUT_DIR, "annotation_key.csv")
    protocol_path = os.path.join(OUT_DIR, "annotation_protocol.txt")

    blind_df.to_csv(blind_path, index=False)
    key_df.to_csv(key_path, index=False)

    with open(protocol_path, "w", encoding="utf-8") as f:
        f.write(PROTOCOL_TEXT.format(N=len(blind_df)))

    print(f"\nOutputs saved to: {OUT_DIR}/")
    print(f"  {blind_path}    <- SEND THIS TO THE CLINICIAN")
    print(f"  {key_path}      <- KEEP THIS (do not share)")
    print(f"  {protocol_path} <- INCLUDE WITH THE CSV")
    print(f"\nTotal instances for annotation: {len(blind_df)}")
    print(f"  C2-violating: {N_SAMPLE}")
    print(f"  Controls:     {N_SAMPLE}")
    print("\nColumn distribution in annotation file:")
    print(blind_df[["Inferred_Grasp_Family", "Object_Shape",
                     "Grip_Aperture"]].value_counts().to_string())
