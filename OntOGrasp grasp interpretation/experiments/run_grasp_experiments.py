import os
import time
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# Optional:
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# ----------------------------
# 0) CONFIG
# ----------------------------
DATA_PATH = "data_with_area.csv"   # or Data-kinematics.csv / data_onto.xlsx (adapt loader below)
TARGET_COL = "Grasp_Type"
N_SPLITS = 5
RANDOM_STATE = 42

# ----------------------------
# Ontology evaluation sets
# ----------------------------
LEAF_LABELS = {
    "Large_Diameter", "Thumb_Adducted", "Quadpod", "Tripod",
    "Medium_Wrap", "Small_Diameter", "Power_Sphere",
    "Sphere_3_Finger", "Sphere_4_Finger"
}

PARENT_LABELS = {"PowerGrasp", "PrecisionGrasp", "IntermediateGrasp", "Grasp"}

# ----------------------------
# Coverage: leaf vs parent vs other
# ----------------------------
def compute_coverage(y_pred):
    y_pred = list(y_pred)
    leaf = sum(1 for p in y_pred if p in LEAF_LABELS)
    parent = sum(1 for p in y_pred if p in PARENT_LABELS)
    other = len(y_pred) - leaf - parent
    n = len(y_pred)
    return {
        "leaf_coverage_pct": 100.0 * leaf / n,
        "parent_only_pct": 100.0 * parent / n,
        "other_pct": 100.0 * other / n
    }

# ----------------------------
# Constraint violations
# (edit constraints later if needed)
# ----------------------------
def compute_violations(df_test, y_pred):
    from collections import Counter
    counts = Counter()
    violated_samples = 0

    for i, pred in enumerate(y_pred):
        shape = str(df_test.iloc[i]["Shape"]).strip()
        grip  = str(df_test.iloc[i]["Grip_Aperature"]).strip()

        v = []

        precision_set = {"Tripod", "Quadpod", "Sphere_3_Finger", "Thumb_Adducted"}
        power_set     = {"Large_Diameter", "Small_Diameter", "Medium_Wrap", "Power_Sphere", "Sphere_4_Finger"}

        # ----------------------
        # C1 (Refined)
        # Precision + Maximal + Cylinder
        # ----------------------
        if pred in precision_set and grip == "Maximal" and shape == "Cylinder":
            v.append("C1")

        # ----------------------
        # C2 (keep existing)
        # Power + Minimal
        # ----------------------
        if pred in power_set and grip == "Minimal":
            v.append("C2")

        # ----------------------
        # C3 (Refined)
        # Cylinder but spherical-local grasp predicted
        # ----------------------
        if shape == "Cylinder" and pred in {"Sphere_3_Finger", "Thumb_Adducted"}:
            v.append("C3")

        if v:
            violated_samples += 1
            counts.update(v)

    rate = 100.0 * violated_samples / len(y_pred)
    return rate, counts

# ----------------------------
# Noise injection + prediction stability
# ----------------------------
def inject_noise(X, sensor_cols, noise_pct, rng: np.random.Generator):
    """
    Add Gaussian noise with std = noise_pct * (max-min) per feature, computed on X (test fold).
    """
    Xn = X.copy()
    for c in sensor_cols:
        col = Xn[c].astype(float).values
        r = float(col.max() - col.min())
        sigma = noise_pct * r
        Xn[c] = col + rng.normal(0.0, sigma, size=len(col))
    return Xn

def prediction_stability(y_base, y_noisy):
    y_base = np.asarray(y_base, dtype=object)
    y_noisy = np.asarray(y_noisy, dtype=object)
    return 100.0 * (y_base == y_noisy).mean()

# Context features (edit if your file differs)
CONTEXT_COLS = ["Grip_Aperature", "Shape", "Material", "Tactility", "Object"]

# Joint columns (edit if your file differs)
COL_INDEX_PIP = "Index_PIP(f/e)"
COL_INDEX_DIP = "Index_DIP(f/e)"
COL_MIDDLE_PIP = "Middle_PIP(f/e)"
COL_MIDDLE_DIP = "Middle_DIP(f/e)"
COL_RING_PIP = "Ring_PIP(f/e)"
COL_RING_DIP = "Ring_DIP(f/e)"
COL_LITTLE_PIP = " Little_PIP(f/e)"
COL_LITTLE_DIP = " Little_DIP(f/e)"
COL_THUMB_IP = "Thumb_IP(f/e)"

# Output
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# 1) LOAD + AGGREGATE DISTAL FLEXION
# ----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    # CSV
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    # XLSX
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")

def add_distal_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Representative distal flexion:
    - Index/Middle/Ring/Little: mean(PIP, DIP)
    - Thumb: IP
    """
    df = df.copy()
    df["F_Index"]  = (df[COL_INDEX_PIP]  + df[COL_INDEX_DIP])  / 2.0
    df["F_Middle"] = (df[COL_MIDDLE_PIP] + df[COL_MIDDLE_DIP]) / 2.0
    df["F_Ring"]   = (df[COL_RING_PIP]   + df[COL_RING_DIP])   / 2.0
    df["F_Little"] = (df[COL_LITTLE_PIP] + df[COL_LITTLE_DIP]) / 2.0
    df["F_Thumb"]  = df[COL_THUMB_IP]
    return df

SENSOR_COLS = ["F_Thumb", "F_Index", "F_Middle", "F_Ring", "F_Little"]

# ----------------------------
# 2) SYMBOLIC DISCRETIZATION (TRAIN-FOLD QUANTILES)
# ----------------------------
def compute_quantile_thresholds(train_df: pd.DataFrame, sensor_cols=SENSOR_COLS):
    """
    Thresholds computed ONLY on training fold:
      low = 33rd percentile, high = 66th percentile
    """
    thr = {}
    for c in sensor_cols:
        v = train_df[c].astype(float).values
        thr[c] = (np.quantile(v, 0.33), np.quantile(v, 0.66))
    # Optional split threshold used in one rule to separate wrap subtypes
    mean_wrap = train_df[["F_Index","F_Middle","F_Ring","F_Little"]].mean(axis=1).values
    thr["wrap_split"] = np.quantile(mean_wrap, 0.60)
    return thr

def flex_level(x: float, low: float, high: float) -> str:
    if x < low:
        return "LowFlexion"
    if x < high:
        return "MediumFlexion"
    return "HighFlexion"

# ----------------------------
# 3) RULE ENGINE (MOST SPECIFIC MATCHING PARENT FALLBACK)
# ----------------------------
def infer_parent(grip_aperture: str) -> str:
    """
    Hierarchical fallback: assign most specific matching parent when no leaf-rule matches.
    This is your chosen policy.
    """
    if grip_aperture == "Minimal":
        return "PrecisionGrasp"
    if grip_aperture in ("Intermediate", "Maximal"):
        return "PowerGrasp"
    return "Grasp"

def rule_engine_predict(df_part: pd.DataFrame, thr: dict) -> np.ndarray:
    """
    Outputs exactly one of the 9 dataset labels:
    Large_Diameter, Thumb_Adducted, Quadpod, Tripod, Medium_Wrap, Small_Diameter,
    Power_Sphere, Sphere_3_Finger, Sphere_4_Finger
    """
    preds = []

    for _, r in df_part.iterrows():
        # ---- Flexion levels per aggregated distal feature ----
        lv = {}
        for c in SENSOR_COLS:
            low, high = thr[c]
            lv[c] = flex_level(float(r[c]), low, high)

        grip = str(r.get("Grip_Aperature", "")).strip()
        shape = str(r.get("Shape", "")).strip()

        # ---- Derived symbolic facts ----
        active_non_thumb = sum(1 for f in ["F_Index","F_Middle","F_Ring","F_Little"] if lv[f] != "LowFlexion")
        thumb_active = (lv["F_Thumb"] != "LowFlexion")

        highwrap = all(lv[f] == "HighFlexion" for f in ["F_Index","F_Middle","F_Ring","F_Little"])
        mediumwrap = sum(1 for f in ["F_Index","F_Middle","F_Ring","F_Little"] if lv[f] in ("MediumFlexion","HighFlexion")) >= 3

        tripod_pat = (
            lv["F_Index"] in ("MediumFlexion","HighFlexion") and
            lv["F_Middle"] in ("MediumFlexion","HighFlexion") and
            lv["F_Ring"] == "LowFlexion" and
            lv["F_Little"] == "LowFlexion" and
            thumb_active
        )

        # ---- Leaf inference rules (aligned to your labels) ----
        pred = None

        # ========== CYLINDER family ==========
        if shape.lower() == "cylinder":
            if grip == "Maximal" and highwrap:
                pred = "Large_Diameter"
            elif grip == "Intermediate" and highwrap:
                pred = "Small_Diameter"
            elif grip == "Intermediate" and mediumwrap:
                pred = "Medium_Wrap"
            else:
                # fallback within cylinder: choose nearest by aperture
                pred = "Large_Diameter" if grip == "Maximal" else "Medium_Wrap"

        # ========== SPHERE family ==========
        elif shape.lower() == "sphere":
            # High force full-hand sphere
            if grip == "Maximal" and active_non_thumb >= 4 and thumb_active and highwrap:
                pred = "Power_Sphere"
            # Tripod on sphere-like object
            elif tripod_pat and grip in ("Minimal", "Intermediate"):
                pred = "Tripod"
            # 3-finger sphere
            elif thumb_active and active_non_thumb == 2 and grip in ("Minimal", "Intermediate"):
                pred = "Sphere_3_Finger"
            # 4-finger sphere
            elif thumb_active and active_non_thumb == 3 and grip in ("Intermediate", "Maximal"):
                pred = "Sphere_4_Finger"
            # Quadpod (thumb + three digits engaged; more “precision support” than Sphere_4_Finger)
            elif thumb_active and active_non_thumb == 3 and grip in ("Minimal", "Intermediate"):
                pred = "Quadpod"
            else:
                # fallback for sphere
                if grip == "Maximal":
                    pred = "Power_Sphere"
                elif active_non_thumb >= 3:
                    pred = "Sphere_4_Finger"
                else:
                    pred = "Sphere_3_Finger"

        # ========== UNKNOWN SHAPE (use kinematics + aperture only) ==========
        else:
            # Tripod signature
            if tripod_pat and grip in ("Minimal", "Intermediate"):
                pred = "Tripod"
            # Thumb adducted is typically minimal aperture + low wrap + thumb engaged
            elif grip == "Minimal" and thumb_active and active_non_thumb <= 2:
                pred = "Thumb_Adducted"
            # Power / wrap cluster
            elif grip == "Maximal" and highwrap:
                pred = "Large_Diameter"
            elif grip == "Intermediate" and mediumwrap:
                pred = "Medium_Wrap"
            else:
                # conservative fallback
                pred = "Medium_Wrap" if grip == "Intermediate" else "Small_Diameter"

        # Final safety: enforce membership in the 9 allowed labels
        allowed = {
            "Large_Diameter","Thumb_Adducted","Quadpod","Tripod","Medium_Wrap","Small_Diameter",
            "Power_Sphere","Sphere_3_Finger","Sphere_4_Finger"
        }
        if pred not in allowed:
            pred = "Medium_Wrap"

        preds.append(pred)

    return np.array(preds, dtype=object)

# ----------------------------
# 4) BASELINE MODELS (DT / RF / ANN)
# ----------------------------
def get_models():
    return {
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=400,
            random_state=RANDOM_STATE
        ),
        # Optional:
        # "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale"),
        # "kNN": KNeighborsClassifier(n_neighbors=15),
    }

# ----------------------------
# 5) CV EVALUATION
# ----------------------------
def evaluate_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average="macro"),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
    }

def summarize_fold_metrics(metric_list):
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out

# ----------------------------
# 5b) RUNTIME MEASUREMENT
# ----------------------------
def time_predict_ms(predict_fn, X, n_repeats: int = 3) -> float:
    """
    Time predict_fn(X) using n_repeats warm runs.
    Returns mean per-sample inference time in milliseconds.
    Timing the full batch and dividing by N is more stable than
    timing individual samples (avoids Python loop overhead artefacts).
    """
    n = len(X)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    mean_total_s = float(np.mean(times))
    return (mean_total_s / n) * 1000.0  # ms per sample

def summarize_runtime(fold_times: list) -> tuple:
    """Return (mean_ms, std_ms) across folds."""
    arr = np.array(fold_times, dtype=float)
    return float(arr.mean()), float(arr.std())

# ----------------------------
# 5c) STATISTICAL SIGNIFICANCE (paired t-test + Cohen's d)
# ----------------------------
def cohens_d_paired(a: list, b: list) -> float:
    """
    Paired Cohen's d: mean(diff) / std(diff, ddof=1).
    Appropriate for repeated-measures / cross-validation fold comparisons.
    """
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    if diff.std(ddof=1) == 0:
        return float("inf")
    return float(diff.mean() / diff.std(ddof=1))

def run_ttest(rf_fold_metrics: list, rule_fold_metrics: list) -> dict:
    """
    Paired two-sided t-test (RF vs RuleEngine) for each metric.
    Returns dict keyed by metric with (t, p, d, rf_mean, rule_mean).
    """
    results = {}
    metric_keys = ["Accuracy", "MacroF1", "BalancedAcc"]
    for k in metric_keys:
        rf_vals   = [m[k] for m in rf_fold_metrics]
        rule_vals = [m[k] for m in rule_fold_metrics]
        t_stat, p_val = stats.ttest_rel(rf_vals, rule_vals)
        d = cohens_d_paired(rf_vals, rule_vals)
        results[k] = {
            "t":         round(float(t_stat), 3),
            "p":         float(p_val),
            "d":         round(d, 2),
            "rf_mean":   round(float(np.mean(rf_vals)), 4),
            "rule_mean": round(float(np.mean(rule_vals)), 4),
        }
    return results

def print_ttest_results(ttest_dict: dict, setting_label: str):
    print(f"\n=== Paired t-test results: RF vs RuleEngine [{setting_label}] ===")
    print(f"{'Metric':<15} {'t':>8} {'p':>12} {'d':>8}  {'RF mean':>10}  {'Rule mean':>10}")
    print("-" * 68)
    for metric, r in ttest_dict.items():
        sig = "*" if r["p"] < 0.05 else ""
        p_str = f"{r['p']:.2e}{sig}"
        print(f"{metric:<15} {r['t']:>8.3f} {p_str:>12} {r['d']:>8.2f}  {r['rf_mean']:>10.4f}  {r['rule_mean']:>10.4f}")

def run_cv(df: pd.DataFrame, use_context: bool):
    X_cols = SENSOR_COLS + (CONTEXT_COLS if use_context else [])
    X = df[X_cols].copy()
    y = df[TARGET_COL].astype(str).copy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # preprocessing for ML models
    num_cols = SENSOR_COLS
    cat_cols = CONTEXT_COLS if use_context else []

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    models = get_models()

    # collect ML metrics
    all_metrics = {"RuleEngine": []}
    for m in models:
        all_metrics[m] = []

    # ----------------------------
    # Ontology-specific fold storage (BEFORE loop)
    # ----------------------------
    fold_cov_list = []
    fold_viol_rate_list = []
    fold_viol_counts_list = []
    fold_stab_list = []

    # Runtime storage: ms per sample, one entry per fold per model
    fold_runtime_ms = {name: [] for name in list(get_models().keys()) + ["RuleEngine"]}

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()

        # Rule thresholds computed on training fold ONLY
        thr = compute_quantile_thresholds(Xtr)

        # Prepare test frame for rules (must contain sensor cols + context cols if present)
        df_test_for_rules = Xte.copy()
        # rule engine may not need TARGET_COL, but your implementation expects join(y)
        # so we keep the same style:
        df_test_for_rules_with_y = df_test_for_rules.join(yte.rename(TARGET_COL))

        # ----------------------------
        # Rule engine prediction + standard metrics
        # ----------------------------
        y_pred_rule = rule_engine_predict(df_test_for_rules_with_y, thr)
        all_metrics["RuleEngine"].append(evaluate_predictions(yte, y_pred_rule))

        # Runtime: rule engine
        fold_runtime_ms["RuleEngine"].append(
            time_predict_ms(
                lambda X: rule_engine_predict(X.join(yte.rename(TARGET_COL))
                                              if TARGET_COL not in X.columns else X, thr),
                df_test_for_rules
            )
        )

        # ----------------------------
        # Ontology metrics for this fold
        # ----------------------------
        # Coverage
        cov = compute_coverage(y_pred_rule)
        fold_cov_list.append(cov)

        # Constraint violations (only meaningful when context exists)
        # If sensor-only, Shape/Grip_Aperature columns won't exist; handle safely.
        if use_context and ("Shape" in df_test_for_rules.columns) and ("Grip_Aperature" in df_test_for_rules.columns):
            viol_rate, viol_counts = compute_violations(df_test_for_rules, y_pred_rule)
        else:
            viol_rate, viol_counts = 0.0, Counter()   # no context → no constraint checking
        fold_viol_rate_list.append(viol_rate)
        fold_viol_counts_list.append(viol_counts)

        # Prediction stability @2% noise
        rng = np.random.default_rng(1000 + fold)
        Xte_noisy = inject_noise(Xte, SENSOR_COLS, noise_pct=0.02, rng=rng)

        df_test_noisy = Xte_noisy.copy()
        if use_context:
            # copy context columns as-is (not noised)
            for c in CONTEXT_COLS:
                df_test_noisy[c] = Xte[c].values

        y_pred_noisy = rule_engine_predict(df_test_noisy.join(yte.rename(TARGET_COL)), thr)
        stab = prediction_stability(y_pred_rule, y_pred_noisy)
        fold_stab_list.append(stab)

        # ----------------------------
        # ML models
        # ----------------------------
        for name, clf in models.items():
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(Xtr, ytr)
            y_pred = pipe.predict(Xte)
            all_metrics[name].append(evaluate_predictions(yte, y_pred))
            # Runtime: time fitted pipeline on this fold's test set
            fold_runtime_ms[name].append(
                time_predict_ms(pipe.predict, Xte)
            )

        print(f"Fold {fold}/{N_SPLITS} done.")

    # ----------------------------
    # Summarize ontology metrics (AFTER loop)
    # ----------------------------
    leaf_cov = [d["leaf_coverage_pct"] for d in fold_cov_list]
    parent_cov = [d["parent_only_pct"] for d in fold_cov_list]
    other_cov = [d["other_pct"] for d in fold_cov_list]

    cov_summary = {
        "leaf_mean": float(np.mean(leaf_cov)), "leaf_std": float(np.std(leaf_cov)),
        "parent_mean": float(np.mean(parent_cov)), "parent_std": float(np.std(parent_cov)),
        "other_mean": float(np.mean(other_cov)), "other_std": float(np.std(other_cov)),
    }

    viol_summary = {
        "viol_rate_mean": float(np.mean(fold_viol_rate_list)),
        "viol_rate_std": float(np.std(fold_viol_rate_list))
    }

    stab_summary = {
        "pred_stability_mean": float(np.mean(fold_stab_list)),
        "pred_stability_std": float(np.std(fold_stab_list))
    }

    total_counts = Counter()
    for c in fold_viol_counts_list:
        total_counts.update(c)
    top3 = total_counts.most_common(3)

    print("\n=== Ontology-specific metrics ===")
    print("Coverage:", cov_summary)
    print("Violation rate (%):", viol_summary)
    print("Prediction stability @2% noise (%):", stab_summary)
    print("Top-3 constraint violations:", top3)

    # ----------------------------
    # Runtime summary (ms per sample, mean ± std across folds)
    # ----------------------------
    runtime_summary = {}
    print("\n=== Runtime: mean inference time per sample (ms) ===")
    print(f"{'Model':<15} {'mean ms':>10} {'std ms':>10}")
    print("-" * 38)
    for name, fold_times in fold_runtime_ms.items():
        mean_ms, std_ms = summarize_runtime(fold_times)
        runtime_summary[name] = {"mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4)}
        print(f"{name:<15} {mean_ms:>10.4f} {std_ms:>10.4f}")

    # ----------------------------
    # Statistical significance: RF vs RuleEngine (paired t-test)
    # ----------------------------
    ttest_results = run_ttest(all_metrics["RF"], all_metrics["RuleEngine"])
    setting_label = "sensor+context" if use_context else "sensor-only"
    print_ttest_results(ttest_results, setting_label)

    # ----------------------------
    # Summarize predictive metrics (ML + RuleEngine) into table
    # ----------------------------
    summary_rows = []
    for name, metrics in all_metrics.items():
        s = summarize_fold_metrics(metrics)
        summary_rows.append([
            name,
            f"{s['Accuracy'][0]:.4f} ± {s['Accuracy'][1]:.4f}",
            f"{s['MacroF1'][0]:.4f} ± {s['MacroF1'][1]:.4f}",
            f"{s['BalancedAcc'][0]:.4f} ± {s['BalancedAcc'][1]:.4f}",
        ])

    summary = pd.DataFrame(
        summary_rows,
        columns=["Model", "Accuracy (mean±std)", "Macro-F1 (mean±std)", "Balanced Acc (mean±std)"]
    )

    # Optional: return ontology summaries too (recommended)
    ontology_summary = {
        "coverage": cov_summary,
        "violations": viol_summary,
        "stability": stab_summary,
        "top3_constraints": top3
    }

    return summary, ontology_summary, runtime_summary, ttest_results


# ----------------------------
# 6) LaTeX TABLE EXPORT
# ----------------------------
def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption=caption,
        label=label
    )
    return latex

def runtime_to_latex(sensor_rt: dict, context_rt: dict) -> str:
    """
    Build a LaTeX table comparing per-sample inference time (ms) for all
    models under sensor-only and sensor+context configurations.
    Row order: RuleEngine first, then ML models alphabetically.
    """
    model_order = ["RuleEngine"] + sorted(
        [k for k in sensor_rt if k != "RuleEngine"]
    )
    display_names = {
        "RuleEngine": "Rule Engine",
        "DT":         "Decision Tree",
        "RF":         "Random Forest",
        "ANN":        "ANN",
    }

    rows = []
    for name in model_order:
        s  = sensor_rt.get(name,  {"mean_ms": float("nan"), "std_ms": float("nan")})
        c  = context_rt.get(name, {"mean_ms": float("nan"), "std_ms": float("nan")})
        rows.append({
            "Model":
                display_names.get(name, name),
            "Sensor-only (ms)":
                f"{s['mean_ms']:.4f} $\\pm$ {s['std_ms']:.4f}",
            "Sensor + context (ms)":
                f"{c['mean_ms']:.4f} $\\pm$ {c['std_ms']:.4f}",
        })

    df_rt = pd.DataFrame(rows)
    lines = [
        "\\begin{table}[h]",
        "    \\centering",
        "    \\small",
        "    \\caption{Mean per-sample inference time (ms, mean~$\\pm$~std across folds) "
        "for all models under sensor-only and sensor+context feature configurations. "
        "Lower is better. Measurements collected on identical hardware across all "
        "cross-validation folds.}",
        "    \\label{tab:runtime}",
        "    \\begin{tabular}{lcc}",
        "        \\hline",
        "        \\textbf{Model} & \\textbf{Sensor-only (ms)} & "
        "\\textbf{Sensor + context (ms)} \\\\",
        "        \\hline",
    ]
    for _, row in df_rt.iterrows():
        lines.append(
            f"        {row['Model']} & {row['Sensor-only (ms)']} & "
            f"{row['Sensor + context (ms)']} \\\\"
        )
    lines += [
        "        \\hline",
        "    \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)

def ttest_to_latex(sensor_tt: dict, context_tt: dict) -> str:
    """
    Build a LaTeX table of paired t-test results (RF vs RuleEngine) for both
    feature configurations.  Reports t-statistic, p-value, Cohen's d, and
    per-paradigm means.  Significant results (p < 0.05) are marked with *.
    """
    metric_labels = {
        "Accuracy":    "Accuracy",
        "MacroF1":     "Macro-F1",
        "BalancedAcc": "Balanced Acc.",
    }
    lines = [
        "\\begin{table}[h]",
        "    \\centering",
        "    \\small",
        "    \\caption{Paired $t$-test results comparing Random Forest (best ML baseline) "
        "against the ontology-driven rule engine across cross-validation folds. "
        "$t$: test statistic; $p$: two-sided $p$-value; $d$: Cohen's $d$ (paired). "
        "Significance level $\\alpha = 0.05$; * denotes $p < 0.05$.}",
        "    \\label{tab:ttest}",
        "    \\begin{tabular}{llrrrr}",
        "        \\hline",
        "        \\textbf{Setting} & \\textbf{Metric} & $t$ & $p$ & $d$ & "
        "\\textbf{RF mean} \\\\",
        "        \\hline",
    ]

    for setting_label, tt in [("Sensor-only", sensor_tt),
                               ("Sensor + context", context_tt)]:
        first = True
        nrows = len(tt)
        for i, (metric_key, r) in enumerate(tt.items()):
            sig   = "*" if r["p"] < 0.05 else ""
            p_str = f"{r['p']:.2e}{sig}"
            mrow  = metric_labels.get(metric_key, metric_key)

            # Use \multirow for the setting label in the first data row
            if first:
                setting_cell = (
                    f"\\multirow{{{nrows}}}{{*}}{{{setting_label}}}"
                )
                first = False
            else:
                setting_cell = ""

            lines.append(
                f"        {setting_cell} & {mrow} & "
                f"{r['t']:.3f} & ${p_str}$ & {r['d']:.2f} & "
                f"{r['rf_mean']:.4f} \\\\"
            )
        lines.append("        \\hline")

    lines += [
        "    \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    df = add_distal_aggregates(df)

    # keep only necessary columns + drop missing
    needed = SENSOR_COLS + CONTEXT_COLS + [TARGET_COL]
    df = df[needed].dropna().reset_index(drop=True)

    # -------------------------------------------------------
    # CV: sensor-only
    # -------------------------------------------------------
    print("\n=== Sensor-only CV ===")
    summary_sensor, onto_sensor, rt_sensor, tt_sensor = run_cv(df, use_context=False)

    summary_sensor.to_csv(os.path.join(OUT_DIR, "results_sensor_only.csv"), index=False)
    tex_sensor = df_to_latex_table(
        summary_sensor,
        caption="Cross-validation performance (sensor-only features).",
        label="tab:results-sensor-only"
    )
    with open(os.path.join(OUT_DIR, "results_sensor_only.tex"), "w") as f:
        f.write(tex_sensor)

    # -------------------------------------------------------
    # CV: sensor + context
    # -------------------------------------------------------
    print("\n=== Sensor + context CV ===")
    summary_context, onto_context, rt_context, tt_context = run_cv(df, use_context=True)

    summary_context.to_csv(os.path.join(OUT_DIR, "results_sensor_context.csv"), index=False)
    tex_context = df_to_latex_table(
        summary_context,
        caption="Cross-validation performance (sensor + contextual descriptors).",
        label="tab:results-sensor-context"
    )
    with open(os.path.join(OUT_DIR, "results_sensor_context.tex"), "w") as f:
        f.write(tex_context)

    # -------------------------------------------------------
    # Runtime table (combined: sensor-only + sensor+context)
    # -------------------------------------------------------
    tex_runtime = runtime_to_latex(rt_sensor, rt_context)
    with open(os.path.join(OUT_DIR, "results_runtime.tex"), "w") as f:
        f.write(tex_runtime)
    print("\n--- Runtime LaTeX table ---")
    print(tex_runtime)

    # Save runtime as CSV too
    rt_rows = []
    for name in rt_sensor:
        rt_rows.append({
            "Model":                  name,
            "SensorOnly_mean_ms":     rt_sensor[name]["mean_ms"],
            "SensorOnly_std_ms":      rt_sensor[name]["std_ms"],
            "SensorContext_mean_ms":  rt_context.get(name, {}).get("mean_ms", float("nan")),
            "SensorContext_std_ms":   rt_context.get(name, {}).get("std_ms",  float("nan")),
        })
    pd.DataFrame(rt_rows).to_csv(
        os.path.join(OUT_DIR, "results_runtime.csv"), index=False
    )

    # -------------------------------------------------------
    # T-test table (combined: sensor-only + sensor+context)
    # -------------------------------------------------------
    tex_ttest = ttest_to_latex(tt_sensor, tt_context)
    with open(os.path.join(OUT_DIR, "results_ttest.tex"), "w") as f:
        f.write(tex_ttest)
    print("\n--- T-test LaTeX table ---")
    print(tex_ttest)

    # Save t-test results as CSV
    tt_rows = []
    for setting, tt in [("sensor-only", tt_sensor), ("sensor+context", tt_context)]:
        for metric, r in tt.items():
            tt_rows.append({"Setting": setting, "Metric": metric, **r})
    pd.DataFrame(tt_rows).to_csv(
        os.path.join(OUT_DIR, "results_ttest.csv"), index=False
    )

    # -------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------
    print("\nSaved outputs to:", OUT_DIR)
    print("\n--- Predictive performance (sensor-only) ---")
    print(summary_sensor)
    print("\n--- Predictive performance (sensor + context) ---")
    print(summary_context)
