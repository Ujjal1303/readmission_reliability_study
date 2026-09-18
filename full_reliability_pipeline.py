# ============================================================
# Reliability of Clinical Prediction Under Missing Information
# MCAR / MAR / MNAR — Multi-Dataset Study
#
# Dataset 1: Diabetes 130-US Hospitals (30-day readmission)
# Dataset 2: Early Stage Diabetes Risk Prediction (diagnosis)
#   -> run independently, NOT merged (different task/population).
#      Framed as cross-domain generalization of the same
#      missingness-reliability methodology.
#
# Fixes applied vs. the original notebook:
#   1. MAR missingness depends on MULTIPLE observed drivers (age +
#      length-of-stay / utilization columns), strongly modulated
#      (was age-only too weak -> MAR looked like MCAR)
#   2. Seeds increased 3 -> 10 for valid Wilcoxon/paired-t tests
#   3. scale_pos_weight added to XGBoost (was missing -> near-zero
#      recall on the imbalanced readmission target)
#   4. XGBoost runs on GPU (device="cuda") for RTX 4060
#   5. One definition per function, single top-to-bottom run
#   6. Bonferroni-corrected p-values + Cohen's d
#
# Run: python full_reliability_pipeline.py
# Requires: pandas, numpy, scikit-learn, xgboost>=2.0, scipy
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit these paths to match your machine
# ============================================================

READMISSION_DATA_PATH = "diabetic_data.csv"
EARLY_STAGE_DATA_PATH = "diabetes_data_upload.csv"

OUTPUT_DIR = "final_research_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MISSINGNESS_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
MECHANISMS = ["MCAR", "MAR", "MNAR"]
APPROACHES = ["Native", "Explicit Imputation"]

# Increased from 3 -> 10 -> 20 seeds to shrink CIs around the
# borderline MCAR-vs-MAR divergence. Raise further if GPU time allows.
SEEDS = list(range(42, 242, 10))  # 42..232 -> 20 seeds (first 10 = original)


def get_device():
    try:
        import subprocess
        subprocess.check_output(["nvidia-smi"])
        return "cuda"
    except Exception:
        print("No GPU detected via nvidia-smi -> falling back to CPU.")
        return "cpu"


DEVICE = get_device()
print(f"XGBoost device: {DEVICE}")


# ============================================================
# 1. DATA LOADERS (one per dataset — kept separate on purpose)
# ============================================================

def load_readmission_data(path: str) -> pd.DataFrame:
    """Diabetes 130-US Hospitals. Target: 30-day readmission."""
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["encounter_id", "patient_nbr", "readmitted"])

    # discharge_disposition_id dropped as a leakage-prone feature
    if "discharge_disposition_id" in df.columns:
        df = df.drop(columns=["discharge_disposition_id"])

    df = df.drop_duplicates()
    return df


def load_early_stage_data(path: str) -> pd.DataFrame:
    """Early Stage Diabetes Risk Prediction. Target: diagnosis class."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["target"] = (df["class"] == "Positive").astype(int)
    df = df.drop(columns=["class"])
    df = df.drop_duplicates()
    return df


# ============================================================
# 2. MISSINGNESS MECHANISMS (fixed versions, dataset-agnostic)
# ============================================================

def apply_mcar(X: pd.DataFrame, missing_rate: float, seed: int) -> pd.DataFrame:
    """Missing Completely at Random: every observed cell has an
    equal, independent probability of being masked."""
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    for col in X_out.columns:
        observed = X_out[col].notna()
        mask = (rng.random(len(X_out)) < missing_rate) & observed
        X_out.loc[mask, col] = np.nan
    return X_out


def _numeric_age_from_bracket(age_series: pd.Series) -> pd.Series:
    """Handles the readmission dataset's '[60-70)' style age brackets.
    Falls back to using the series directly if it's already numeric."""
    if pd.api.types.is_numeric_dtype(age_series):
        return age_series.astype(float)

    extracted = age_series.astype(str).str.extract(r"\[(\d+)-(\d+)\)")
    if extracted.notna().any().any():
        return extracted.astype(float).mean(axis=1)

    # Not a bracket format -> try direct numeric coercion
    return pd.to_numeric(age_series, errors="coerce")


def apply_mar(
    X: pd.DataFrame,
    missing_rate: float,
    seed: int,
    driver_col: str = "Age",
) -> pd.DataFrame:
    """Missing at Random: probability of missingness depends on MULTIPLE
    OBSERVED driver variables, not on the value being masked.

    FIX v2: age alone was too weak a signal (v1 produced an MCAR/MAR gap
    smaller than seed-to-seed noise). Now combines age with up to 2 more
    numeric columns (e.g. time_in_hospital, num_lab_procedures) so the
    missingness pattern is structurally distinct from MCAR.

    driver_col is resolved case-insensitively so this works for both
    datasets ('age' in the readmission set, 'Age' in the early-stage set).
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()

    # Secondary drivers: prefer clinically meaningful length-of-stay /
    # utilization columns; fall back to any other numeric columns.
    SECONDARY_DRIVER_PRIORITY = [
        "time_in_hospital",
        "num_lab_procedures",
        "number_diagnoses",
        "num_procedures",
        "num_medications",
        "number_inpatient",
        "number_outpatient",
        "number_emergency",
    ]

    actual_driver = None
    for col in X_out.columns:
        if col.lower() == driver_col.lower():
            actual_driver = col
            break
    if actual_driver is None:
        # No age column found -> fall back to MCAR behaviour safely
        return apply_mcar(X_out, missing_rate, seed)

    age_numeric = _numeric_age_from_bracket(X_out[actual_driver])
    age_numeric = age_numeric.fillna(age_numeric.median())

    age_min, age_max = age_numeric.min(), age_numeric.max()
    if age_max == age_min:
        age_norm = pd.Series(0.5, index=X_out.index)
    else:
        age_norm = (age_numeric - age_min) / (age_max - age_min)

    numeric_cols = set(X_out.select_dtypes(include=["int64", "float64"]).columns)
    secondary_candidates = [
        c for c in SECONDARY_DRIVER_PRIORITY
        if c in numeric_cols and c != actual_driver
    ]
    secondary_candidates += [
        c for c in X_out.columns
        if c in numeric_cols and c != actual_driver and c not in secondary_candidates
    ]
    secondary_candidates = secondary_candidates[:2]

    combined_norm = age_norm.copy()
    n_components = 1
    for col in secondary_candidates:
        vals = pd.to_numeric(X_out[col], errors="coerce")
        vals = vals.fillna(vals.median())
        vmin, vmax = vals.min(), vals.max()
        norm = (
            pd.Series(0.5, index=X_out.index)
            if vmax == vmin
            else (vals - vmin) / (vmax - vmin)
        )
        combined_norm = combined_norm + norm
        n_components += 1

    combined_norm = combined_norm / n_components  # in [0,1]

    # --- wide spread: 0.1x to 3.1x on the COMBINED signal ---
    weights = 0.1 + 3.0 * combined_norm

    for col in X_out.columns:
        if col == actual_driver:
            continue
        probability = missing_rate * weights / weights.mean()
        probability = np.clip(probability, 0, 1)
        observed = X_out[col].notna()
        mask = (rng.random(len(X_out)) < probability) & observed
        X_out.loc[mask, col] = np.nan

    return X_out


def apply_mnar(X: pd.DataFrame, missing_rate: float, seed: int) -> pd.DataFrame:
    """Missing Not at Random: probability of missingness depends on the
    feature's OWN value (higher values / rarer categories more likely
    to be masked). Exact-count weighted sampling without replacement."""
    rng = np.random.default_rng(seed)
    X_out = X.copy()

    for col in X_out.columns:
        observed_idx = X_out.index[X_out[col].notna()]
        if len(observed_idx) == 0:
            continue

        n_missing = int(round(len(observed_idx) * missing_rate))
        if n_missing == 0:
            continue
        if n_missing >= len(observed_idx):
            X_out.loc[observed_idx, col] = np.nan
            continue

        if pd.api.types.is_numeric_dtype(X_out[col]):
            values = pd.to_numeric(X_out.loc[observed_idx, col], errors="coerce")
            values_filled = values.fillna(values.median())
            vmin, vmax = values_filled.min(), values_filled.max()
            scores = (
                np.ones(len(observed_idx))
                if vmax == vmin
                else 0.5 + (values_filled - vmin) / (vmax - vmin)
            )
        else:
            freqs = X_out.loc[observed_idx, col].value_counts(normalize=True)
            value_score = X_out.loc[observed_idx, col].map(freqs).fillna(0)
            smin, smax = value_score.min(), value_score.max()
            scores = (
                np.ones(len(observed_idx))
                if smax == smin
                else 0.5 + (value_score - smin) / (smax - smin)
            )

        scores = np.asarray(scores, dtype=float)
        probabilities = scores / scores.sum()

        selected_positions = rng.choice(
            len(observed_idx), size=n_missing, replace=False, p=probabilities
        )
        selected_idx = observed_idx[selected_positions]
        X_out.loc[selected_idx, col] = np.nan

    return X_out


MISSINGNESS_FUNCS = {"MCAR": apply_mcar, "MAR": apply_mar, "MNAR": apply_mnar}


# ============================================================
# 3. PREPROCESSING PIPELINES (shared across both datasets)
# ============================================================

def build_imputed_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_native_preprocessor(numeric_features, categorical_features):
    # No imputer: NaNs pass through / are ordinal-encoded as NaN so
    # XGBoost's native missing-value handling applies.
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                    dtype=np.float32,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_xgb(scale_pos_weight: float, seed: int) -> XGBClassifier:
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,  # FIX: addresses near-zero recall
    )
    if DEVICE == "cuda":
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    else:
        params["tree_method"] = "hist"
        params["n_jobs"] = -1
    return XGBClassifier(**params)


# ============================================================
# 4. METRICS
# ============================================================

def calculate_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        mean_pred = y_prob[mask].mean()
        frac_pos = y_true[mask].mean()
        weight = mask.sum() / len(y_true)
        ece += weight * abs(mean_pred - frac_pos)
    return ece


def evaluate(pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_test, y_prob),
        "AUPRC": average_precision_score(y_test, y_prob),
        "Brier": brier_score_loss(y_test, y_prob),
        "ECE": calculate_ece(y_test, y_prob),
    }


# ============================================================
# 5. GENERIC EXPERIMENT RUNNER (used for both datasets)
# ============================================================

def run_experiment(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    print(f"\n{'='*60}\nRunning experiment on: {dataset_name}\n{'='*60}")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"[{dataset_name}] scale_pos_weight = {scale_pos_weight:.3f} "
          f"| n={len(df)} | positive rate={y.mean():.3f}")

    all_runs = []
    total_jobs = len(APPROACHES) * len(MECHANISMS) * len(MISSINGNESS_LEVELS) * len(SEEDS)
    job_i = 0

    for approach in APPROACHES:
        for mechanism in MECHANISMS:
            missingness_fn = MISSINGNESS_FUNCS[mechanism]
            for level in MISSINGNESS_LEVELS:
                for seed in SEEDS:
                    job_i += 1
                    print(f"[{dataset_name}] [{job_i}/{total_jobs}] {approach} | "
                          f"{mechanism} | {int(level*100)}% | seed={seed}")

                    X_train_missing = missingness_fn(X_train, level, seed=seed)
                    X_test_missing = missingness_fn(X_test, level, seed=seed + 10_000)

                    if approach == "Native":
                        preprocessor = build_native_preprocessor(
                            numeric_features, categorical_features
                        )
                    else:
                        preprocessor = build_imputed_preprocessor(
                            numeric_features, categorical_features
                        )

                    model = build_xgb(scale_pos_weight=scale_pos_weight, seed=seed)
                    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

                    metrics = evaluate(
                        pipeline, X_train_missing, y_train, X_test_missing, y_test
                    )
                    metrics.update(
                        Dataset=dataset_name,
                        Approach=approach,
                        Mechanism=mechanism,
                        Missingness=int(level * 100),
                        Seed=seed,
                    )
                    all_runs.append(metrics)

    results_df = pd.DataFrame(all_runs)
    out_path = f"{OUTPUT_DIR}/robustness_results_{dataset_name}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return results_df


# ============================================================
# 6. SUMMARY TABLES + STATISTICAL TESTS (dataset-agnostic)
# ============================================================

def build_summary_tables(results_df: pd.DataFrame, dataset_name: str):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC", "Brier", "ECE"]
    group_cols = ["Approach", "Mechanism", "Missingness"]

    # --- Mean +/- SD summary ---
    summary = results_df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s.capitalize()}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(f"{OUTPUT_DIR}/summary_mean_sd_{dataset_name}.csv", index=False)

    # --- 95% CI table (t-distribution, correct for small n) ---
    ci_rows = []
    for keys, group in results_df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["N"] = len(group)
        for metric in ["AUROC", "AUPRC", "Brier", "ECE"]:
            vals = group[metric].values
            mean, sd = vals.mean(), vals.std(ddof=1)
            se = sd / np.sqrt(len(vals))
            tcrit = stats.t.ppf(0.975, df=len(vals) - 1)
            row[f"{metric}_Mean"] = mean
            row[f"{metric}_CI_Lower"] = mean - tcrit * se
            row[f"{metric}_CI_Upper"] = mean + tcrit * se
        ci_rows.append(row)
    pd.DataFrame(ci_rows).to_csv(f"{OUTPUT_DIR}/confidence_intervals_95_{dataset_name}.csv", index=False)

    # --- Statistical tests: 10% vs 50% missingness, Bonferroni-corrected ---
    test_rows = []
    combos = results_df[["Approach", "Mechanism"]].drop_duplicates().values
    n_tests = len(combos)

    for approach, mechanism in combos:
        sub = results_df[
            (results_df["Approach"] == approach) & (results_df["Mechanism"] == mechanism)
        ]
        low = sub[sub["Missingness"] == 10].sort_values("Seed")["AUROC"].values
        high = sub[sub["Missingness"] == 50].sort_values("Seed")["AUROC"].values

        t_stat, t_p = stats.ttest_rel(low, high)
        w_stat, w_p = stats.wilcoxon(low, high)

        diff = low - high
        cohens_d = diff.mean() / diff.std(ddof=1)

        test_rows.append({
            "Approach": approach,
            "Mechanism": mechanism,
            "AUROC_10_Mean": low.mean(),
            "AUROC_50_Mean": high.mean(),
            "Mean_Difference": diff.mean(),
            "Cohens_D": cohens_d,
            "Paired_t_stat": t_stat,
            "Paired_t_p": t_p,
            "Paired_t_p_bonferroni": min(t_p * n_tests, 1.0),
            "Wilcoxon_stat": w_stat,
            "Wilcoxon_p": w_p,
            "Wilcoxon_p_bonferroni": min(w_p * n_tests, 1.0),
        })

    pd.DataFrame(test_rows).to_csv(
        f"{OUTPUT_DIR}/statistical_tests_10_vs_50_{dataset_name}.csv", index=False
    )

    # --- MAR vs MCAR divergence check (sanity check for the fix) ---
    # Reports the mean AUROC gap paired per-seed (n = len(SEEDS)) with a
    # paired t-test, Wilcoxon, and the gap relative to its standard error
    # so the mechanism effect is judged against seed-to-seed noise.
    divergence_rows = []
    for approach in results_df["Approach"].unique():
        for level in results_df["Missingness"].unique():
            mcar = results_df[
                (results_df.Approach == approach)
                & (results_df.Mechanism == "MCAR")
                & (results_df.Missingness == level)
            ].sort_values("Seed")["AUROC"].values
            mar = results_df[
                (results_df.Approach == approach)
                & (results_df.Mechanism == "MAR")
                & (results_df.Missingness == level)
            ].sort_values("Seed")["AUROC"].values

            signed_diff = mar - mcar
            gap = abs(signed_diff.mean())
            se_diff = (
                signed_diff.std(ddof=1) / np.sqrt(len(signed_diff))
                if len(signed_diff) > 1
                else np.nan
            )
            gap_over_se = gap / se_diff if se_diff and se_diff > 0 else np.nan

            if len(signed_diff) >= 8:
                t_stat, t_p = stats.ttest_rel(mcar, mar)
                w_stat, w_p = stats.wilcoxon(mcar, mar)
            else:
                t_stat = t_p = w_stat = w_p = np.nan

            divergence_rows.append({
                "Approach": approach,
                "Missingness": level,
                "MCAR_AUROC": mcar.mean(),
                "MAR_AUROC": mar.mean(),
                "Absolute_Gap": gap,
                "Gap_SE": se_diff,
                "Gap_Over_SE": gap_over_se,
                "Paired_t_stat": t_stat,
                "Paired_t_p": t_p,
                "Wilcoxon_stat": w_stat,
                "Wilcoxon_p": w_p,
                "Gap_Ge_0_01": bool(gap >= 0.01),
            })
    pd.DataFrame(divergence_rows).to_csv(
        f"{OUTPUT_DIR}/mar_vs_mcar_divergence_{dataset_name}.csv", index=False
    )

    print(f"[{dataset_name}] Summary tables saved to {OUTPUT_DIR}")


# ============================================================
# 7. CROSS-DATASET COMPARISON TABLE
# ============================================================

def build_cross_dataset_comparison(all_results: dict):
    """Combines both datasets' summaries into one comparison table,
    for the 'does the pattern generalize across domains' figure/table."""
    rows = []
    for dataset_name, results_df in all_results.items():
        for keys, group in results_df.groupby(["Approach", "Mechanism", "Missingness"]):
            approach, mechanism, level = keys
            rows.append({
                "Dataset": dataset_name,
                "Approach": approach,
                "Mechanism": mechanism,
                "Missingness": level,
                "AUROC_Mean": group["AUROC"].mean(),
                "AUROC_SD": group["AUROC"].std(ddof=1),
                "ECE_Mean": group["ECE"].mean(),
                "ECE_SD": group["ECE"].std(ddof=1),
            })
    combined = pd.DataFrame(rows)
    combined.to_csv(f"{OUTPUT_DIR}/cross_dataset_comparison.csv", index=False)
    print(f"Cross-dataset comparison saved -> {OUTPUT_DIR}/cross_dataset_comparison.csv")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    all_results = {}

    # --- Dataset 1: Readmission (primary task) ---
    if os.path.exists(READMISSION_DATA_PATH):
        df_readm = load_readmission_data(READMISSION_DATA_PATH)
        results_readm = run_experiment(df_readm, dataset_name="readmission")
        build_summary_tables(results_readm, dataset_name="readmission")
        all_results["readmission"] = results_readm
    else:
        print(f"WARNING: {READMISSION_DATA_PATH} not found, skipping readmission dataset.")

    # --- Dataset 2: Early-stage diabetes diagnosis (generalization check) ---
    if os.path.exists(EARLY_STAGE_DATA_PATH):
        df_early = load_early_stage_data(EARLY_STAGE_DATA_PATH)
        results_early = run_experiment(df_early, dataset_name="early_stage_diabetes")
        build_summary_tables(results_early, dataset_name="early_stage_diabetes")
        all_results["early_stage_diabetes"] = results_early
    else:
        print(f"WARNING: {EARLY_STAGE_DATA_PATH} not found, skipping early-stage dataset.")

    if len(all_results) >= 2:
        build_cross_dataset_comparison(all_results)

    print("\nAll experiments complete. Check the "
          "mar_vs_mcar_divergence_*.csv files first: the Absolute_Gap "
          "column should now be clearly non-trivial (not ~0.001 like "
          "in the original, unfixed MAR function).")


if __name__ == "__main__":
    main()
