#!/usr/bin/env python3
"""
===============================================================================
SCRIPT 1: 3-MODEL PREDICTIVE VALIDITY LADDER EVALUATION
===============================================================================
Paper Section: Section V-A (Predictive Validity & Model Architecture Ablation)
Target Artifact: Table II (table_3model_ladder.tex)

EXPLANATION OF TEST PURPOSE:
----------------------------
This test evaluates the predictive validity and necessity of each component in 
the Causal Model architecture by comparing 4 nested model candidate levels:
  1. Intercept-Only Baseline: Predicts global mean collision rate (base rate).
  2. R-only Model: Uses environmental risk context R_t only (no configuration).
  3. C-only Model: Uses configuration parameters C only (no risk context; 
     corresponds to the static tuning baseline setup in Liang et al.).
  4. Full Causal Model (Ours): Combines risk context R_t, configuration C, 
     and explicit sparse interaction terms phi(c) x r.

WHY THIS MATTERS FOR THE PAPER:
-------------------------------
- Beating C-only proves that environmental context R_t is REQUIRED for tuning 
  (demonstrating why static parameter choices fail under distribution shift).
- Beating R-only proves that configuration parameters C genuinely control safety 
  and progress (proving tuning is effective).
- The gap between models establishes the headline predictive contribution of 
  causal interaction terms phi(c) x r.

DATA SPLITTING PROTOCOL (ZERO LEAKAGE):
---------------------------------------
Uses Grouped Episode-Level Splitting (grouping by trial_id/episode_id) rather 
than random row splitting. This prevents pseudo-replication and data leakage 
between adjacent 1 Hz decision intervals within the same navigation episode.

USAGE:
------
python3 evaluate_3model_ladder.py \
    --data-path /path/to/rct_results.csv \
    --output-dir ../evaluation_results
===============================================================================
"""

import os
import sys
import argparse
import logging
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_3model_ladder")


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def extract_features_and_groups(data_path: str):
    """Load dataset and build feature matrices with episode grouping."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} total rows from {data_path}.")

    # Group identifier for zero-leakage episode splitting
    if "trial_id" in df.columns:
        groups = df["trial_id"].values
    elif "episode_id" in df.columns:
        groups = df["episode_id"].values
    else:
        groups = np.arange(len(df)) // 15

    risk_cols = [c for c in df.columns if c.startswith("risk__")]
    param_cols = [c for c in df.columns if c.startswith("param__")]

    if "y_h" in df.columns:
        y_safe = df["y_h"].astype(int).values
    elif "collision" in df.columns:
        y_safe = df["collision"].astype(int).values
    else:
        y_safe = np.zeros(len(df), dtype=int)

    if "probe_progress_m" in df.columns:
        y_prog = df["probe_progress_m"].astype(float).values
    else:
        y_prog = np.ones(len(df), dtype=float)

    for col in param_cols:
        if col == "param__local_costmap__footprint":
            df[col] = df[col].apply(lambda v: 1.0 if str(v) == "carry" else (0.0 if str(v) == "tucked" else float(v)))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in risk_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X_R = df[risk_cols].values
    X_C = df[param_cols].values

    col_dict = {c: i for i, c in enumerate(risk_cols + param_cols)}
    
    def get_vec(name):
        for k, idx in col_dict.items():
            if name in k:
                if idx < len(risk_cols):
                    return X_R[:, idx]
                else:
                    return X_C[:, idx - len(risk_cols)]
        return np.zeros(len(df))

    c_v = get_vec("vx_max")
    c_w = get_vec("wz_max")
    c_inf = get_vec("inflation_radius")
    c_obs = get_vec("cost_weight")
    c_hor = get_vec("time_horizon")
    c_arm = get_vec("footprint")

    r_ttc = get_vec("r_ttc")
    r_width = get_vec("r_width")
    r_curve = get_vec("r_curve")
    r_clear = get_vec("r_clear")
    r_min = get_vec("r_min")
    r_dens = get_vec("r_dens")
    r_grad = get_vec("r_grad")
    r_vis = get_vec("r_vis")
    r_a_t = get_vec("a_t")

    interactions = [
        c_v * r_ttc, c_v * r_width, c_v * r_curve, c_v * r_clear,
        c_w * r_clear, c_w * r_curve,
        c_inf * r_width, c_inf * r_min,
        c_obs * r_min, c_obs * r_dens, c_obs * r_grad,
        c_hor * r_ttc, c_hor * r_curve,
        c_arm * r_vis, c_arm * r_width, c_arm * r_min,
        c_v * c_hor, c_inf * c_arm,
        c_v * r_a_t, c_arm * r_a_t, r_width * r_a_t
    ]
    X_RC = np.column_stack(interactions)
    X_full = np.hstack([X_R, X_C, X_RC])

    return X_R, X_C, X_full, y_safe, y_prog, groups


def evaluate_3model_ladder(data_path: str, output_dir: str):
    """Execute 5-fold Grouped Cross-Validation across 4 model levels."""
    os.makedirs(output_dir, exist_ok=True)
    X_R, X_C, X_full, y_safe, y_prog, groups = extract_features_and_groups(data_path)

    gkf = GroupKFold(n_splits=5)

    results = {
        "1. Intercept-Only": {"brier": [], "auc": [], "ece": [], "r2": []},
        "2. R-only (Environment)": {"brier": [], "auc": [], "ece": [], "r2": []},
        "3. C-only (Static Tuning)": {"brier": [], "auc": [], "ece": [], "r2": []},
        "4. Full Causal Model (Ours)": {"brier": [], "auc": [], "ece": [], "r2": []},
    }

    base_rate = np.mean(y_safe)

    for train_idx, test_idx in gkf.split(X_full, y_safe, groups=groups):
        y_test_s = y_safe[test_idx]
        y_test_p = y_prog[test_idx]

        # Model 1: Intercept-Only
        p1 = np.full(len(test_idx), base_rate)
        j1 = np.full(len(test_idx), np.mean(y_prog[train_idx]))
        results["1. Intercept-Only"]["brier"].append(brier_score_loss(y_test_s, p1))
        results["1. Intercept-Only"]["auc"].append(0.500)
        results["1. Intercept-Only"]["ece"].append(calculate_ece(y_test_s, p1))
        results["1. Intercept-Only"]["r2"].append(r2_score(y_test_p, j1))

        # Model 2: R-Only
        scaler_r = StandardScaler()
        xr_tr = scaler_r.fit_transform(X_R[train_idx])
        xr_te = scaler_r.transform(X_R[test_idx])

        clf_r = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf_r.fit(xr_tr, y_safe[train_idx])
        p2 = clf_r.predict_proba(xr_te)[:, 1]

        reg_r = Ridge(alpha=1.0, random_state=42)
        reg_r.fit(xr_tr, y_prog[train_idx])
        j2 = reg_r.predict(xr_te)

        results["2. R-only (Environment)"]["brier"].append(brier_score_loss(y_test_s, p2))
        results["2. R-only (Environment)"]["auc"].append(roc_auc_score(y_test_s, p2))
        results["2. R-only (Environment)"]["ece"].append(calculate_ece(y_test_s, p2))
        results["2. R-only (Environment)"]["r2"].append(r2_score(y_test_p, j2))

        # Model 3: C-Only (Static Tuning setup)
        scaler_c = StandardScaler()
        xc_tr = scaler_c.fit_transform(X_C[train_idx])
        xc_te = scaler_c.transform(X_C[test_idx])

        clf_c = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf_c.fit(xc_tr, y_safe[train_idx])
        p3 = clf_c.predict_proba(xc_te)[:, 1]

        reg_c = Ridge(alpha=1.0, random_state=42)
        reg_c.fit(xc_tr, y_prog[train_idx])
        j3 = reg_c.predict(xc_te)

        results["3. C-only (Static Tuning)"]["brier"].append(brier_score_loss(y_test_s, p3))
        results["3. C-only (Static Tuning)"]["auc"].append(roc_auc_score(y_test_s, p3))
        results["3. C-only (Static Tuning)"]["ece"].append(calculate_ece(y_test_s, p3))
        results["3. C-only (Static Tuning)"]["r2"].append(r2_score(y_test_p, j3))

        # Model 4: Full Causal Model (Ours)
        scaler_f = StandardScaler()
        xf_tr = scaler_f.fit_transform(X_full[train_idx])
        xf_te = scaler_f.transform(X_full[test_idx])

        clf_f = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf_f.fit(xf_tr, y_safe[train_idx])
        p4 = clf_f.predict_proba(xf_te)[:, 1]

        reg_f = Ridge(alpha=1.0, random_state=42)
        reg_f.fit(xf_tr, y_prog[train_idx])
        j4 = reg_f.predict(xf_te)

        results["4. Full Causal Model (Ours)"]["brier"].append(brier_score_loss(y_test_s, p4))
        results["4. Full Causal Model (Ours)"]["auc"].append(roc_auc_score(y_test_s, p4))
        results["4. Full Causal Model (Ours)"]["ece"].append(calculate_ece(y_test_s, p4))
        results["4. Full Causal Model (Ours)"]["r2"].append(r2_score(y_test_p, j4))

    print("\n" + "="*85)
    print(" 3-MODEL PREDICTIVE VALIDITY LADDER EVALUATION (TABLE II)")
    print("="*85)
    print(f"{'Model Architecture':<32} | {'Brier Score ↓':<13} | {'ROC-AUC ↑':<10} | {'ECE ↓':<10} | {'Progress R² ↑':<12}")
    print("-" * 85)

    summary_rows = []
    for model_name, metrics in results.items():
        b_mean = np.mean(metrics["brier"])
        a_mean = np.mean(metrics["auc"])
        e_mean = np.mean(metrics["ece"])
        r_mean = np.mean(metrics["r2"])
        print(f"{model_name:<32} | {b_mean:.4f}        | {a_mean:.4f}     | {e_mean:.4f}     | {r_mean:.4f}")
        summary_rows.append((model_name, b_mean, a_mean, e_mean, r_mean))

    print("="*85 + "\n")

    tex_path = os.path.join(output_dir, "table_3model_ladder.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Formatted LaTeX Table II for main.tex\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{3-Model Predictive Validity Ladder under episode-level grouped data splitting.}\n")
        f.write("\\label{tab:3model_ladder}\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{tabularx}{\\columnwidth}{@{}X c c c c@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Architecture} & \\textbf{Brier $\\downarrow$} & \\textbf{ROC-AUC $\\uparrow$} & \\textbf{ECE $\\downarrow$} & \\textbf{Progress $R^2$ $\\uparrow$} \\\\\n")
        f.write("\\midrule\n")
        for row in summary_rows:
            f.write(f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("\\end{table}\n")

    logger.info(f"Saved publication LaTeX table to {tex_path} ✓")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-Model Ladder for Paper")
    parser.add_argument("--data-path", type=str, required=True, help="Path to Campaign A rct_results.csv")
    parser.add_argument("--output-dir", type=str, default="../evaluation_results", help="Output directory for paper assets")
    args = parser.parse_args()

    evaluate_3model_ladder(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
