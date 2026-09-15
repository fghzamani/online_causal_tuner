#!/usr/bin/env python3
"""
Campaign B Closed-Loop Benchmarking Evaluation (P2.2 & P2.3).

Reads consolidated master evaluation CSV:
  campaign-b-data/campaign_b_evaluation_results.csv

Computes:
- Success Rate (%), Collision Rate (%), Plan Failure Rate (%), Timeout Rate (%), Stuck Rate (%)
- Measured Carry Fraction (%) & Out-of-Support Fraction (%)
- Reachability & Conditional Collision Rates on Common Support
- Formatted LaTeX Table VI (table_campaign_b_sota_benchmark.tex)
"""

import os
import sys
import math
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_campaign_b_closed_loop")


def compute_mcnemar_pvalue(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Exact McNemar test p-value for paired binary outcomes."""
    b = int(np.sum((y_a == 1) & (y_b == 0)))
    c = int(np.sum((y_a == 0) & (y_b == 1)))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p=0.5).pvalue)
    except ImportError:
        try:
            from scipy.stats import binom_test
            return float(binom_test(k, n, p=0.5))
        except ImportError:
            z = (abs(b - c) - 1.0) / math.sqrt(n)
            from scipy.stats import norm
            return float(2.0 * (1.0 - norm.cdf(abs(z))))


def reachability_and_conditional_collision(df: pd.DataFrame) -> pd.DataFrame:
    """P2.3: Separate reachability (plan creation) from conditional collision rate given a plan."""
    planned = df[df["status"] != "PLANNING_FAILED"]
    out = df.groupby("strategy").agg(
        n=("episode_id", "size"),
        reachability=("status", lambda s: (s != "PLANNING_FAILED").mean()),
    )
    out["coll_given_plan"] = planned.groupby("strategy")["collision"].mean()

    common = set.intersection(*(
        set(planned[planned["strategy"] == s]["episode_id"])
        for s in planned["strategy"].unique()
    )) if not planned.empty else set()

    if common:
        both = planned[planned["episode_id"].isin(common)]
        out["coll_common_support"] = both.groupby("strategy")["collision"].mean()
        out["n_common"] = len(common)
    else:
        out["coll_common_support"] = np.nan
        out["n_common"] = 0

    return out


def evaluate_campaign_b_closed_loop(results_csv: str, output_dir: str = "paper_artifacts"):
    if not os.path.exists(results_csv):
        logger.error(f"Results CSV file not found: {results_csv}")
        logger.error("Run Campaign B navigation trials first using run_campaign_b_benchmark.py!")
        sys.exit(1)

    df = pd.read_csv(results_csv)
    logger.info(f"Loaded Campaign B master evaluation results from {results_csv} ({len(df)} total trial rows).")

    # §3.3 Fix planning-failure distance artefact: set goal_distance_remaining to NaN for PLANNING_FAILED rows
    if "goal_distance_remaining" in df.columns:
        df.loc[df["status"] == "PLANNING_FAILED", "goal_distance_remaining"] = np.nan

    os.makedirs(output_dir, exist_ok=True)
    strategies = df["strategy"].unique()

    print("\n" + "=" * 140)
    print(" CAMPAIGN B CLOSED-LOOP BENCHMARKING VS. SOTA LITERATURE (TABLE VI)")
    print("=" * 140)
    print(f"{'Strategy':<24} | {'N':<4} | {'Reached ↑':<10} | {'Pose OK ↑':<10} | {'Collision ↓':<12} | {'Plan Fail ↓':<11} | {'Stuck/Timeout ↓':<15} | {'Carry Frac':<11} | {'Out-of-Supp':<12} | {'Mean Time (s) ↓'}")
    print("-" * 155)

    summary_rows = []
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        n = len(sub)
        reach_rate = (sub["reached"].mean() * 100.0) if "reached" in sub else (sub["success"].mean() * 100.0)
        pose_rate = (sub["pose_ok"].mean() * 100.0) if "pose_ok" in sub else (sub["success"].mean() * 100.0)
        coll_rate = sub["collision"].mean() * 100.0

        st = sub["status"].astype(str)
        plan_fail = (st == "PLANNING_FAILED").mean() * 100.0
        timeout = (st == "TIMEOUT").mean() * 100.0
        stuck = (st.isin(["STUCK", "FAILED"])).mean() * 100.0
        stuck_timeout = timeout + stuck

        # Partition check: Reached + Collision + PlanFail + Stuck/Timeout must be 100%.
        # PoseOK is a quality metric and is deliberately excluded from the sum.
        total = reach_rate + coll_rate + plan_fail + stuck_timeout
        if abs(total - 100.0) > 0.5:
            print(f"  WARNING: {strat} outcome partition sums to {total:.1f}%, not 100%. "
                  f"An unhandled status value is being dropped.")

        # P2.2: Measured only. NaN where run did not measure it, never a constant assigned from strategy name
        carry_s = sub["carry_fraction"].dropna()
        carry_frac = carry_s.mean() * 100.0 if not carry_s.empty else float("nan")

        out_supp_s = sub["out_of_support_fraction"].dropna()
        out_supp = out_supp_s.mean() * 100.0 if not out_supp_s.empty else float("nan")

        reach_col = "reached" if "reached" in sub else "success"
        succ_trials = sub[sub[reach_col] == 1]
        t_succ = succ_trials["travel_time_s"].mean() if len(succ_trials) else float("nan")
        t_std = succ_trials["travel_time_s"].std() if len(succ_trials) else float("nan")

        c_str = f"{carry_frac:.1f}%" if not math.isnan(carry_frac) else "N/A"
        o_str = f"{out_supp:.1f}%" if not math.isnan(out_supp) else "N/A"
        t_str = f"{t_succ:.1f} ± {t_std:.1f}s" if not math.isnan(t_succ) else "N/A"

        print(f"{strat:<24} | {n:<4} | {reach_rate:.1f}%      | {pose_rate:.1f}%     | {coll_rate:.1f}%        | {plan_fail:.1f}%        | {stuck_timeout:.1f}%           | {c_str:<11} | {o_str:<12} | {t_str}")
        summary_rows.append((strat, n, reach_rate, pose_rate, coll_rate, plan_fail, stuck_timeout, carry_frac, out_supp, t_succ, t_std))

    print("=" * 155 + "\n")

    # P2.3: Reachability and Common Support Analysis
    reach_df = reachability_and_conditional_collision(df)
    print("--- Reachability & Common Support Analysis ---")
    print(reach_df.to_string())
    print("\n")

    # Paired McNemar Test vs. Ours
    if "Online Causal (Ours)" in strategies:
        df_ours = df[df["strategy"] == "Online Causal (Ours)"].sort_values("episode_id")
        for strat in strategies:
            if strat == "Online Causal (Ours)":
                continue
            df_other = df[df["strategy"] == strat].sort_values("episode_id")
            if len(df_ours) == len(df_other) and len(df_ours) > 0:
                col_mcn = "reached" if "reached" in df_ours else "success"
                p_mcn = compute_mcnemar_pvalue(df_ours[col_mcn].values, df_other[col_mcn].values)
                print(f"Paired McNemar Test (Ours vs '{strat}'): p-value = {p_mcn:.4f}")

    # Save LaTeX Table VI with distinct name per world
    csv_basename = os.path.basename(results_csv).replace(".csv", "")
    if "causal_benchmark_2" in csv_basename:
        tex_filename = "table_causal_benchmark_2_sota_benchmark.tex"
    else:
        tex_filename = "table_campaign_b_sota_benchmark.tex"
    tex_path = os.path.join(output_dir, tex_filename)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Formatted LaTeX Table VI for main.tex\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Campaign B Closed-Loop Benchmarking in unseen complex Gazebo environment vs. SOTA literature.}\n")
        f.write("\\label{tab:campaign_b_sota}\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{tabularx}{\\columnwidth}{@{}X c c c c c c c c@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Strategy} & \\textbf{Reached \\% $\\uparrow$} & \\textbf{Pose OK \\% $\\uparrow$} & \\textbf{Collision \\% $\\downarrow$} & \\textbf{Plan Fail \\% $\\downarrow$} & \\textbf{Stuck \\% $\\downarrow$} & \\textbf{Carry \\% $\\uparrow$} & \\textbf{Out-of-Supp \\%} & \\textbf{Mean Time (s) $\\downarrow$} \\\\\n")
        f.write("\\midrule\n")
        for r in summary_rows:
            c_s = f"{r[7]:.1f}\\%" if not math.isnan(r[7]) else "--"
            o_s = f"{r[8]:.1f}\\%" if not math.isnan(r[8]) else "--"
            t_s = f"{r[9]:.1f} $\\pm$ {r[10]:.1f}s" if not math.isnan(r[9]) else "--"
            f.write(f"{r[0]} & {r[2]:.1f}\\% & {r[3]:.1f}\\% & {r[4]:.1f}\\% & {r[5]:.1f}\\% & {r[6]:.1f}\\% & {c_s} & {o_s} & {t_s} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("\\end{table}\n")

    logger.info(f"Saved publication LaTeX table to {tex_path} ✓")


def main():
    default_csv = "/home/forough/phd_projects/online_tuner/campaign-b-data/campaign_b_evaluation_results.csv"
    default_paper_dir = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/paper_artifacts"
    parser = argparse.ArgumentParser(description="Evaluate Campaign B Closed-Loop Benchmarks")
    parser.add_argument("--results-csv", type=str, default=default_csv, help="Path to master campaign_b_evaluation_results.csv")
    parser.add_argument("--output-dir", type=str, default=default_paper_dir, help="Output directory")
    args = parser.parse_args()

    evaluate_campaign_b_closed_loop(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()
