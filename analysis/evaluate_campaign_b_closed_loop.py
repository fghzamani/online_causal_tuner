#!/usr/bin/env python3
"""
Campaign B Closed-Loop Benchmarking Evaluation.

Evaluates full-mission navigation trials across Campaign B strategies:
1. Nav2 Shipping Default
2. Static Best-Fixed (Campaign A)
3. Full MOBO Auto-Tuning
4. CURE (Hossen et al. 2025)
5. Triggered-Intervention Ablation
6. Online Causal Tuner (Ours)

Computes:
- Paired McNemar's test for binary success and collision outcomes.
- Paired Wilcoxon signed-rank test for continuous travel time.
- Formatted LaTeX Table VI (table_campaign_b_sota_benchmark.tex) and Pareto plot (fig_pareto_frontier.pdf).
"""

import os
import sys
import math
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def evaluate_campaign_b_closed_loop(results_csv: str, output_dir: str = "paper_artifacts"):
    if not os.path.exists(results_csv):
        logger.error(f"Results CSV file not found: {results_csv}")
        logger.error("Run Campaign B navigation trials first to collect real results!")
        sys.exit(1)

    df = pd.read_csv(results_csv)
    logger.info(f"Loaded Campaign B results from {results_csv} ({len(df)} total trial rows).")

    os.makedirs(output_dir, exist_ok=True)

    strategies = df["strategy"].unique()

    print("\n" + "=" * 95)
    print(" CAMPAIGN B CLOSED-LOOP BENCHMARKING VS. SOTA LITERATURE (TABLE VI)")
    print("=" * 95)
    print(f"{'Strategy':<26} | {'Success Rate ↑':<15} | {'Collision Rate ↓':<16} | {'Blocked Rate ↓':<14} | {'Mean Time (s) ↓'}")
    print("-" * 95)

    summary_rows = []
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        succ_rate = sub["success"].mean() * 100.0
        coll_rate = sub["collision"].mean() * 100.0
        block_rate = sub["blocked"].mean() * 100.0

        succ_trials = sub[sub["success"] == 1]
        if len(succ_trials) > 0:
            t_mean = succ_trials["travel_time_s"].mean()
            t_std = succ_trials["travel_time_s"].std()
        else:
            t_mean, t_std = float("nan"), float("nan")

        print(f"{strat:<26} | {succ_rate:.1f}%             | {coll_rate:.1f}%               | {block_rate:.1f}%             | {t_mean:.1f} ± {t_std:.1f}s")
        summary_rows.append((strat, succ_rate, coll_rate, block_rate, t_mean, t_std))

    print("=" * 95 + "\n")

    # Paired McNemar Test vs. Ours
    if "Online Causal (Ours)" in strategies and "CURE (Hossen 2025)" in strategies:
        df_ours = df[df["strategy"] == "Online Causal (Ours)"].sort_values("episode_id")
        df_cure = df[df["strategy"] == "CURE (Hossen 2025)"].sort_values("episode_id")

        if len(df_ours) == len(df_cure) and len(df_ours) > 0:
            p_mcn = compute_mcnemar_pvalue(df_ours["success"].values, df_cure["success"].values)
            print(f"Paired McNemar Test (Ours vs CURE SOTA): p-value = {p_mcn:.4f}")

            w_stat, w_p = wilcoxon(df_ours["travel_time_s"].values, df_cure["travel_time_s"].values)
            print(f"Paired Wilcoxon Test on Travel Time (Ours vs CURE SOTA): p-value = {w_p:.4f}\n")

    # Save LaTeX Table VI
    tex_path = os.path.join(output_dir, "table_campaign_b_sota_benchmark.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Formatted LaTeX Table VI for main.tex\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Campaign B Closed-Loop Benchmarking in unseen complex Gazebo environment vs. SOTA literature.}\n")
        f.write("\\label{tab:campaign_b_sota}\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{tabularx}{\\columnwidth}{@{}X c c c c@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Strategy} & \\textbf{Success \\% $\\uparrow$} & \\textbf{Collision \\% $\\downarrow$} & \\textbf{Blocked \\% $\\downarrow$} & \\textbf{Mean Time (s) $\\downarrow$} \\\\\n")
        f.write("\\midrule\n")
        for r in summary_rows:
            f.write(f"{r[0]} & {r[1]:.1f}\\% & {r[2]:.1f}\\% & {r[3]:.1f}\\% & {r[4]:.1f} $\\pm$ {r[5]:.1f}s \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("\\end{table}\n")

    logger.info(f"Saved publication LaTeX table to {tex_path} ✓")

    # Plot Pareto Frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Nav2 Shipping Default": "#7f7f7f", "Static Best-Fixed": "#ff7f0e", "MOBO Auto-Tuning": "#1f77b4", "CURE (Hossen 2025)": "#9467bd", "Triggered Intervention": "#e377c2", "Online Causal (Ours)": "#d62728"}
    markers = {"Nav2 Shipping Default": "s", "Static Best-Fixed": "^", "MOBO Auto-Tuning": "D", "CURE (Hossen 2025)": "v", "Triggered Intervention": "P", "Online Causal (Ours)": "*"}

    for r in summary_rows:
        name = r[0]
        c_rate = r[2]
        speed_est = 100.0 / r[4] if math.isfinite(r[4]) else 0.0
        ax.scatter(c_rate, speed_est, color=colors.get(name, "blue"), marker=markers.get(name, "o"), s=150 if "Ours" in name else 100, label=name, zorder=5)

    ax.set_xlabel("Collision Rate (%) ↓ [Safer Left]")
    ax.set_ylabel("Normalized Progress Speed (100 / Time) ↑ [Faster Up]")
    ax.set_title("Safety-Progress Pareto Frontier vs. SOTA Literature")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig_pdf = os.path.join(output_dir, "fig_pareto_frontier.pdf")
    fig_png = os.path.join(output_dir, "fig_pareto_frontier.png")
    plt.savefig(fig_pdf, dpi=300)
    plt.savefig(fig_png, dpi=300)
    plt.close()
    logger.info(f"Saved publication Pareto plots to {fig_pdf} and {fig_png} ✓")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Campaign B Closed-Loop Benchmarks")
    parser.add_argument("--results-csv", type=str, required=True, help="Path to campaign_b_results.csv")
    parser.add_argument("--output-dir", type=str, default="paper_artifacts", help="Output directory")
    args = parser.parse_args()

    evaluate_campaign_b_closed_loop(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()
