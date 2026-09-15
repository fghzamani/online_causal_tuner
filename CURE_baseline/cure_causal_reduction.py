#!/usr/bin/env python3
"""
CURE stage 1 + 2: causal structure learning and causal reduction.

This is a faithful port of what `run_cure_MOO.py` does in
https://github.com/softsys4ai/cure before it hands off to the optimizer,
adapted to Campaign A data, ROS 2 and Nav2.

What CURE does here, in plain terms
-----------------------------------
1. Learn a causal graph over (options, metrics, objectives) from observational
   data using FCI (Fast Causal Inference), constrained by tabu edges that
   forbid physically impossible directions.
2. Resolve FCI's output into an ADMG (a graph with directed edges A -> B and
   bidirected edges A <-> B meaning "hidden common cause").
3. For every objective, estimate the ACE (Average Causal Effect, defined as
   E[Y | do(T=1)] - E[Y | do(T=0)]) of each configuration option, using the
   AIPW estimator (Augmented Inverse Probability Weighting) on that ADMG.
4. Keep the top-k options by |ACE|, union over objectives. Those, and only
   those, are the dimensions the optimizer is later allowed to search.

Step 4 is the entire point of CURE. It is why CURE differs from plain
multi-objective Bayesian optimization: same optimizer, smaller search space.

Two deviations from the shipped CURE code, both forced and both documented
-------------------------------------------------------------------------
(a) TREATMENTS ARE BINARIZED. `ananke.estimation.CausalEffect` estimates
    E[Y(1)] - E[Y(0)] and fits the treatment model with a binomial GLM
    (`_aipw` calls `model_binary(data, formula_T)`). It is defined for binary
    treatments only. CURE passes continuous option columns to it and wraps the
    call in a bare `except: continue`, so options whose treatment model fails
    are silently dropped from the ranking. Here each option is split at its
    Campaign A median into low/high, so the ACE is "effect of moving this knob
    from its low half to its high half" -- a well-defined quantity that the
    estimator can actually compute. Report it that way in the paper.

(b) A CONTEXT LAYER IS ADDED. CURE's campaigns hold the mission fixed, so their
    schema has no context variable. Yours varies. Context enters as an
    exogenous pre-treatment layer (nothing points into it), so it is adjusted
    for rather than treated as an outcome. See cure_graph_spec.py.

Install
-------
    pip install causal-learn "ananke-causal==0.5.0" "pandas<2.0" statsmodels

If `ananke` will not install (it needs pandas < 2.0 because it calls the
removed `DataFrame.iteritems`), the script falls back to a self-contained AIPW
implementation and says so in the output. The fallback is a standard
doubly-robust estimator; it is not bit-identical to ananke's, so state which
one you used.

Usage
-----
    python3 cure_causal_reduction.py \
        --data rct_data_campaign_a/rct_results.csv \
        --arm tucked \
        --out-dir baselines/cure \
        --top-k 3

    # same, but let CURE tune the arm too (arm becomes a searchable option)
    python3 cure_causal_reduction.py --data ... --arm both --include-arm

Outputs
-------
    <out-dir>/cure_reduction_<arm>.json     the reduced option set + all ACEs
    <out-dir>/cure_graph_<arm>.json         the learned ADMG edges
"""

import os
import sys
import json
import argparse
import logging
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cure_graph_specification as spec

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cure_reduction")


# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------
def _try_import_causallearn():
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from causallearn.graph.GraphNode import GraphNode
        return fci, BackgroundKnowledge, GraphNode
    except Exception as exc:
        logger.warning("causal-learn unavailable (%s). "
                       "--graph-mode learned and hybrid will not work.", exc)
        return None, None, None


_ANANKE_CACHE = None


def _try_import_ananke():
    """Import ananke once and remember the outcome, so a missing install warns
    a single time instead of once per treatment-outcome pair."""
    global _ANANKE_CACHE
    if _ANANKE_CACHE is not None:
        return _ANANKE_CACHE
    try:
        # ananke 0.5.0 calls DataFrame.iteritems, removed in pandas 2.0.
        if not hasattr(pd.DataFrame, "iteritems"):
            pd.DataFrame.iteritems = pd.DataFrame.items
        from ananke.graphs import ADMG
        from ananke.estimation import CausalEffect
        _ANANKE_CACHE = (ADMG, CausalEffect)
    except Exception as exc:
        logger.warning("ananke unavailable (%s). Using the built-in AIPW "
                       "estimator instead; say so in the paper.", exc)
        _ANANKE_CACHE = (None, None)
    return _ANANKE_CACHE


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def check_columns(df, needed, label):
    """Fail loudly and specifically rather than dropping columns in silence."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.error("Missing %s column(s) in the data: %s", label, missing)
        logger.error("Columns actually present: %s", sorted(df.columns.tolist()))
        raise SystemExit(
            f"\nCannot proceed. Edit the {label.upper()} list in "
            f"cure_graph_spec.py to match your Campaign A schema, or add the "
            f"missing columns to the collector.\n"
        )


def binarize(series: pd.Series, name: str):
    """Split a treatment at its median into 0 (low) and 1 (high).

    Already-binary columns pass through unchanged. Returns (values, cut_info).
    """
    vals = pd.to_numeric(series, errors="coerce")
    uniq = sorted(v for v in vals.dropna().unique())
    if len(uniq) <= 2:
        lo = uniq[0] if uniq else 0.0
        hi = uniq[-1] if uniq else 1.0
        out = (vals > lo).astype(int)
        return out, {"kind": "already_binary", "low": float(lo), "high": float(hi)}

    cut = float(np.nanmedian(vals))
    out = (vals > cut).astype(int)
    lo_mean = float(np.nanmean(vals[vals <= cut]))
    hi_mean = float(np.nanmean(vals[vals > cut]))
    return out, {"kind": "median_split", "cut": cut,
                 "low_mean": lo_mean, "high_mean": hi_mean}


def prepare_frame(df, include_arm, arm_filter):
    """Subset to one arm stratum (or keep both) and build the model frame."""
    fp_col = spec.ARM_OPTION
    if fp_col in df.columns:
        arm_is_carry = df[fp_col].apply(
            lambda v: 1.0 if ("carry" in str(v).lower()
                              or str(v).strip() in ("1.0", "1")
                              or "0.698" in str(v)) else 0.0)
    else:
        arm_is_carry = pd.Series(np.zeros(len(df)), index=df.index)
        if include_arm or arm_filter != "both":
            raise SystemExit(f"Column {fp_col} not found; cannot stratify by arm.")

    if arm_filter == "tucked":
        sub = df[arm_is_carry == 0.0].copy()
    elif arm_filter == "carry":
        sub = df[arm_is_carry == 1.0].copy()
    else:
        sub = df.copy()
    sub["_arm_is_carry"] = arm_is_carry.reindex(sub.index)

    opt_cols = list(spec.OPTIONS)
    if include_arm:
        opt_cols = opt_cols + [fp_col]

    check_columns(sub, spec.CONTEXT, "context")
    check_columns(sub, spec.OPTIONS, "options")
    check_columns(sub, spec.METRICS, "metrics")
    check_columns(sub, spec.OBJECTIVES, "objectives")

    frame = pd.DataFrame(index=sub.index)
    cuts = {}

    for c in spec.CONTEXT:
        frame[c] = pd.to_numeric(sub[c], errors="coerce")

    for c in opt_cols:
        raw = sub["_arm_is_carry"] if c == fp_col else sub[c]
        frame[c], cuts[c] = binarize(raw, c)

    for c in spec.METRICS:
        frame[c] = pd.to_numeric(sub[c], errors="coerce")

    for c in spec.OBJECTIVES:
        frame[c] = pd.to_numeric(sub[c], errors="coerce")

    before = len(frame)
    frame = frame.dropna()
    if len(frame) < before:
        logger.info("Dropped %d rows with missing values (%d remain).",
                    before - len(frame), len(frame))
    return frame, opt_cols, cuts


# ---------------------------------------------------------------------------
# Stage 1: structure
# ---------------------------------------------------------------------------
def learn_graph_fci(frame, columns, tabu, required, alpha, verbose):
    """FCI with forbidden and required background knowledge.

    Returns (di_edges, bi_edges). Unlike CURE's implementation this passes
    node_names to fci() and reads edge endpoints by name, so nothing depends on
    parsing "X7" back into a column index.
    """
    fci, BackgroundKnowledge, GraphNode = _try_import_causallearn()
    if fci is None:
        raise SystemExit("causal-learn is required for --graph-mode learned/hybrid.")

    data = frame[columns].to_numpy(dtype=float)
    name_to_node = {c: GraphNode(c) for c in columns}

    bk = BackgroundKnowledge()
    for a, b in tabu:
        if a in name_to_node and b in name_to_node:
            bk.add_forbidden_by_node(name_to_node[a], name_to_node[b])
    for a, b in required:
        if a in name_to_node and b in name_to_node:
            bk.add_required_by_node(name_to_node[a], name_to_node[b])

    logger.info("Running FCI on %d rows x %d variables (alpha=%.3f, "
                "%d forbidden edges, %d required edges)...",
                data.shape[0], data.shape[1], alpha, len(tabu), len(required))
    try:
        G, edges = fci(data, "fisherz", alpha, depth=3, verbose=verbose,
                       background_knowledge=bk, show_progress=False,
                       node_names=columns)
    except Exception as exc:
        logger.warning("FCI with background knowledge failed (%s); "
                       "retrying unconstrained.", exc)
        G, edges = fci(data, "fisherz", alpha, depth=3, verbose=verbose,
                       show_progress=False, node_names=columns)

    return resolve_edges(edges, columns, tabu)


def resolve_edges(edges, columns, tabu):
    """Turn FCI's PAG edges into ADMG directed / bidirected edge lists.

    Same resolution policy as CURE: a partially directed edge (o->) and an
    undirected edge (o-o) are both promoted to a directed edge; a bidirected
    edge (<->) stays bidirected. Forbidden edges are removed afterwards.
    """
    tabu_set = set(tabu)
    di, bi = set(), set()

    for e in edges:
        s = str(e)
        parts = s.split()
        if len(parts) < 3:
            continue
        a, mark, b = parts[0], parts[1], parts[2]
        a = _clean_node(a, columns)
        b = _clean_node(b, columns)
        if a is None or b is None or a == b:
            continue

        if "<->" in mark:
            if (a, b) not in tabu_set and (b, a) not in tabu_set:
                bi.add(tuple(sorted((a, b))))
        elif "-->" in mark or "o->" in mark:
            if (a, b) not in tabu_set:
                di.add((a, b))
        elif "o-o" in mark or "---" in mark:
            if (a, b) not in tabu_set:
                di.add((a, b))
            elif (b, a) not in tabu_set:
                di.add((b, a))

    di = [e for e in di if e not in tabu_set]
    return sorted(di), sorted(bi)


def _clean_node(token, columns):
    token = token.strip()
    if token in columns:
        return token
    if token.startswith("X") and token[1:].isdigit():
        idx = int(token[1:]) - 1
        if 0 <= idx < len(columns):
            return columns[idx]
    return None


def graph_from_spec(columns, tabu):
    """Use the hand-written abstract graph from cure_graph_spec.py."""
    tabu_set = set(tabu)
    di = [(a, b) for a, b in spec.ABSTRACT_GRAPH_DI
          if a in columns and b in columns and (a, b) not in tabu_set]
    bi = [tuple(sorted((a, b))) for a, b in spec.ABSTRACT_GRAPH_BI
          if a in columns and b in columns]
    if not di:
        raise SystemExit(
            "--graph-mode given was requested but ABSTRACT_GRAPH_DI in "
            "cure_graph_spec.py is empty. Fill it in, or use "
            "--graph-mode learned.")
    return sorted(set(di)), sorted(set(bi))


def ensure_connected_to_targets(di, options, metrics, objectives):
    """Guarantee every option has at least one outgoing edge into a metric.

    FCI can leave an option isolated when the campaign is small. An isolated
    option gets an undefined ACE and would drop out of the ranking for a reason
    that has nothing to do with its effect size. Adding a single edge into the
    metric layer keeps it in the comparison. Every such addition is logged and
    recorded in the output JSON so the graph you report is the graph you used.
    """
    added = []
    have_out = {a for a, _ in di}
    for opt in options:
        if opt not in have_out and metrics:
            di.append((opt, metrics[0]))
            added.append((opt, metrics[0]))
    for met in metrics:
        if not any(a == met for a, _ in di) and objectives:
            di.append((met, objectives[0]))
            added.append((met, objectives[0]))
    if added:
        logger.warning("Added %d fallback edge(s) so every option reaches an "
                       "outcome: %s", len(added), added)
    return sorted(set(di)), added


# ---------------------------------------------------------------------------
# Stage 2: average causal effects
# ---------------------------------------------------------------------------
def ace_ananke(frame, di, bi, columns, treatment, outcome, n_boot, alpha):
    ADMG, CausalEffect = _try_import_ananke()
    if ADMG is None:
        return None
    try:
        G = ADMG(columns, di_edges=di, bi_edges=bi)
        obj = CausalEffect(graph=G, treatment=treatment, outcome=outcome)
        out = obj.compute_effect(frame, "aipw", n_bootstraps=n_boot, alpha=alpha)
        if isinstance(out, tuple):
            ace, ql, qu = out
        else:
            ace, ql, qu = float(out), float("nan"), float("nan")
        if not np.isfinite(ace):
            return None
        return {"ace": float(ace), "ci_low": float(ql), "ci_high": float(qu),
                "estimator": "ananke-aipw"}
    except Exception as exc:
        logger.debug("ananke AIPW failed for %s -> %s: %s",
                     treatment, outcome, exc)
        return None


def ace_fallback(frame, treatment, outcome, adjust, n_boot, alpha, rng):
    """Self-contained AIPW (doubly-robust) estimator.

    For binary treatment T, outcome Y, adjustment set X:
        e(X)   = P(T=1 | X)                      logistic
        mu_t(X)= E[Y | T=t, X]                   ridge (or logistic if Y binary)
        ACE    = mean( mu1 - mu0
                       + T (Y - mu1)/e
                       - (1-T)(Y - mu0)/(1-e) )
    Doubly robust: consistent if either the treatment model or the outcome
    models are right. Propensities are clipped to [0.05, 0.95] so a single
    near-deterministic row cannot dominate the average.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    T = frame[treatment].to_numpy(dtype=float)
    Y = frame[outcome].to_numpy(dtype=float)
    X = frame[adjust].to_numpy(dtype=float) if adjust else np.zeros((len(T), 1))
    y_binary = set(np.unique(Y)).issubset({0.0, 1.0})

    def point(idx):
        Ti, Yi, Xi = T[idx], Y[idx], X[idx]
        if Ti.sum() < 5 or (1 - Ti).sum() < 5:
            return np.nan
        try:
            ps = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, C=1.0))
            ps.fit(Xi, Ti)
            e = np.clip(ps.predict_proba(Xi)[:, 1], 0.05, 0.95)
        except Exception:
            e = np.full(len(Ti), float(Ti.mean()))
            e = np.clip(e, 0.05, 0.95)

        mu = {}
        for t in (0, 1):
            m = Ti == t
            if m.sum() < 5:
                return np.nan
            try:
                if y_binary:
                    mdl = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000))
                    mdl.fit(Xi[m], Yi[m])
                    mu[t] = mdl.predict_proba(Xi)[:, 1]
                else:
                    mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    mdl.fit(Xi[m], Yi[m])
                    mu[t] = mdl.predict(Xi)
            except Exception:
                mu[t] = np.full(len(Ti), float(Yi[m].mean()))

        psi = (mu[1] - mu[0]
               + Ti * (Yi - mu[1]) / e
               - (1 - Ti) * (Yi - mu[0]) / (1 - e))
        return float(np.mean(psi))

    all_idx = np.arange(len(T))
    est = point(all_idx)
    if not np.isfinite(est):
        return None

    boots = []
    for _ in range(max(0, n_boot)):
        idx = rng.integers(0, len(T), len(T))
        b = point(idx)
        if np.isfinite(b):
            boots.append(b)
    if boots:
        ql = float(np.quantile(boots, alpha / 2))
        qu = float(np.quantile(boots, 1 - alpha / 2))
    else:
        ql = qu = float("nan")
    return {"ace": est, "ci_low": ql, "ci_high": qu, "estimator": "aipw-fallback"}


def compute_reduction(frame, di, bi, columns, opt_cols, targets,
                      top_k, n_boot, alpha, seed, prefer_ananke):
    """Rank options by |ACE| on each target; keep the top-k union."""
    rng = np.random.default_rng(seed)
    adjust = [c for c in spec.CONTEXT if c in frame.columns]
    per_target = {}
    all_rows = []

    for target in targets:
        scored = []
        for opt in opt_cols:
            res = None
            if prefer_ananke:
                res = ace_ananke(frame, di, bi, columns, opt, target,
                                 n_boot, alpha)
            if res is None:
                res = ace_fallback(frame, opt, target, adjust,
                                   n_boot, alpha, rng)
            if res is None:
                logger.warning("No ACE for %s -> %s; excluded from ranking.",
                               opt, target)
                continue
            row = dict(option=opt, target=target, **res)
            row["abs_ace"] = abs(res["ace"])
            ci_lo, ci_hi = res["ci_low"], res["ci_high"]
            row["ci_excludes_zero"] = bool(
                np.isfinite(ci_lo) and np.isfinite(ci_hi)
                and (ci_lo > 0 or ci_hi < 0))
            scored.append(row)
            all_rows.append(row)

        scored.sort(key=lambda r: r["abs_ace"], reverse=True)
        per_target[target] = scored[:top_k]

        logger.info("--- ACE on %s ---", target)
        for r in scored:
            mark = "*" if r in per_target[target] else " "
            logger.info("  %s %-62s ACE=%+.4f  [%+.4f, %+.4f] %s",
                        mark, r["option"].split("__")[-1], r["ace"],
                        r["ci_low"], r["ci_high"],
                        "significant" if r["ci_excludes_zero"] else "")

    reduced = sorted({r["option"] for rows in per_target.values() for r in rows})
    return reduced, per_target, all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="CURE stage 1+2: causal structure learning and reduction.")
    p.add_argument("--data", required=True,
                   help="Campaign A CSV (rct_results.csv).")
    p.add_argument("--arm", default="tucked",
                   choices=["tucked", "carry", "both"],
                   help="Which arm stratum to fit. 'both' pools the strata; "
                        "use it together with --include-arm.")
    p.add_argument("--include-arm", action="store_true",
                   help="Treat arm state as a tunable option, so CURE gets the "
                        "chance to discover the arm effect for itself.")
    p.add_argument("--graph-mode", default="learned",
                   choices=["learned", "given", "hybrid"],
                   help="learned: FCI + tabu edges, what CURE does. "
                        "given: use ABSTRACT_GRAPH_DI from cure_graph_spec.py. "
                        "hybrid: FCI with your graph's edges forced in.")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="FCI independence-test level. CURE uses 0.2.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Options kept per objective. CURE's default is 5, "
                        "which would keep your whole space; 3 of 6-7 is the "
                        "smallest choice that still reduces anything.")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Outcomes to rank against. Default: the objectives.")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--ci-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ananke", action="store_true",
                   help="Skip ananke and use the built-in AIPW estimator.")
    p.add_argument("--out-dir", default="baselines/cure")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Data file not found: {args.data}")
    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows from %s", len(df), args.data)

    frame, opt_cols, cuts = prepare_frame(df, args.include_arm, args.arm)
    if len(frame) < 50:
        logger.warning("Only %d usable rows. FCI and AIPW will both be noisy.",
                       len(frame))

    columns = (list(spec.CONTEXT) + list(opt_cols)
               + list(spec.METRICS) + list(spec.OBJECTIVES))
    frame = frame[columns]

    tabu = spec.tabu_edges(include_arm=args.include_arm)
    required = spec.ABSTRACT_GRAPH_DI if args.graph_mode == "hybrid" else []

    if args.graph_mode == "given":
        di, bi = graph_from_spec(columns, tabu)
        logger.info("Using the abstract graph from cure_graph_spec.py.")
    else:
        di, bi = learn_graph_fci(frame, columns, tabu, required,
                                 args.alpha, args.verbose)
        logger.info("FCI returned %d directed and %d bidirected edges.",
                    len(di), len(bi))

    di, added = ensure_connected_to_targets(di, opt_cols, spec.METRICS,
                                            spec.OBJECTIVES)

    targets = args.targets or list(spec.OBJECTIVES)
    reduced, per_target, all_rows = compute_reduction(
        frame, di, bi, columns, opt_cols, targets,
        args.top_k, args.bootstrap, args.ci_alpha, args.seed,
        prefer_ananke=not args.no_ananke)

    pinned = {c: float(np.median(pd.to_numeric(df[c], errors="coerce").dropna()))
              for c in spec.OPTIONS if c in df.columns and c not in reduced}

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.arm + ("_witharm" if args.include_arm else "")

    graph_path = os.path.join(args.out_dir, f"cure_graph_{tag}.json")
    with open(graph_path, "w") as f:
        json.dump({"columns": columns,
                   "di_edges": [list(e) for e in di],
                   "bi_edges": [list(e) for e in bi],
                   "fallback_edges_added": [list(e) for e in added],
                   "graph_mode": args.graph_mode,
                   "alpha": args.alpha}, f, indent=2)

    red_path = os.path.join(args.out_dir, f"cure_reduction_{tag}.json")
    with open(red_path, "w") as f:
        json.dump({
            "arm": args.arm,
            "include_arm": args.include_arm,
            "n_rows": int(len(frame)),
            "top_k": args.top_k,
            "targets": targets,
            "reduced_options": reduced,
            "pinned_options_at_campaign_median": pinned,
            "per_target_top_k": {t: [r["option"] for r in rows]
                                 for t, rows in per_target.items()},
            "ace_table": all_rows,
            "binarization": cuts,
            "estimator_used": sorted({r["estimator"] for r in all_rows}),
        }, f, indent=2)

    logger.info("")
    logger.info("Reduced option set (%d of %d): %s",
                len(reduced), len(opt_cols),
                [o.split("__")[-1] for o in reduced])
    logger.info("Pinned at Campaign A median: %s",
                {k.split('__')[-1]: round(v, 3) for k, v in pinned.items()})
    logger.info("Wrote %s", red_path)
    logger.info("Wrote %s", graph_path)

    if len(reduced) == len(opt_cols):
        logger.warning("")
        logger.warning("The reduction kept every option, so CURE and MOBO will "
                       "search the same space and produce the same answer. "
                       "Lower --top-k or report the two as one baseline.")


if __name__ == "__main__":
    main()