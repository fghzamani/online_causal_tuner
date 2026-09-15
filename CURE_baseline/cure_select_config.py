#!/usr/bin/env python3
"""
CURE end to end, offline: Campaign A CSV in, one configuration YAML out.

    python3 cure_select_config.py --data rct_results.csv --arm tucked

Read this before you use the output
-----------------------------------
This runs CURE's stages 1 and 2 exactly as published, and stage 3 with one
substitution: candidate configurations are scored by a SURROGATE MODEL fitted
to Campaign A instead of by running the robot.

That substitution is not cosmetic and you have to declare it. CURE's stage 3 is
an optimizer with a live evaluation budget: propose, run the robot, observe,
update, repeat. Every configuration it returns has actually been executed. A
configuration returned by this script has never been executed; it is the
argmin of a model. If the model is wrong somewhere in the space, the optimizer
will walk straight to that spot, because that is what optimizers do.

What is still faithful:
  - FCI structure learning under CURE's tabu-edge constraints
  - AIPW average causal effects on the resulting ADMG
  - top-k causal reduction, which is the step that makes CURE different
    from plain multi-objective Bayesian optimization
  - two objectives, two outcome constraints, Pareto frontier, and a stated
    rule for picking one point off it

What is not:
  - the evaluations

If you have simulator hours before the deadline, use `run_cure_mobo.py`
instead: it is the same procedure with real evaluations. Use this script when
you do not, and say in the paper that CURE was tuned offline against a
surrogate fitted to the same randomized data used to train our model. That
framing is defensible, and it also removes the "you gave CURE less data" reply,
because both methods then see exactly the same campaign.

A note on the optimizer
-----------------------
Bayesian optimization exists to spend a small number of EXPENSIVE evaluations
well. Surrogate evaluations are free. With a surrogate, exhaustive enumeration
over the candidate grid finds the true optimum of the surrogate every time, and
Bayesian optimization can only approximate it. The default here is `--optimizer
grid` for that reason. `--optimizer ax` is available if you would rather keep
the procedural resemblance to CURE, but be aware it is decorative in this mode
and can only do worse.

Outputs (in --out-dir)
----------------------
    cure_config_<arm>.yaml       the configuration, in the shape
                                 run_campaign_b_benchmark.py loads
    cure_selection_<arm>.json    ACE table, reduced set, frontier, the pick,
                                 and the support check for every candidate
    cure_graph_<arm>.json        the learned graph
"""

import os
import sys
import json
import argparse
import logging
import itertools

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cure_graph_specification as spec
import cure_causal_reduction as red

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cure_select")

ARM_OPTION = spec.ARM_OPTION

# Candidate values per option. Taken from PARAM_CANDIDATE_GRID in
# online_tuner_node.py so CURE searches the same action space the tuner does.
# Section 6 of the validity review flagged that they currently do not, and a
# baseline that can reach configurations the tuner cannot is not a baseline.
CANDIDATE_GRID = {
    "param__controller_server__speed_limit_pct": [30.0, 50.0, 70.0, 90.0, 95.0],
    "param__controller_server__FollowPath.vx_std": [0.15, 0.35, 0.40],
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight": [0.5, 6.0],
    "param__controller_server__FollowPath.CostCritic.cost_weight": [1.0, 3.81, 5.0, 12.0, 20.0],
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight": [4.0, 15.0, 30.0, 32.0],
    "param__local_costmap__inflation_layer.inflation_radius": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
}

PARAM_TARGET = {
    "param__controller_server__speed_limit_pct": ("speed_limit", "speed_limit_pct"),
    "param__controller_server__FollowPath.vx_std": ("controller_server", "FollowPath.vx_std"),
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight": ("controller_server", "FollowPath.ConstraintCritic.cost_weight"),
    "param__controller_server__FollowPath.CostCritic.cost_weight": ("controller_server", "FollowPath.CostCritic.cost_weight"),
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight": ("controller_server", "FollowPath.PathAlignCritic.cost_weight"),
    "param__local_costmap__inflation_layer.inflation_radius": ("local_costmap", "inflation_layer.inflation_radius"),
}

STALL_EPS_M = 0.001


def short(name):
    return name.split("__")[-1]


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------
class Surrogate:
    """Predicts campaign outcomes for a configuration, averaged over context.

    Campaign A randomizes the configuration independently of the context R, so
    the average of the fitted model over the empirical distribution of R is an
    estimate of E[Y | do(C=c)] rather than a conditional association. That is
    the quantity a tuner should be optimizing, and it is why this has to be an
    average over the observed contexts and not a prediction at the mean context.

    One model per outcome:
      collision      P(y_h = 1 | do(c))            classifier
      progress       E[probe_progress_m | do(c)]   regressor
      clearance      E[mediator | do(c)]           regressor, if the column exists
      moved          P(progress > eps | do(c))     classifier

    Gradient boosting, because the interactions between arm state, inflation
    and clearance are the whole point and a linear model cannot see them.
    """

    def __init__(self, df, opt_cols, ctx_cols, clearance_col, seed=42,
                 n_context=400):
        from sklearn.ensemble import (GradientBoostingClassifier,
                                      GradientBoostingRegressor)
        self.opt_cols = list(opt_cols)
        self.ctx_cols = list(ctx_cols)
        self.clearance_col = clearance_col

        X = df[self.opt_cols + self.ctx_cols].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        ctx = df[self.ctx_cols].to_numpy(dtype=float)
        if len(ctx) > n_context:
            idx = rng.choice(len(ctx), n_context, replace=False)
            ctx = ctx[idx]
        self.context_grid = ctx
        logger.info("Surrogate marginalizes over %d Campaign A contexts.",
                    len(self.context_grid))

        y_coll = pd.to_numeric(df["y_h"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_prog = pd.to_numeric(df["probe_progress_m"], errors="coerce").fillna(0.0).to_numpy()
        y_moved = (y_prog > STALL_EPS_M).astype(int)

        gb_c = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                    random_state=seed)
        gb_r = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                    random_state=seed)

        self.m_coll = GradientBoostingClassifier(**gb_c).fit(X, y_coll)
        self.m_prog = GradientBoostingRegressor(**gb_r).fit(X, y_prog)
        self.m_moved = (GradientBoostingClassifier(**gb_c).fit(X, y_moved)
                        if len(set(y_moved)) > 1 else None)
        self.m_clear = None
        if clearance_col and clearance_col in df.columns:
            y_clr = pd.to_numeric(df[clearance_col], errors="coerce")
            ok = y_clr.notna().to_numpy()
            if ok.sum() > 50:
                self.m_clear = GradientBoostingRegressor(**gb_r).fit(
                    X[ok], y_clr[ok].to_numpy())

        # Nearest-neighbour distance in the option space, for the support check.
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        self._opt_scaler = StandardScaler().fit(df[self.opt_cols].to_numpy(dtype=float))
        self._nn = NearestNeighbors(n_neighbors=1).fit(
            self._opt_scaler.transform(df[self.opt_cols].to_numpy(dtype=float)))
        d, _ = self._nn.kneighbors(
            self._opt_scaler.transform(df[self.opt_cols].to_numpy(dtype=float)),
            n_neighbors=2)
        self._support_radius = float(np.quantile(d[:, 1], 0.95))

    def _design(self, config):
        opt = np.array([[float(config[c]) for c in self.opt_cols]])
        opt = np.repeat(opt, len(self.context_grid), axis=0)
        return np.hstack([opt, self.context_grid])

    def predict(self, config):
        X = self._design(config)
        out = {
            "collision_prob": float(self.m_coll.predict_proba(X)[:, 1].mean()),
            "progress_m": float(self.m_prog.predict(X).mean()),
        }
        out["moved_rate"] = (float(self.m_moved.predict_proba(X)[:, 1].mean())
                             if self.m_moved is not None else 1.0)
        out["clearance_m"] = (float(self.m_clear.predict(X).mean())
                              if self.m_clear is not None else float("nan"))
        return out

    def in_support(self, config):
        """Is this configuration near anything the campaign actually ran?

        Guards against the failure you already hit once, where a candidate sat
        outside the randomization range and won on extrapolation alone.
        """
        v = self._opt_scaler.transform(
            np.array([[float(config[c]) for c in self.opt_cols]]))
        d, _ = self._nn.kneighbors(v, n_neighbors=1)
        return bool(d[0, 0] <= self._support_radius), float(d[0, 0])


# ---------------------------------------------------------------------------
# Candidate enumeration and selection
# ---------------------------------------------------------------------------
def enumerate_candidates(searchable, pinned, arm_value, grid):
    keys = list(searchable)
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg.update(pinned)
        cfg[ARM_OPTION] = arm_value
        yield cfg


def score_all(candidates, surrogate, args):
    rows = []
    for cfg in candidates:
        pred = surrogate.predict(cfg)
        ok, dist = surrogate.in_support(cfg)
        rows.append({"config": {k: float(v) for k, v in cfg.items()},
                     "in_support": ok, "support_distance": dist, **pred})
    return rows


def pareto_and_pick(rows, args):
    """Feasible non-dominated set on (collision_prob, -progress), then one pick.

    Constraints mirror CURE's --sc (obstacle distance) and --tcr (task
    completion rate). Out-of-support candidates are excluded before anything
    else: a point the campaign never visited has no evidence behind it, and
    including it means reporting an extrapolation as a result.
    """
    pool = [r for r in rows if r["in_support"] or args.allow_out_of_support]
    n_dropped = len(rows) - len(pool)
    if not pool:
        pool = rows
        n_dropped = 0

    feas = [r for r in pool
            if r["moved_rate"] >= args.completion_constraint
            and (not np.isfinite(r["clearance_m"])
                 or r["clearance_m"] >= args.safety_constraint)]
    used_feasible = bool(feas)
    pool2 = feas if feas else pool

    front = []
    for a in pool2:
        dominated = any(
            b is not a
            and b["collision_prob"] <= a["collision_prob"]
            and b["progress_m"] >= a["progress_m"]
            and (b["collision_prob"] < a["collision_prob"]
                 or b["progress_m"] > a["progress_m"])
            for b in pool2)
        if not dominated:
            front.append(a)

    # CURE picks by preference targets on both objectives. Same idea here,
    # normalized so a probability and a distance in metres compare sensibly.
    best, best_d = None, float("inf")
    for r in front:
        d = np.hypot(r["collision_prob"] / max(args.f1_pref, 1e-9),
                     max(0.0, args.f2_pref - r["progress_m"]) / max(args.f2_pref, 1e-9))
        if d < best_d:
            best, best_d = r, float(d)
    return front, best, used_feasible, n_dropped


def optimize_with_ax(searchable, pinned, arm_value, surrogate, args):
    """Procedural match to CURE's stage 3, against the surrogate.

    Included for fidelity. It cannot beat enumeration here, because the
    surrogate is cheap enough to evaluate everywhere.
    """
    from ax.service.ax_client import AxClient, ObjectiveProperties
    params = []
    for name in searchable:
        vals = CANDIDATE_GRID[name]
        params.append({"name": name, "type": "range",
                       "bounds": [float(min(vals)), float(max(vals))],
                       "value_type": "float"})
    client = AxClient(random_seed=args.seed, verbose_logging=False)
    client.create_experiment(
        name=f"cure_surrogate_{args.arm}",
        parameters=params,
        objectives={
            "collision_prob": ObjectiveProperties(minimize=True,
                                                  threshold=float(args.f1_pref)),
            "neg_progress": ObjectiveProperties(minimize=True,
                                                threshold=float(-args.f2_pref)),
        },
        outcome_constraints=[f"moved_rate >= {args.completion_constraint}"],
        choose_generation_strategy_kwargs={"num_initialization_trials":
                                           int(args.init_trials)},
    )
    rows = []
    for _ in range(args.budget):
        cfg, idx = client.get_next_trial()
        full = dict(cfg)
        full.update(pinned)
        full[ARM_OPTION] = arm_value
        pred = surrogate.predict(full)
        ok, dist = surrogate.in_support(full)
        client.complete_trial(trial_index=idx, raw_data={
            "collision_prob": (pred["collision_prob"], 0.0),
            "neg_progress": (-pred["progress_m"], 0.0),
            "moved_rate": (pred["moved_rate"], 0.0)})
        rows.append({"config": {k: float(v) for k, v in full.items()},
                     "in_support": ok, "support_distance": dist, **pred})
    return rows


def export_yaml(config, arm_label, path):
    out = {"controller_server": {}, "local_costmap": {}}
    speed_pct = None
    for key, value in config.items():
        if key not in PARAM_TARGET:
            continue
        node_key, param_name = PARAM_TARGET[key]
        if node_key == "speed_limit":
            speed_pct = round(float(value), 3)
        else:
            out.setdefault(node_key, {})[param_name] = round(float(value), 3)
    if speed_pct is not None:
        # Not a ROS parameter: Nav2 takes it as nav2_msgs/SpeedLimit on
        # /speed_limit. See section 6 of README_CURE_BASELINE.md for the
        # four-line patch apply_all_params needs.
        out["speed_limit_pct"] = speed_pct
    out["local_costmap"]["footprint"] = arm_label
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="CURE end to end, offline: dataset in, configuration out.")
    p.add_argument("--data", required=True, help="Campaign A CSV.")
    p.add_argument("--arm", default="tucked", choices=["tucked", "carry"],
                   help="Arm stratum to fit and to emit a configuration for.")
    p.add_argument("--graph-mode", default="learned",
                   choices=["learned", "given", "hybrid"])
    p.add_argument("--alpha", type=float, default=0.2,
                   help="FCI independence-test level. CURE uses 0.2.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Options kept per objective by the causal reduction.")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--no-ananke", action="store_true")

    p.add_argument("--optimizer", choices=["grid", "ax"], default="grid",
                   help="grid: exhaustive over the candidate grid, exact for "
                        "the surrogate. ax: Bayesian optimization, procedurally "
                        "closer to CURE but strictly worse here.")
    p.add_argument("--budget", type=int, default=60,
                   help="Evaluations, --optimizer ax only.")
    p.add_argument("--init-trials", type=int, default=12)

    p.add_argument("--f1-pref", type=float, default=0.10,
                   help="Target collision probability. CURE's --f1_pref.")
    p.add_argument("--f2-pref", type=float, default=0.60,
                   help="Target expected progress in metres. CURE's --f2_pref.")
    p.add_argument("--safety-constraint", type=float, default=0.25,
                   help="Minimum predicted clearance. CURE's --sc. Ignored if "
                        "no clearance mediator column exists.")
    p.add_argument("--completion-constraint", type=float, default=0.80,
                   help="Minimum predicted rate of moving at all. CURE's --tcr.")
    p.add_argument("--allow-out-of-support", action="store_true",
                   help="Keep candidates the campaign never visited. Off by "
                        "default; turning it on means reporting an "
                        "extrapolation as a result.")

    p.add_argument("--out-dir", default="baselines/cure")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Data file not found: {args.data}")
    df_raw = pd.read_csv(args.data)
    logger.info("Loaded %d rows from %s", len(df_raw), args.data)

    # ---- stages 1 and 2, unchanged from the published method ----------
    frame, opt_cols, cuts = red.prepare_frame(df_raw, include_arm=False,
                                              arm_filter=args.arm)
    if len(frame) < 50:
        logger.warning("Only %d usable rows in the %s stratum. Both the graph "
                       "and the effect estimates will be noisy.",
                       len(frame), args.arm)

    columns = (list(spec.CONTEXT) + list(opt_cols)
               + list(spec.METRICS) + list(spec.OBJECTIVES))
    frame = frame[columns]
    tabu = spec.tabu_edges(include_arm=False)
    required = spec.ABSTRACT_GRAPH_DI if args.graph_mode == "hybrid" else []

    if args.graph_mode == "given":
        di, bi = red.graph_from_spec(columns, tabu)
    else:
        di, bi = red.learn_graph_fci(frame, columns, tabu, required,
                                     args.alpha, args.verbose)
    di, added = red.ensure_connected_to_targets(di, opt_cols, spec.METRICS,
                                                spec.OBJECTIVES)

    targets = list(spec.OBJECTIVES)
    reduced, per_target, ace_rows = red.compute_reduction(
        frame, di, bi, columns, opt_cols, targets, args.top_k,
        args.bootstrap, 0.05, args.seed, prefer_ananke=not args.no_ananke)

    searchable = [o for o in reduced if o in CANDIDATE_GRID]
    pinned = {}
    for c in CANDIDATE_GRID:
        if c not in searchable:
            vals = pd.to_numeric(df_raw[c], errors="coerce").dropna()
            med = float(np.median(vals))
            pinned[c] = min(CANDIDATE_GRID[c], key=lambda v: abs(v - med))

    logger.info("")
    logger.info("Causal reduction keeps %d of %d options: %s",
                len(searchable), len(CANDIDATE_GRID),
                [short(s) for s in searchable])
    logger.info("Pinned at the nearest grid value to the campaign median: %s",
                {short(k): v for k, v in pinned.items()})
    if len(searchable) == len(CANDIDATE_GRID):
        logger.warning("Nothing was reduced, so CURE and plain multi-objective "
                       "optimization are the same computation here. Lower "
                       "--top-k, or report them as one baseline row.")

    # ---- stage 3, against the surrogate --------------------------------
    df_arm = df_raw.copy()
    if ARM_OPTION in df_arm.columns:
        is_carry = df_arm[ARM_OPTION].apply(
            lambda v: 1.0 if ("carry" in str(v).lower()
                              or str(v).strip() in ("1.0", "1")
                              or "0.698" in str(v)) else 0.0)
        df_arm = df_arm[is_carry == (1.0 if args.arm == "carry" else 0.0)].copy()
    for c in list(CANDIDATE_GRID) + list(spec.CONTEXT):
        df_arm[c] = pd.to_numeric(df_arm[c], errors="coerce")
    df_arm = df_arm.dropna(subset=list(CANDIDATE_GRID) + list(spec.CONTEXT)
                           + ["y_h", "probe_progress_m"])
    logger.info("Surrogate fitted on %d rows from the %s stratum.",
                len(df_arm), args.arm)

    clearance_col = next((m for m in spec.METRICS
                          if "obstacle" in m or "clear" in m), None)
    surrogate = Surrogate(df_arm, list(CANDIDATE_GRID), list(spec.CONTEXT),
                          clearance_col, seed=args.seed)
    if clearance_col is None:
        logger.warning("No clearance mediator column found; the --sc "
                       "constraint is inactive.")

    arm_value = 1.0 if args.arm == "carry" else 0.0
    if args.optimizer == "ax":
        try:
            rows = optimize_with_ax(searchable, pinned, arm_value, surrogate, args)
            optimizer_used = "ax-qnehvi-on-surrogate"
        except Exception as exc:
            logger.error("Ax unavailable (%s); enumerating the grid instead.", exc)
            rows = score_all(enumerate_candidates(searchable, pinned, arm_value,
                                                  CANDIDATE_GRID),
                             surrogate, args)
            optimizer_used = "exhaustive-grid-on-surrogate"
    else:
        cands = list(enumerate_candidates(searchable, pinned, arm_value,
                                          CANDIDATE_GRID))
        logger.info("Enumerating %d candidate configurations.", len(cands))
        rows = score_all(cands, surrogate, args)
        optimizer_used = "exhaustive-grid-on-surrogate"

    front, best, used_feasible, n_dropped = pareto_and_pick(rows, args)
    if n_dropped:
        logger.info("Excluded %d candidate(s) outside the Campaign A support.",
                    n_dropped)
    if not used_feasible:
        logger.warning("No candidate met the constraints (moved rate >= %.2f, "
                       "clearance >= %.2f m). The frontier is over infeasible "
                       "points; report that rather than hiding it.",
                       args.completion_constraint, args.safety_constraint)

    os.makedirs(args.out_dir, exist_ok=True)
    yaml_path = os.path.join(args.out_dir, f"cure_config_{args.arm}.yaml")
    sel_path = os.path.join(args.out_dir, f"cure_selection_{args.arm}.json")
    graph_path = os.path.join(args.out_dir, f"cure_graph_{args.arm}.json")

    cfg_out = export_yaml(best["config"], args.arm, yaml_path)

    with open(graph_path, "w") as f:
        json.dump({"columns": columns,
                   "di_edges": [list(e) for e in di],
                   "bi_edges": [list(e) for e in bi],
                   "fallback_edges_added": [list(e) for e in added],
                   "graph_mode": args.graph_mode, "alpha": args.alpha}, f, indent=2)

    with open(sel_path, "w") as f:
        json.dump({
            "arm": args.arm,
            "method": "CURE stages 1-2 as published; stage 3 optimized against "
                      "a surrogate fitted to Campaign A, not against live "
                      "simulator evaluations",
            "optimizer": optimizer_used,
            "n_rows_stratum": int(len(df_arm)),
            "reduced_options": searchable,
            "pinned_options": pinned,
            "per_target_top_k": {t: [r["option"] for r in rows_]
                                 for t, rows_ in per_target.items()},
            "ace_table": ace_rows,
            "binarization": cuts,
            "constraints": {
                "moved_rate_min": args.completion_constraint,
                "clearance_min_m": args.safety_constraint,
                "clearance_column": clearance_col},
            "any_feasible": used_feasible,
            "n_candidates_scored": len(rows),
            "n_dropped_out_of_support": n_dropped,
            "pareto_front": front,
            "selected": best,
            "exported_config": cfg_out,
        }, f, indent=2)

    logger.info("")
    logger.info("Selected configuration for arm=%s:", args.arm)
    for k, v in sorted(best["config"].items()):
        logger.info("   %-42s %s", short(k), v)
    logger.info("Predicted: collision %.3f | progress %.3f m | moved %.2f | "
                "clearance %s",
                best["collision_prob"], best["progress_m"], best["moved_rate"],
                ("%.3f m" % best["clearance_m"])
                if np.isfinite(best["clearance_m"]) else "n/a")
    logger.info("Frontier size %d, in support: %s",
                len(front), best["in_support"])
    logger.info("Wrote %s", yaml_path)
    logger.info("Wrote %s", sel_path)
    logger.info("")
    logger.info("These numbers are surrogate predictions, not measurements. "
                "The configuration has not been executed. Benchmark it before "
                "reporting anything about it.")


if __name__ == "__main__":
    main()