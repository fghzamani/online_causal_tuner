#!/usr/bin/env python3
"""
Offline Causal Model Training Pipeline for the Online Causal Tuner.

Trains THREE models from Campaign A RCT data:

  1. Safety      P(Y^H = 1 | do(C=c), R=r)          logistic
  2. Feasibility P(stall  | do(C=c), R=r)           logistic   <-- NEW (hurdle part 1)
  3. Speed       E[J^H | moves, do(C=c), R=r]       ridge      <-- NEW (hurdle part 2)

Expected progress is then  E[J^H] = (1 - P(stall)) * E[J^H | moves].

Why the hurdle split: 22% of Campaign A probes have EXACTLY zero progress
(carry ~38%, tucked ~2.4%). A single Ridge fits a mean through a spike-at-zero
mixture. More importantly, the stall channel is the only one that varies
monotonically with openness, which is what the runtime arm policy needs.

Hyperparameters are selected by REGION-HELD-OUT cross-validation (GroupKFold on
map quadrant), not by in-sample fit. Measured effect: worst-region Brier skill
goes from -0.070 (C=1.0) to +0.006 (C~0.005).

Usage:
  python3 train_causal_models.py \
      --data-path ./rct_data_campaign_a/rct_results.csv \
      --output-dir ./models \
      --bootstrap 200
"""

import os
import sys
import argparse
import logging
import csv
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import brier_score_loss

# Ensure package root is importable so pickled classes resolve by module path
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_causal_models")

ARTIFACT_VERSION = 3  # bump so a stale pickle fails loudly at load time

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# r_vis and r_grad are DROPPED. Measured justification:
#   r_vis  : 6 distinct values, median 1.0 (saturates - _compute_r_vis returns
#            the fraction of forward beams occluded, which is ~1 indoors)
#   r_grad : 15 distinct values, median 0.0 (NaN when scipy missing or the robot
#            pose is unresolvable in the costmap frame)
# Dropping them costs nothing in transfer (region-held-out skill +0.0489 ->
# +0.0493) and removes two coefficients fitted to noise.
RISK_FEATURE_KEYS = [
    "risk__r_min",
    "risk__r_width",
    "risk__r_ttc",
    "risk__r_dens",
    "risk__r_clear",
    "risk__r_curve",
]

PARAM_KEYS = [
    "param__controller_server__speed_limit_pct",
    "param__controller_server__FollowPath.vx_std",
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight",
    "param__controller_server__FollowPath.CostCritic.cost_weight",
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight",
    "param__local_costmap__inflation_layer.inflation_radius",
    "param__local_costmap__footprint",
]

FOOTPRINT_KEY = "param__local_costmap__footprint"

# Progress below this counts as "did not move at all"
STALL_EPS_M = 0.001

# Objective weights matching node parameters.
RISK_LAMBDA = 10.0
STALL_MU = 2.0
# Calibrated, not chosen: omega = RISK_LAMBDA * DELTA / J_med, so the tuner
# accepts at most DELTA extra collision probability in exchange for carrying.
# DELTA is the design tolerance; J_med is the Campaign A median expected
# progress per H-second window. Recompute J_med if the campaign changes.
CARRY_RISK_TOLERANCE = 0.05     # DELTA
PROGRESS_MEDIAN_M = 0.60        # J_med -- measured Campaign A median (0.588 m)
PAYLOAD_OMEGA = RISK_LAMBDA * CARRY_RISK_TOLERANCE / PROGRESS_MEDIAN_M

# Coarse action set for policy learning and off-policy evaluation.
POLICY_ACTION_SET = {
    "param__local_costmap__footprint": [0.0, 1.0],               # tucked / carry
    "param__local_costmap__inflation_layer.inflation_radius": [0.15, 0.30, 0.45, 0.60],
    "param__controller_server__speed_limit_pct": [30.0, 60.0, 90.0],
}


# ---------------------------------------------------------------------------
# Interaction transformer
# ---------------------------------------------------------------------------
class CausalInteractionTransformer(BaseEstimator, TransformerMixin):
    """Append mean-centred configuration x context product terms.

    Centring matters for interpretation: each configuration main effect then
    reads as "effect at AVERAGE context" rather than "effect at R = 0", where
    r_min = 0 would mean an obstacle touching the robot.

    Column lookup is EXACT (not substring) so a knob can never be silently
    matched to the wrong column.
    """

    __module__ = "online_causal_tuner.train_causal_models"

    # (config_column, context_column) pairs
    INTERACTIONS = [
        ("param__controller_server__speed_limit_pct", "risk__r_min"),
        ("param__controller_server__speed_limit_pct", "risk__r_ttc"),
        ("param__controller_server__speed_limit_pct", "risk__r_width"),
        ("param__controller_server__speed_limit_pct", "risk__r_curve"),
        ("param__controller_server__speed_limit_pct", "risk__r_clear"),
        ("param__controller_server__FollowPath.vx_std", "risk__r_clear"),
        ("param__controller_server__FollowPath.vx_std", "risk__r_curve"),
        ("param__controller_server__FollowPath.ConstraintCritic.cost_weight", "risk__r_min"),
        ("param__controller_server__FollowPath.CostCritic.cost_weight", "risk__r_min"),
        ("param__controller_server__FollowPath.CostCritic.cost_weight", "risk__r_dens"),
        ("param__controller_server__FollowPath.PathAlignCritic.cost_weight", "risk__r_curve"),
        ("param__controller_server__FollowPath.PathAlignCritic.cost_weight", "risk__r_width"),
        ("param__local_costmap__inflation_layer.inflation_radius", "risk__r_width"),
        ("param__local_costmap__inflation_layer.inflation_radius", "risk__r_min"),
        (FOOTPRINT_KEY, "risk__r_width"),
        (FOOTPRINT_KEY, "risk__r_min"),
        (FOOTPRINT_KEY, "risk__r_clear"),
    ]

    # configuration x configuration terms, kept SEPARATE from the C x R table
    # so they do not muddy what the context-interaction table claims
    CONFIG_CROSS = [
        ("param__controller_server__speed_limit_pct", FOOTPRINT_KEY),
        ("param__local_costmap__inflation_layer.inflation_radius", FOOTPRINT_KEY),
    ]

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or []
        self.means_ = None

    def _pairs(self):
        return list(self.INTERACTIONS) + list(self.CONFIG_CROSS)

    def interaction_names(self):
        return [f"{a.split('__')[-1]} x {b.split('__')[-1]}" for a, b in self._pairs()]

    def output_names(self):
        return list(self.feature_cols) + self.interaction_names()

    def fit(self, X, y=None):
        self.means_ = np.mean(np.asarray(X, dtype=np.float64), axis=0)
        return self

    def transform(self, X):
        X_mat = np.asarray(X, dtype=np.float64)
        if not self.feature_cols or X_mat.shape[1] == 0:
            return X_mat

        means = self.means_ if self.means_ is not None else np.mean(X_mat, axis=0)
        X_centered = X_mat - means
        idx = {col: i for i, col in enumerate(self.feature_cols)}

        cols = []
        for a, b in self._pairs():
            if a not in idx or b not in idx:
                raise KeyError(
                    f"Interaction ({a}, {b}) references a column not in feature_cols. "
                    "Feature list and interaction list are out of sync."
                )
            cols.append(X_centered[:, idx[a]] * X_centered[:, idx[b]])

        return np.hstack([X_mat, np.column_stack(cols)])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _flag_true(row, key):
    """Explicit validity: anything that is not clearly true is invalid.

    Accepts '1', '1.0', 'true', 't' (case-insensitive).
    """
    val = str(row.get(key, "")).strip().lower()
    return val in ("1", "1.0", "true", "t")


def load_dataset_csv(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    rows, dropped = [], {"treatment_valid": 0, "baseline_valid": 0}
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        clean = (line.replace("\x00", "") for line in f)
        for row in csv.DictReader(clean):
            if not _flag_true(row, "treatment_valid"):
                dropped["treatment_valid"] += 1
                continue
            if not _flag_true(row, "baseline_valid"):
                dropped["baseline_valid"] += 1
                continue
            rows.append(row)

    logger.info("Loaded %d valid rows (dropped %d treatment_valid, %d baseline_valid).",
                len(rows), dropped["treatment_valid"], dropped["baseline_valid"])
    logger.info("NOTE: treatment_valid==0 rows are BASELINE_COLLISION / RUNNER_EXCEPTION "
                "aborts with an empty risk vector and empty outcome, so dropping them "
                "does not condition on a post-treatment variable.")
    return rows


def _to_float(val, default=None):
    try:
        return float(val) if str(val).strip() != "" else default
    except (TypeError, ValueError):
        return default


def extract_features(rows):
    """Build X, targets and grouping. Rows with missing targets are DROPPED,
    never defaulted -- a missing collision label must not become 'safe'."""
    if not rows:
        raise ValueError("Dataset rows list is empty.")

    risk_cols = list(RISK_FEATURE_KEYS)
    param_cols = list(PARAM_KEYS)
    feature_cols = risk_cols + param_cols

    X, y_safety, y_progress, start_xy, kept_rows = [], [], [], [], []
    skipped = {"feature": 0, "label": 0, "progress": 0, "pose": 0}

    param_defaults = {
        "param__controller_server__speed_limit_pct": 100.0,
        "param__controller_server__FollowPath.vx_std": 0.25,
        "param__controller_server__FollowPath.ConstraintCritic.cost_weight": 10.0,
        "param__controller_server__FollowPath.CostCritic.cost_weight": 3.81,
        "param__controller_server__FollowPath.PathAlignCritic.cost_weight": 32.0,
        "param__local_costmap__inflation_layer.inflation_radius": 0.55,
        "param__local_costmap__footprint": 0.0,
    }

    for r in rows:
        feat, bad = [], False
        for col in feature_cols:
            raw = r.get(col, "")
            if col == FOOTPRINT_KEY:
                s_raw = str(raw).strip()
                if s_raw in ("carry", "1.0", "1"):
                    feat.append(1.0)
                elif s_raw in ("tucked", "0.0", "0"):
                    feat.append(0.0)
                elif col in param_defaults:
                    feat.append(param_defaults[col])
                else:
                    bad = True
                    break
            elif col == "param__controller_server__speed_limit_pct" and str(raw).strip() == "":
                vx_raw = r.get("param__controller_server__FollowPath.vx_max", "")
                v = _to_float(vx_raw)
                if v is not None:
                    feat.append((v / 0.55) * 100.0)
                elif col in param_defaults:
                    feat.append(param_defaults[col])
                else:
                    bad = True
                    break
            else:
                v = _to_float(raw)
                if v is None or not np.isfinite(v):
                    if col in param_defaults:
                        v = param_defaults[col]
                    else:
                        bad = True
                        break
                feat.append(v)
        if bad:
            skipped["feature"] += 1
            continue

        ys = _to_float(r.get("y_h", ""))
        if ys is None:
            skipped["label"] += 1
            continue

        yp = _to_float(r.get("probe_progress_m", ""))
        if yp is None:
            skipped["progress"] += 1
            continue

        sx, sy = _to_float(r.get("start_x", "")), _to_float(r.get("start_y", ""))
        if sx is None or sy is None:
            skipped["pose"] += 1
            continue

        X.append(feat)
        y_safety.append(int(round(ys)))
        y_progress.append(yp)
        start_xy.append((sx, sy))
        kept_rows.append(r)

    logger.info("Feature extraction: kept %d rows; skipped %s", len(X), skipped)
    if not X:
        raise ValueError("No usable rows after extraction.")

    X = np.asarray(X, dtype=np.float64)
    y_safety = np.asarray(y_safety, dtype=np.int32)
    y_progress = np.asarray(y_progress, dtype=np.float64)
    start_xy = np.asarray(start_xy, dtype=np.float64)

    # Spatial groups: map quadrants from the episode START pose. Held-out
    # evaluation splits by REGION, not random rows, to avoid spatial leakage.
    groups = ((start_xy[:, 0] > np.median(start_xy[:, 0])).astype(int) * 2
              + (start_xy[:, 1] > np.median(start_xy[:, 1])).astype(int))

    return X, y_safety, y_progress, groups, risk_cols, param_cols, feature_cols, kept_rows


# ---------------------------------------------------------------------------
# Model fitting with region-held-out hyperparameter selection
# ---------------------------------------------------------------------------
class ConstantProbClassifier(BaseEstimator):
    """Fallback classifier for single-class or rare target folds."""
    __module__ = "online_causal_tuner.train_causal_models"

    def __init__(self, p=0.0):
        self.p = float(p)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        if y is not None and len(y) > 0:
            self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        p_val = max(0.0, min(1.0, self.p))
        return np.column_stack([np.full(n, 1.0 - p_val), np.full(n, p_val)])

    def predict(self, X):
        return np.full(len(X), int(self.p >= 0.5))


def _pipeline(feature_cols, estimator):
    return Pipeline([
        ("interaction", CausalInteractionTransformer(feature_cols=feature_cols)),
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ])


def _group_skill(X, y, groups, feature_cols, estimator_factory, n_splits=4):
    """Mean and worst Brier SKILL across region-held-out folds.

    Skill is normalised by each fold's own base rate. Raw Brier is NOT
    comparable across regions here because collision base rates differ by 2x
    between quadrants (0.9% to 2.2%), so selecting on raw Brier silently
    favours whichever fold is easiest.
    """
    skills = []
    for g in sorted(set(groups.tolist())):
        tr, te = groups != g, groups == g
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            skills.append(0.0)
            continue
        model = _pipeline(feature_cols, estimator_factory()).fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        b = brier_score_loss(y[te], p)
        base = brier_score_loss(y[te], np.full(int(te.sum()), y[tr].mean()))
        skills.append(1.0 - b / base if base > 0 else 0.0)
    return float(np.mean(skills)), float(np.min(skills)) if skills else 0.0


def fit_classifier(X, y, groups, feature_cols, name, criterion="worst"):
    """Logistic regression; C selected by region-held-out Brier SKILL.

    criterion="worst" maximises the WORST region's skill, which is what
    supports a transfer claim (the model is never worse than the base rate
    anywhere). criterion="mean" maximises average skill instead.
    """
    if len(np.unique(y)) < 2:
        c_val = float(np.mean(y))
        logger.warning("[%s] target has only 1 class (mean=%.4f) in entire dataset! Using ConstantProbClassifier.", name, c_val)
        model = _pipeline(feature_cols, ConstantProbClassifier(p=c_val)).fit(X, y)
        return model, 0.001

    best = None
    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        mean_s, worst_s = _group_skill(
            X, y, groups, feature_cols,
            lambda C=C: LogisticRegression(max_iter=5000, C=C, solver="lbfgs",
                                           random_state=42))
        score = worst_s if criterion == "worst" else mean_s
        logger.info("  [%s] C=%-6s mean skill %+.4f | worst %+.4f", name, C, mean_s, worst_s)
        if best is None or score > best[0]:
            best = (score, C, mean_s, worst_s)
    _, C_best, mean_s, worst_s = best
    logger.info("[%s] selected C=%s (mean skill %+.4f, worst %+.4f, criterion=%s)",
                name, C_best, mean_s, worst_s, criterion)
    try:
        model = _pipeline(feature_cols,
                          LogisticRegression(max_iter=5000, C=C_best, solver="lbfgs",
                                             random_state=42)).fit(X, y)
    except ValueError:
        logger.warning("[%s] LogisticRegression fit failed (likely single class); falling back to ConstantProbClassifier.", name)
        model = _pipeline(feature_cols, ConstantProbClassifier(p=float(y.mean()))).fit(X, y)
    return model, C_best


def fit_regressor(X, y, groups, feature_cols, name, n_splits=4):
    """Ridge; alpha selected by region-held-out MSE (no base-rate issue here)."""
    grid = {"estimator__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    search = GridSearchCV(
        _pipeline(feature_cols, Ridge(random_state=42)),
        param_grid=grid,
        cv=GroupKFold(n_splits=n_splits),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    logger.info("[%s] selected alpha=%s (region-held-out MSE %.5f)",
                name, search.best_params_["estimator__alpha"], -search.best_score_)
    return search.best_estimator_, search.best_params_["estimator__alpha"]


def fit_bootstrap_ensemble(X_mat, y_safe, y_stall, y_prog, moved, feature_cols, C_safety, C_stall, alpha_speed, B=50, seed=42):
    """Fit bootstrap ensembles for safety, stall, and speed models.
    
    Sampling is with replacement at the row level.
    """
    rng = np.random.RandomState(seed)
    n = X_mat.shape[0]
    safety_ensemble = []
    stall_ensemble = []
    speed_ensemble = []
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        
        # Fit safety model
        if len(np.unique(y_safe[idx])) < 2:
            p_safe = _pipeline(feature_cols, ConstantProbClassifier(p=float(y_safe[idx].mean())))
            p_safe.fit(X_mat[idx], y_safe[idx])
        else:
            p_safe = Pipeline([
                ("interaction", CausalInteractionTransformer(feature_cols=feature_cols)),
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression(max_iter=5000, C=C_safety, solver="lbfgs", random_state=42 + b)),
            ])
            p_safe.fit(X_mat[idx], y_safe[idx])
        safety_ensemble.append(p_safe)
        
        # Fit stall model
        if len(np.unique(y_stall[idx])) < 2:
            p_stall = _pipeline(feature_cols, ConstantProbClassifier(p=float(y_stall[idx].mean())))
            p_stall.fit(X_mat[idx], y_stall[idx])
        else:
            p_stall = Pipeline([
                ("interaction", CausalInteractionTransformer(feature_cols=feature_cols)),
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression(max_iter=5000, C=C_stall, solver="lbfgs", random_state=42 + b)),
            ])
            p_stall.fit(X_mat[idx], y_stall[idx])
        stall_ensemble.append(p_stall)
        
        # Fit speed model (only on moving trials of the bootstrap sample)
        sub_idx = idx[moved[idx] == 1]
        p_speed = Pipeline([
            ("interaction", CausalInteractionTransformer(feature_cols=feature_cols)),
            ("scaler", StandardScaler()),
            ("estimator", Ridge(alpha=alpha_speed, random_state=42 + b)),
        ])
        p_speed.fit(X_mat[sub_idx], y_prog[sub_idx])
        speed_ensemble.append(p_speed)
        
    return safety_ensemble, stall_ensemble, speed_ensemble


def fit_support_model(X_mat, feature_cols, risk_cols, param_cols, k=10):
    """Local effective sample size in (C, R) space, for overlap diagnostics.

    Returns a dict the tuner can use to compute, for any candidate (c, r), the
    distance to its k-th nearest Campaign A neighbour. Large distance means the
    causal estimate there is extrapolation, not identification.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_mat)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(X_mat))
    d, _ = nn.kneighbors(sc.transform(X_mat))
    return {
        "scaler": sc,
        "nn": nn,
        "k": k,
        # Reference quantiles of the in-sample k-NN distance, so the runtime
        # threshold is a data-derived quantile rather than a chosen constant.
        "d_q50": float(np.quantile(d[:, -1], 0.50)),
        "d_q95": float(np.quantile(d[:, -1], 0.95)),
        "d_q99": float(np.quantile(d[:, -1], 0.99)),
    }


def assignment_propensity(rows, action_cols, action_levels):
    """e(c) for a randomized design, estimated from realised assignment frequencies.

    Campaign A assigned C independently of R, so e(c | r) = e(c) and the
    empirical marginal is the right estimate.
    """
    import itertools
    from collections import Counter
    cells = list(itertools.product(*[action_levels[c] for c in action_cols]))

    def _snap(row):
        vals = []
        for c in action_cols:
            val_str = str(row.get(c, 0.0)).strip()
            if c == "param__local_costmap__footprint":
                if val_str in ("carry", "1.0", "1"):
                    val = 1.0
                elif val_str in ("tucked", "0.0", "0"):
                    val = 0.0
                else:
                    val = 0.0
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    val = 0.0
            vals.append(min(action_levels[c], key=lambda L: abs(val - L)))
        return tuple(vals)

    counts = Counter(_snap(r) for r in rows)
    n = sum(counts.values())
    e = {cell: counts.get(cell, 0) / n for cell in cells}
    thin = [c for c, v in e.items() if v < 0.01]
    if thin:
        logger.warning(f"{len(thin)} action cells with propensity < 1%: {thin}")
    return e, cells, _snap


def build_dr_scores(X_mat, rows, utility_obs, mu_hat_fn, action_cols,
                    action_levels, feature_cols):
    """Gamma[i, a] = mu_hat(a, R_i) + 1{C_i = a}/e(a) * (U_i - mu_hat(C_i, R_i))

    Doubly robust: consistent if either mu_hat or e is correct. Here e is known
    by design, so the estimator is unbiased regardless of mu_hat -- which is a
    stronger position than most policy-learning applications get to claim.
    """
    if len(rows) != X_mat.shape[0]:
        raise ValueError(
            f"Row/feature misalignment: {len(rows)} rows vs {X_mat.shape[0]} feature "
            f"vectors. Pass the KEPT rows from extract_features, not the raw CSV rows."
        )
    e, cells, snap = assignment_propensity(rows, action_cols, action_levels)
    n, A = len(rows), len(cells)
    Gamma = np.zeros((n, A))
    obs_cell = [snap(r) for r in rows]
    idx_of = {c: j for j, c in enumerate(cells)}

    # mu_hat evaluated at every (row, action): overwrite the action columns of
    # X and re-predict. Everything else (R_i) is held at its observed value.
    col_idx = [feature_cols.index(c) for c in action_cols]
    mu = np.zeros((n, A))
    for j, cell in enumerate(cells):
        Xa = X_mat.copy()
        for k, ci in enumerate(col_idx):
            Xa[:, ci] = cell[k]
        mu[:, j] = mu_hat_fn(Xa)

    for i in range(n):
        j_obs = idx_of[obs_cell[i]]
        resid = utility_obs[i] - mu[i, j_obs]
        Gamma[i, :] = mu[i, :]
        p = max(e[obs_cell[i]], 1e-3)          # clip: an unclipped 1/e explodes
        Gamma[i, j_obs] += resid / p
    return Gamma, cells, mu


def fit_policy_tree(R_mat, Gamma, mu_mat, max_depth=3):
    """Policy tree learning (Wager & Athey / Sun et al.).

    Grow tree structure on low-variance mu_hat predictions; assign each leaf node
    the action with the highest mean DR score Gamma over the rows in that leaf.
    """
    from sklearn.tree import DecisionTreeClassifier
    n = len(R_mat)
    min_leaf = max(20, int(0.05 * n / (2**max_depth)))

    # Grow tree structure on mu_hat best actions
    best_mu = mu_mat.argmax(axis=1)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42)
    tree.fit(R_mat, best_mu)

    # Assign leaf optimal action using mean DR scores
    leaf_ids = tree.apply(R_mat)
    leaf_action = {}
    leaf_stats = {}
    for leaf in np.unique(leaf_ids):
        mask = (leaf_ids == leaf)
        leaf_dr_means = Gamma[mask].mean(axis=0)
        best_a = int(np.argmax(leaf_dr_means))
        leaf_action[int(leaf)] = best_a
        leaf_stats[int(leaf)] = {
            "n": int(mask.sum()),
            "best_action": best_a,
            "mean_dr_value": float(leaf_dr_means[best_a]),
        }

    logger.info("Policy tree: %d leaves, min leaf n=%d", len(leaf_action), min(s["n"] for s in leaf_stats.values()))
    return tree, leaf_action, leaf_stats


def apply_policy_tree(tree, leaf_action, R_mat):
    """Perform leaf lookup on trained policy tree."""
    leaf_ids = tree.apply(R_mat)
    return np.array([leaf_action[int(l)] for l in leaf_ids], dtype=int)


def region_holdout_report(X, y, groups, feature_cols, C):
    """Per-region Brier and skill vs the base rate. This is the generalisation
    number to report in the paper -- do not substitute a random-row split."""
    out, skills = [], []
    for g in sorted(set(groups.tolist())):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2:
            p = np.full(int(te.sum()), float(y[tr].mean()))
        else:
            try:
                model = _pipeline(feature_cols,
                                  LogisticRegression(max_iter=5000, C=C, solver="lbfgs",
                                                     random_state=42)).fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
            except Exception:
                p = np.full(int(te.sum()), float(y[tr].mean()))
        b = brier_score_loss(y[te], p)
        base = brier_score_loss(y[te], np.full(int(te.sum()), y[tr].mean()))
        skill = 1.0 - b / base if base > 0 else float("nan")
        skills.append(skill if not np.isnan(skill) else 0.0)
        out.append({"region": int(g), "n": int(te.sum()), "events": int(y[te].sum()),
                    "brier": float(b), "brier_base": float(base), "skill": float(skill)})
        logger.info("  region %d: n=%d events=%d Brier=%.5f base=%.5f skill=%+.4f",
                    g, te.sum(), int(y[te].sum()), b, base, skill)
    mean_s = float(np.mean(skills)) if skills else 0.0
    worst_s = float(np.min(skills)) if skills else 0.0
    logger.info("  MEAN skill %+.4f | WORST region %+.4f", mean_s, worst_s)
    return {"per_region": out, "mean_skill": mean_s,
            "worst_skill": worst_s}


def bootstrap_coefficients(X, y, feature_cols, C, n_boot, names, classifier=True, seed=42):
    """Percentile bootstrap CIs for the effect table (paper Table I).

    Coefficients are on the STANDARDISED scale, because StandardScaler runs
    after the interaction terms are formed. Label the table accordingly.
    """
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    coefs = []
    for i in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        est = (LogisticRegression(max_iter=5000, C=C, solver="lbfgs", random_state=42)
               if classifier else Ridge(alpha=C, random_state=42))
        try:
            m = _pipeline(feature_cols, est).fit(X[idx], y[idx])
        except Exception:
            continue
        c = m.named_steps["estimator"].coef_
        coefs.append(np.ravel(c))
        if (i + 1) % 50 == 0:
            logger.info("    bootstrap %d/%d", i + 1, n_boot)
    if not coefs:
        return None
    A = np.vstack(coefs)
    lo, hi = np.percentile(A, 2.5, axis=0), np.percentile(A, 97.5, axis=0)
    med = np.median(A, axis=0)
    return [{"term": names[j], "median": float(med[j]),
             "ci_low": float(lo[j]), "ci_high": float(hi[j]),
             "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
            for j in range(A.shape[1])]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train causal models for the online tuner")
    ap.add_argument("--data-path", required=True, help="Campaign A rct_results.csv")
    ap.add_argument("--output-dir", default="./models")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Bootstrap resamples for effect-table CIs (use 200 for the paper)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_dataset_csv(args.data_path)
    X, y_safety, y_progress, groups, risk_cols, param_cols, feature_cols, kept_rows = \
        extract_features(rows)

    moved = (y_progress > STALL_EPS_M).astype(int)
    logger.info("Collisions %d/%d (%.2f%%) | stalls %d/%d (%.2f%%)",
                y_safety.sum(), len(y_safety), 100.0 * y_safety.mean(),
                (1 - moved).sum(), len(moved), 100.0 * (1 - moved).mean())

    # 1. safety
    logger.info("Fitting SAFETY model P(Y^H=1 | do(C), R) ...")
    safety_model, C_safety = fit_classifier(X, y_safety, groups, feature_cols, "safety")

    # 2. feasibility (hurdle part 1)
    logger.info("Fitting FEASIBILITY model P(stall | do(C), R) ...")
    stall_model, C_stall = fit_classifier(X, 1 - moved, groups, feature_cols, "stall")

    # 3. speed (hurdle part 2) -- fitted ONLY on trials that moved
    logger.info("Fitting SPEED model E[J^H | moves, do(C), R] on %d moving trials ...",
                int(moved.sum()))
    speed_model, alpha_speed = fit_regressor(X[moved == 1], y_progress[moved == 1],
                                             groups[moved == 1], feature_cols, "speed")

    # Generalisation report -- the number to quote in the paper
    logger.info("Region-held-out safety performance:")
    generalisation = region_holdout_report(X, y_safety, groups, feature_cols, C_safety)

    # Support envelope, so the runtime policy can detect extrapolation
    support = {col: {"min": float(X[:, i].min()), "max": float(X[:, i].max())}
               for i, col in enumerate(feature_cols)}
    progress_support = {"min": 0.0, "max": float(y_progress.max())}

    # Fit bootstrap ensembles (B=50) for safety, stall, and speed models
    logger.info("Fitting bootstrap ensembles (B=50) for confidence bounds...")
    safety_ensemble, stall_ensemble, speed_ensemble = fit_bootstrap_ensemble(
        X, y_safety, 1 - moved, y_progress, moved, feature_cols, C_safety, C_stall, alpha_speed, B=50
    )

    # Support model (k-NN) for positivity diagnostics
    logger.info("Fitting support model...")
    support_model = fit_support_model(X, feature_cols, risk_cols, param_cols, k=10)

    # Policy tree learning via Doubly-Robust Scores
    logger.info("Fitting policy tree via Doubly-Robust scores...")
    action_cols = list(POLICY_ACTION_SET.keys())
    risk_idx = [feature_cols.index(c) for c in risk_cols]
    R_mat = X[:, risk_idx]

    fp_idx = feature_cols.index(FOOTPRINT_KEY)
    carry_obs = X[:, fp_idx]                      # 1.0 carry, 0.0 tucked
    stall_obs = (y_progress <= STALL_EPS_M).astype(float)

    # Observed utility, same functional form as the runtime objective.
    U_obs = ((1.0 + PAYLOAD_OMEGA * carry_obs) * np.asarray(y_progress, float)
             - RISK_LAMBDA * np.asarray(y_safety, float)
             - STALL_MU * stall_obs)

    def _mu(Xa):
        carry = Xa[:, fp_idx]
        p_st = stall_model.predict_proba(Xa)[:, 1]
        e_sp = np.clip(speed_model.predict(Xa), 0.0, progress_support["max"])
        prog = (1.0 - p_st) * e_sp
        p_sf = safety_model.predict_proba(Xa)[:, 1]
        return ((1.0 + PAYLOAD_OMEGA * carry) * prog
                - RISK_LAMBDA * p_sf
                - STALL_MU * p_st)

    Gamma, cells, mu_mat = build_dr_scores(X, kept_rows, U_obs, _mu,
                                           action_cols, POLICY_ACTION_SET, feature_cols)
    policy_tree, leaf_action, leaf_stats = fit_policy_tree(R_mat, Gamma, mu_mat, max_depth=3)

    from sklearn.tree import export_text
    logger.info("Learned policy:\n" + export_text(
        policy_tree, feature_names=list(risk_cols)))

    # Effect table
    names = CausalInteractionTransformer(feature_cols=feature_cols).output_names()
    effect_table = None
    if args.bootstrap > 0:
        logger.info("Bootstrapping safety coefficients (%d resamples) ...", args.bootstrap)
        effect_table = bootstrap_coefficients(X, y_safety, feature_cols, C_safety,
                                              args.bootstrap, names, classifier=True)
        if effect_table:
            sig = [t for t in effect_table if t["excludes_zero"]]
            logger.info("Effect table: %d/%d terms have 95%% CI excluding zero",
                        len(sig), len(effect_table))
            for t in sig:
                logger.info("   %-42s %+8.4f  [%+.4f, %+.4f]",
                            t["term"], t["median"], t["ci_low"], t["ci_high"])

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "safety_model": safety_model,
        "stall_model": stall_model,
        "speed_model": speed_model,
        "safety_ensemble": safety_ensemble,
        "stall_ensemble": stall_ensemble,
        "speed_ensemble": speed_ensemble,
        "support_model": support_model,
        "policy_tree": policy_tree,
        "policy_leaf_action": leaf_action,
        "policy_leaf_stats": leaf_stats,
        "policy_action_cells": cells,
        "policy_action_cols": action_cols,
        "policy_risk_cols": list(risk_cols),
        "objective_weights": {"risk_lambda": RISK_LAMBDA,
                              "stall_mu": STALL_MU,
                              "payload_omega": PAYLOAD_OMEGA},
        "risk_cols": risk_cols,
        "param_cols": param_cols,
        "feature_cols": feature_cols,
        "footprint_key": FOOTPRINT_KEY,
        "support": support,
        "progress_support": progress_support,
        "hyperparams": {"C_safety": C_safety, "C_stall": C_stall,
                        "alpha_speed": alpha_speed},
        "generalisation": generalisation,
        "effect_table": effect_table,
        "n_train": int(len(X)),
        "model_type": "hurdle_logistic_ridge_regioncv",
    }

    out_path = os.path.join(args.output_dir, "causal_tuner_models.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Saved artifact v%d to %s", ARTIFACT_VERSION, out_path)
    logger.info("Runtime expected progress: E[J] = (1 - P_stall) * E[J|moves], "
                "clipped to [%.3f, %.3f]",
                progress_support["min"], progress_support["max"])


if __name__ == "__main__":
    # Pickle resolves these classes by module path. When this file is run as a
    # script, register a stub package so the path exists in sys.modules.
    import types
    _pkg = sys.modules.setdefault("online_causal_tuner", types.ModuleType("online_causal_tuner"))
    _mod = sys.modules.setdefault("online_causal_tuner.train_causal_models",
                                  sys.modules["__main__"])
    setattr(_pkg, "train_causal_models", _mod)
    main()