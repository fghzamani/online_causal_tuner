#!/usr/bin/env python3
"""
Offline Causal Model Training Pipeline for Online Causal Tuner.

Trains two models from Campaign A RCT data:
  1. Causal Safety Evaluator P(Y^H = 1 | do(C=c), R=r):
     Predicts collision probability within horizon H given treatment c and risk r.
  2. Progress Estimator E[J^H | do(C=c), R=r]:
     Predicts expected forward progress distance / speed given treatment c and risk r.

Usage:
  ros2 run online_causal_tuner train_causal_models \
    --data-path ./rct_data_campaign_a/rct_results.csv \
    --output-dir ./models
"""

import os
import sys
import argparse
import logging
import csv
import math
import pickle
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_causal_models")

# Standard feature column keys
RISK_FEATURE_KEYS = [
    "risk__r_clear",
    "risk__r_vis",
    "risk__r_dens",
    "risk__r_ttc",
    "risk__r_width",
    "risk__r_curve",
    "risk__r_grad",
    "risk__r_min",
    "risk__a_t",
    "risk__r_a_t",
    "risk__arm_extension",
    "risk__clearance_m",
    "risk__visibility_occluded_fraction",
    "risk__obstacle_density",
    "risk__ttc_sec",
    "risk__corridor_width_m",
    "risk__path_curvature",
]

PARAM_KEYS = [
    "param__controller_server__FollowPath.vx_max",
    "param__controller_server__FollowPath.wz_max",
    "param__local_costmap__inflation_layer.inflation_radius",
    "param__local_costmap__obstacle_layer.scan.obstacle_max_range",
    "param__controller_server__FollowPath.time_horizon",
    "param__controller_server/FollowPath.vx_max",
    "param__controller_server/FollowPath.wz_max",
    "param__local_costmap/local_costmap.inflation_layer.inflation_radius",
]


class KNNModel:
    """Lightweight k-NN Classifier/Regressor fallback for environments without scikit-learn."""
    def __init__(self, k=5, is_classifier=True):
        self.k = k
        self.is_classifier = is_classifier
        self.X_train = []
        self.y_train = []
        self.mean = []
        self.std = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        # Standard scaling
        n_features = len(X[0]) if X else 0
        self.mean = [sum(X[i][j] for i in range(len(X))) / max(1, len(X)) for j in range(n_features)]
        self.std = [
            math.sqrt(sum((X[i][j] - self.mean[j]) ** 2 for i in range(len(X))) / max(1, len(X))) or 1.0
            for j in range(n_features)
        ]

    def _scale(self, sample):
        return [(sample[j] - self.mean[j]) / self.std[j] for j in range(len(sample))]

    def predict_proba(self, X_eval):
        scaled_train = [self._scale(row) for row in self.X_train]
        probs = []
        for sample in X_eval:
            s_sample = self._scale(sample)
            dists = [
                (math.sqrt(sum((s_sample[j] - tr[j]) ** 2 for j in range(len(s_sample)))), self.y_train[i])
                for i, tr in enumerate(scaled_train)
            ]
            dists.sort(key=lambda x: x[0])
            top_k = dists[: self.k]
            p1 = sum(y for _, y in top_k) / float(self.k)
            probs.append([1.0 - p1, p1])
        return probs

    def predict(self, X_eval):
        scaled_train = [self._scale(row) for row in self.X_train]
        preds = []
        for sample in X_eval:
            s_sample = self._scale(sample)
            dists = [
                (math.sqrt(sum((s_sample[j] - tr[j]) ** 2 for j in range(len(s_sample)))), self.y_train[i])
                for i, tr in enumerate(scaled_train)
            ]
            dists.sort(key=lambda x: x[0])
            top_k = dists[: self.k]
            if self.is_classifier:
                p1 = sum(y for _, y in top_k) / float(self.k)
                preds.append(1 if p1 >= 0.5 else 0)
            else:
                avg = sum(y for _, y in top_k) / float(self.k)
                preds.append(avg)
        return preds


def load_dataset_csv(data_path: str):
    """Load CSV rows into list of dicts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    rows = []
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        clean_lines = (line.replace('\x00', '') for line in f)
        reader = csv.DictReader(clean_lines)
        for row in reader:
            # Filter valid treatment rows
            if row.get("treatment_valid") in ("0", 0):
                continue
            if row.get("baseline_valid") in ("0", 0):
                continue
            rows.append(row)

    logger.info(f"Loaded {len(rows)} valid rows from {data_path}.")
    return rows


def extract_features(rows: list):
    if not rows:
        raise ValueError("Dataset rows list is empty.")

    header_keys = list(rows[0].keys())
    risk_cols = [k for k in header_keys if k in RISK_FEATURE_KEYS or k.startswith("risk__")]
    param_cols = [k for k in header_keys if k in PARAM_KEYS or k.startswith("param__")]

    feature_cols = risk_cols + param_cols
    logger.info(f"Extracted {len(risk_cols)} risk features: {risk_cols[:4]}...")
    logger.info(f"Extracted {len(param_cols)} parameters: {param_cols[:4]}...")

    X = []
    y_safety = []
    y_progress = []

    for r in rows:
        feat_vec = []
        for col in feature_cols:
            val = r.get(col, "0")
            if col == "param__local_costmap__footprint":
                if val == "carry":
                    feat_vec.append(1.0)
                elif val == "tucked":
                    feat_vec.append(0.0)
                else:
                    feat_vec.append(0.0)
            else:
                try:
                    feat_vec.append(float(val) if val != "" else 0.0)
                except ValueError:
                    feat_vec.append(0.0)

        # Safety target: y_h or collision
        y_s = r.get("y_h", r.get("collision", "0"))
        try:
            y_safety.append(int(float(y_s)) if y_s != "" else 0)
        except ValueError:
            y_safety.append(0)

        # Progress target: probe_progress_m
        y_p = r.get("probe_progress_m", "1.0")
        try:
            y_progress.append(float(y_p) if y_p != "" else 1.0)
        except ValueError:
            y_progress.append(1.0)

        X.append(feat_vec)

    return X, y_safety, y_progress, risk_cols, param_cols, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train Causal Models for Online Tuner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to Campaign A rct_results.csv")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory for trained models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_dataset_csv(args.data_path)
    X, y_safety, y_progress, risk_cols, param_cols, feature_cols = extract_features(rows)

try:
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


if SKLEARN_AVAILABLE:
    class CausalInteractionTransformer(BaseEstimator, TransformerMixin):
        """
        Constructs interaction terms phi(c) x r matching main.tex Table I.
        Appends active configuration-risk interaction terms, configuration cross-terms,
        and dynamic arm actuation lag (a_t) interaction terms.
        """
        def __init__(self, feature_cols=None):
            self.feature_cols = feature_cols or []

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_mat = np.array(X, dtype=np.float64)
            if not self.feature_cols or X_mat.shape[1] == 0:
                return X_mat

            col_to_idx = {col: i for i, col in enumerate(self.feature_cols)}

            def get_col(name_substr):
                for k, idx in col_to_idx.items():
                    if name_substr in k:
                        return X_mat[:, idx]
                return np.zeros(X_mat.shape[0])

            c_v = get_col("vx_max")
            c_w = get_col("wz_max")
            c_inf = get_col("inflation_radius")
            c_obs = get_col("cost_weight")
            c_hor = get_col("time_horizon")
            c_arm = get_col("footprint")

            r_ttc = get_col("r_ttc")
            r_width = get_col("r_width")
            r_curve = get_col("r_curve")
            r_clear = get_col("r_clear")
            r_min = get_col("r_min")
            r_dens = get_col("r_dens")
            r_grad = get_col("r_grad")
            r_vis = get_col("r_vis")
            r_a_t = get_col("a_t")

            interactions = [
                c_v * r_ttc,
                c_v * r_width,
                c_v * r_curve,
                c_v * r_clear,
                c_w * r_clear,
                c_w * r_curve,
                c_inf * r_width,
                c_inf * r_min,
                c_obs * r_min,
                c_obs * r_dens,
                c_obs * r_grad,
                c_hor * r_ttc,
                c_hor * r_curve,
                c_arm * r_vis,
                c_arm * r_width,
                c_arm * r_min,
                c_v * c_hor,
                c_inf * c_arm,
                # Dynamic Arm Actuation Lag (a_t in [0, 1]) Interactions
                c_v * r_a_t,
                c_arm * r_a_t,
                r_width * r_a_t,
            ]

            inter_mat = np.column_stack(interactions)
            return np.hstack([X_mat, inter_mat])


def main():
    parser = argparse.ArgumentParser(description="Train Causal Models for Online Tuner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to Campaign A rct_results.csv")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory for trained models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_dataset_csv(args.data_path)
    X, y_safety, y_progress, risk_cols, param_cols, feature_cols = extract_features(rows)

    if SKLEARN_AVAILABLE:
        logger.info("Fitting Causal Parametric Models (main.tex formulation)...")
        X_mat = np.array(X, dtype=np.float64)

        y_safe_mat = np.array(y_safety, dtype=np.int32)
        y_prog_mat = np.array(y_progress, dtype=np.float64)

        # =====================================================================
        # RANDOM FOREST IMPLEMENTATION (COMMENTED OUT AS REQUESTED)
        # =====================================================================
        # from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        # logger.info("Using Scikit-Learn RandomForest for model training...")
        # safety_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        # safety_model.fit(X_mat, y_safe_mat)
        # progress_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        # progress_model.fit(X_mat, y_prog_mat)
        # model_type = "rf"
        # =====================================================================

        # Causal Model 1: Logistic Regression for Safety P(Y^H = 1 | do(C=c), R_t=r)
        transformer = CausalInteractionTransformer(feature_cols=feature_cols)
        safety_pipeline = Pipeline([
            ("interaction", transformer),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42))
        ])
        safety_pipeline.fit(X_mat, y_safe_mat)
        safety_model = safety_pipeline

        # Causal Model 2: Ridge Linear Regression for Progress E[J^H | do(C=c), R_t=r]
        progress_pipeline = Pipeline([
            ("interaction", transformer),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=42))
        ])
        progress_pipeline.fit(X_mat, y_prog_mat)
        progress_model = progress_pipeline

        model_type = "causal_logistic_ridge"
        logger.info("Successfully trained Causal Logistic & Ridge Regression models with sparse interaction terms ✓")

    else:
        logger.info("Scikit-Learn not found in environment; using KNNModel fallback...")
        safety_model = KNNModel(k=5, is_classifier=True)
        safety_model.fit(X, y_safety)

        progress_model = KNNModel(k=5, is_classifier=False)
        progress_model.fit(X, y_progress)

        model_type = "knn"

    artifact = {
        "safety_model": safety_model,
        "progress_model": progress_model,
        "feature_scaler": None,
        "risk_cols": risk_cols,
        "param_cols": param_cols,
        "feature_cols": feature_cols,
        "safety_model_type": model_type,
        "progress_model_type": model_type,
    }

    out_path = os.path.join(args.output_dir, "causal_tuner_models.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    logger.info(f"Successfully trained and saved causal models to {out_path} ✓")


if __name__ == "__main__":
    main()

