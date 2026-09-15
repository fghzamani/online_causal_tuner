#!/usr/bin/env python3
"""
Graph specification for the CURE baseline.

CURE (Hossen et al., T-RO 2025) does not take a causal graph as input. It learns
one with FCI (Fast Causal Inference, a constraint-based structure-learning
algorithm) and constrains that search with "tabu edges": edges the algorithm is
forbidden to draw. Those tabu edges encode a layered assumption:

    configuration options  ->  intermediate metrics  ->  objectives

with no direct option -> objective edge. Everything an option does to an
objective has to pass through a measured intermediate quantity.

This file holds three things:

  1. LAYERS      -- which of your columns are options, metrics, objectives, and
                    (an addition of ours) pre-treatment context.
  2. tabu_edges  -- the forbidden-edge list, same construction as CURE's
                    `CausalModel.get_tabu_edges_care_remove`, extended for the
                    context layer.
  3. ABSTRACT_GRAPH -- optional. If you supply your own abstract causal graph,
                    the reduction script can use it directly instead of running
                    FCI, or use it as required-edge background knowledge.

Terminology used throughout, spelled out once:
  ACE   Average Causal Effect: E[Y | do(T=1)] - E[Y | do(T=0)].
  ADMG  Acyclic Directed Mixed Graph: a graph with directed edges (A -> B) and
        bidirected edges (A <-> B, meaning "A and B share a hidden common
        cause"). FCI outputs a graph class that gets resolved into an ADMG.
  FCI   Fast Causal Inference, the structure-learning algorithm CURE uses.
  MOBO  Multi-Objective Bayesian Optimization.
  PAG   Partial Ancestral Graph, the object FCI actually returns.
"""

# ---------------------------------------------------------------------------
# LAYER 0: pre-treatment context  (R in your notation)
# ---------------------------------------------------------------------------
# NOT part of CURE. CURE's Turtlebot/Husky campaigns run one fixed mission, so
# there is no context variable in their schema at all. Your Campaign A varies
# context across probes, and context is measured BEFORE the configuration is
# applied. Leaving it out would let context confound the option -> outcome
# associations. It is included as an exogenous layer: nothing points into it.
CONTEXT = [
    "risk__r_min",
    "risk__r_width",
    "risk__r_ttc",
    "risk__r_dens",
    "risk__r_clear",
    "risk__r_curve",
]

# ---------------------------------------------------------------------------
# LAYER 1: configuration options  (C in your notation)
# ---------------------------------------------------------------------------
# The tuner's action space, so that CURE and the tuner search the same set.
# Section 6 of the validity review flagged that they currently do not.
OPTIONS = [
    "param__controller_server__speed_limit_pct",
    "param__controller_server__FollowPath.vx_std",
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight",
    "param__controller_server__FollowPath.CostCritic.cost_weight",
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight",
    "param__local_costmap__inflation_layer.inflation_radius",
]

# Arm state. Kept separate because you may want to run CURE both with and
# without it (see --include-arm in the reduction and MOBO scripts).
ARM_OPTION = "param__local_costmap__footprint"

# ---------------------------------------------------------------------------
# LAYER 2: intermediate metrics  (the mediators)
# ---------------------------------------------------------------------------
# CURE's metrics are: Planner_failed, Recovery_executed, Positional_error,
# Traveled_distance, Mission_time, Obstacle_distance.
#
# These are the CAMPAIGN A per-probe equivalents. Edit the names to whatever
# your rct_results.csv actually contains -- the scripts check and will tell you
# which ones are missing rather than silently proceeding.
#
# If your probe rows genuinely have no mediators, read section 3 of
# README_CURE_BASELINE.md before going further: without a metrics layer the
# tabu-edge construction collapses and CURE degenerates into plain regression.
METRICS = [
    "achieved_min_obstacle_distance",  # <- CURE's Obstacle_distance
    "d_total_plan_m",                  # <- CURE's Traveled_distance / path
    "achieved_max_vx",                  # <- CURE's Speed metric
    "mean_path_deviation_m",            # <- CURE's Positional/Path error
    "probe_stalled",                    # <- CURE's Recovery/Stall metric
]

# ---------------------------------------------------------------------------
# LAYER 3: objectives
# ---------------------------------------------------------------------------
# CURE optimizes two objectives under two constraints. The mapping used here:
#
#   CURE                  ->  yours
#   Energy (min)          ->  negative progress, i.e. minimize -probe_progress_m
#   Positional_error (min)->  handled at the closed-loop stage (final_xy_error)
#   Task_success_rate     ->  constraint, reached rate
#   Obstacle_distance     ->  constraint, min clearance
#
# At the Campaign A (probe) stage there are only two outcomes worth modelling.
OBJECTIVES = [
    "probe_progress_m",   # maximize
    "y_h",                # collision indicator, minimize
]


def all_columns(include_arm: bool = True):
    """Every column the graph is defined over, in layer order."""
    opts = OPTIONS + ([ARM_OPTION] if include_arm else [])
    return list(CONTEXT) + list(opts) + list(METRICS) + list(OBJECTIVES)


def options(include_arm: bool = True):
    return OPTIONS + ([ARM_OPTION] if include_arm else [])


# ---------------------------------------------------------------------------
# Tabu (forbidden) edges
# ---------------------------------------------------------------------------
def tabu_edges(include_arm: bool = True, forbid_option_option: bool = True):
    """Edges the structure search may not draw.

    Reproduces CURE's `get_tabu_edges_care_remove` and adds the context layer.

    CURE's three rules:
      (a) metric -> option        forbidden  (options are set before we measure)
      (b) objective -> metric     forbidden
      (c) option -> objective AND objective -> option  both forbidden, which is
          what forces every option effect to be mediated by a metric.

    Our two additions:
      (d) nothing -> context      forbidden. Context is measured before the
          configuration is applied, so it cannot be caused by anything in the
          model. This is the pre-treatment requirement from your design.
      (e) option -> option        forbidden by default. CURE leaves this on
          (their rule is commented out in causal_model.py), but in a randomized
          campaign the options are drawn independently, so an option -> option
          edge is guaranteed to be a false positive. Set
          forbid_option_option=False to match CURE's shipped code exactly.
    """
    opts = options(include_arm)
    tabu = []

    for met in METRICS:                                  # (a)
        for opt in opts:
            tabu.append((met, opt))

    for obj in OBJECTIVES:                               # (b)
        for met in METRICS:
            tabu.append((obj, met))

    for opt in opts:                                     # (c)
        for obj in OBJECTIVES:
            tabu.append((opt, obj))
            tabu.append((obj, opt))

    for ctx in CONTEXT:                                  # (d)
        for other in opts + METRICS + OBJECTIVES:
            tabu.append((other, ctx))

    if forbid_option_option:                             # (e)
        for a in opts:
            for b in opts:
                if a != b:
                    tabu.append((a, b))

    return tabu


# ---------------------------------------------------------------------------
# Optional: our own abstract causal graph
# ---------------------------------------------------------------------------
# Fill this in if you want to skip FCI (--graph-mode given) or to force edges
# you are confident about (--graph-mode hybrid). Each entry is (cause, effect).
# Leave the list empty to use FCI alone, which is what CURE does.
#
# Example of the shape, using your own variables:
#
# ABSTRACT_GRAPH_DI = [
#     ("risk__r_width", "probe_min_obstacle_dist_m"),
#     ("param__local_costmap__footprint", "probe_min_obstacle_dist_m"),
#     ("param__local_costmap__inflation_layer.inflation_radius",
#      "probe_min_obstacle_dist_m"),
#     ("probe_min_obstacle_dist_m", "y_h"),
#     ("param__controller_server__speed_limit_pct", "probe_path_length_m"),
#     ("probe_path_length_m", "probe_progress_m"),
# ]
ABSTRACT_GRAPH_DI = []      # directed edges  A -> B
ABSTRACT_GRAPH_BI = []      # bidirected edges A <-> B (shared hidden cause)