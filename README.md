# online_causal_tuner

A ROS 2 package for offline causal model training and online dynamic parameter adaptation for Nav2. 

This package uses the randomized controlled trial (RCT) data collected under **Campaign A** to train progress and safety estimation models, then dynamically tunes Nav2 parameters at runtime to maximize progress while maintaining a guaranteed safety boundary.

---

## Architecture Overview

1. **Causal Safety Evaluator $P(Y^H = 1 \mid do(C=c), R_t)$**:
   Predicts the probability of a collision within a short time horizon $H$ given the selected configuration $c$ and the current risk context $R_t$.
2. **Progress Estimator $E[J^H \mid do(C=c), R_t]$**:
   Predicts the expected forward progress distance given the configuration $c$ and risk $R_t$.
3. **Runtime Optimizer**:
   Periodically solves the constrained optimization problem at runtime to choose the parameter configuration $C_t^*$:
   $$C_t^* = \operatorname{argmax}_{c \in C} E[J^H \mid do(C=c), R_t] \quad \text{s.t.} \quad P(Y^H=1 \mid do(C=c), R_t) \le p_{\text{max}}$$

---

## Package Components

### 1. Offline Training (`train_causal_models`)
Trains the progress and safety models using Scikit-Learn Random Forests (or falls back to a custom k-NN model if Scikit-Learn is not installed).

- **Executable name**: `train_causal_models`
- **Inputs**: Campaign A RCT CSV results (`rct_results.csv`).
- **Outputs**: Pickled models artifact `causal_tuner_models.pkl`.

### 2. Runtime Tuner Node (`online_tuner_node`)
Subscribes to `/risk_state` to get the live risk vector $R_t$, solves the optimization problem, and calls Nav2 parameter services to adapt navigation behavior in real-time.

- **Executable name**: `online_tuner_node`
- **Topics**:
  - Subscribes to `/risk_state` (`std_msgs/Float64MultiArray`, 8-dimensional risk vector).
- **Service Clients**:
  - `/controller_server/set_parameters` (Updates `vx_max`, `wz_max`, `time_steps`, `CostCritic.cost_weight`)
  - `/local_costmap/local_costmap/set_parameters` (Updates `inflation_layer.inflation_radius`)
  - `/global_costmap/global_costmap/set_parameters` (Updates `inflation_layer.inflation_radius`)

---

## Installation & Build

Build the package using `colcon` from your workspace root:
```bash
colcon build --packages-select online_causal_tuner
source install/setup.bash
```

---

## Usage

### Step 1: Train Causal Models
Train the models using your collected Campaign A RCT dataset:
```bash
ros2 run online_causal_tuner train_causal_models \
  --data-path /path/to/rct_data_campaign_a/rct_results.csv \
  --output-dir /path/to/models_dir
```

### Step 2: Launch Online Tuner (Dry-Run Mode)
To safely test the tuner and view proposed parameter changes in the console without applying them to the robot:
```bash
ros2 run online_causal_tuner online_tuner_node \
  --ros-args \
  -p model_path:=/path/to/models_dir/causal_tuner_models.pkl \
  -p dry_run:=true
```

### Step 3: Launch Online Tuner (Active Mode)
To let the tuner dynamically update parameters on the live Nav2 stack:
```bash
ros2 run online_causal_tuner online_tuner_node \
  --ros-args \
  -p model_path:=/path/to/models_dir/causal_tuner_models.pkl \
  -p dry_run:=false
```

---

## Parameters Config

You can adjust tuner settings in the YAML configuration file (`config/default_tuner_config.yaml`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `./models/causal_tuner_models.pkl` | Path to trained model pkl |
| `risk_threshold_p_max` | double | `0.15` | Maximum allowed collision probability ($p_{\text{max}}$) |
| `epsilon_exploration` | double | `0.05` | $\epsilon$-greedy exploration rate |
| `tuning_rate_hz` | double | `1.0` | Tuning loop frequency in Hz |
| `n_candidate_samples` | int | `100` | Size of the candidate configuration grid |
| `dry_run` | bool | `true` | When true, parameter updates are only logged |
