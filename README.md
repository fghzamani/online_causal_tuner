# Online Causal Tuner (ICRA Submission Release)

A ROS 2 package for real-time risk-aware parameter adaptation and dynamic arm posture tuning for mobile manipulators navigating in complex and dynamic environments.

This package uses pre-trained causal risk, stall, and progress models to solve a constrained utility maximization problem at runtime (20 Hz). It dynamically adjusts Nav2 controller and costmap parameters, as well as arm posture (carry vs. tucked), to maintain safety while maximizing navigation progress.

---

## Architecture Overview

1. **Causal Risk & Feasibility Evaluators**:
   - **Collision Risk $\hat{P}_{\text{coll}}(c, R_t)$**: Estimates 80th-percentile Upper Confidence Bound (UCB) collision risk given candidate parameters $c$ and live environmental risk context $R_t$.
   - **Stall Probability $\hat{P}_{\text{stall}}(c, R_t)$**: Predicts the probability of robot immobilisation.
   - **Progress Model $\hat{\mathbb{E}}[J^H \mid \text{moves}]$**: Estimates expected forward progress over time horizon $H = 2.8\,\text{s}$.

2. **Feasibility Envelope & Optimizer**:
   - **Gate B1 (Geometric/Kinematic Feasibility $A(R_t)$)**: Filters out inadmissible candidate configurations based on corridor width ($R^{\text{width}}$) and obstacle clearance ($R^{\text{min}}$).
   - **Gate B2 (Pessimistic Utility Maximization)**: Selects configuration $c^*$ maximizing progress LCB minus penalized collision UCB and stall UCB, incorporating hysteresis switching penalties.

3. **Dynamic Obstacle Controller**:
   - Controls an unmapped dynamic obstacle in Gazebo, supporting both **One-Shot Trigger & Hold** mode and **Continuous Oscillating** mode.

---

## Prerequisites & Installation

### Requirements
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill
- **Simulator**: Gazebo 11 / PAL TiAGo Navigation Stack

### Build Package
From your ROS 2 workspace root (e.g., `~/ros2_ws` or `~/phd_projects/online_tuner`):

```bash
cd <your_workspace_root>
colcon build --packages-select online_causal_tuner
source install/setup.bash
```

---

## Quick Start Guide

### Step 1: Launch Gazebo Simulation with Navigation Stack
Launch the robot and Nav2 stack in the benchmark environment with the dynamic obstacle:

```bash
source /opt/ros/humble/setup.bash
source <your_workspace_root>/install/setup.bash

ros2 launch gazebo_simulation gazebo_with_navigation.launch.py world_name:=causal_benchmark_3_dynamic
```

---

### Step 2: Launch Dynamic Obstacle Controller Node
In a second terminal, launch the dynamic obstacle controller.

#### Mode A: One-Shot Trigger & Hold (Default)
Obstacle waits off-path and slides across to block the corridor when the robot approaches:

```bash
source /opt/ros/humble/setup.bash
source <your_workspace_root>/install/setup.bash

ros2 run online_causal_tuner dynamic_obstacle_controller
```

#### Mode B: Continuous Oscillating Back-and-Forth
Obstacle moves back and forth across the corridor continuously from trial start:

```bash
source /opt/ros/humble/setup.bash
source <your_workspace_root>/install/setup.bash

ros2 run online_causal_tuner dynamic_obstacle_controller --ros-args -p mode:=oscillating
```

---

### Step 3: Launch Online Causal Tuner Node
In a third terminal, launch the online tuner node to dynamically adapt parameters in real time.

#### Active Tuning Mode (Updates Nav2 & Arm Posture Live)
```bash
source /opt/ros/humble/setup.bash
source <your_workspace_root>/install/setup.bash

ros2 run online_causal_tuner online_tuner_node --ros-args -p dry_run:=false
```

#### Dry-Run Mode (Console Logging Only)
```bash
source /opt/ros/humble/setup.bash
source <your_workspace_root>/install/setup.bash

ros2 run online_causal_tuner online_tuner_node --ros-args -p dry_run:=true
```

---

## Running Benchmarks & Baselines

You can run automated closed-loop evaluation trials across 30 seed-42 benchmark episodes:

### 1. Online Causal Tuner (Ours)
```bash
python3 analysis/run_campaign_b_benchmark.py \
  --strategy "Online Causal (Ours)" \
  --num-episodes 30 \
  --map-yaml <your_workspace_root>/src/gazebo_simulation/maps/causal_benchmark_3_dynamic.yaml \
  --master-csv evaluation_results/causal_benchmark_3/dynamic_test/campaign_b_world3_dynamic_test_results.csv
```

### 2. Baseline Evaluations (CURE & Nav2 Default)
```bash
# CURE Baseline (Carry Arm Pose)
python3 analysis/run_campaign_b_benchmark.py \
  --strategy "CURE (Carry)" \
  --arm-pose carry \
  --num-episodes 30 \
  --map-yaml <your_workspace_root>/src/gazebo_simulation/maps/causal_benchmark_3_dynamic.yaml \
  --master-csv evaluation_results/causal_benchmark_3/dynamic_test/campaign_b_world3_dynamic_test_results.csv

# Default Nav2 Strategy (Carry Arm Pose)
python3 analysis/run_campaign_b_benchmark.py \
  --strategy "Nav2 Default" \
  --arm-pose carry \
  --num-episodes 30 \
  --map-yaml <your_workspace_root>/src/gazebo_simulation/maps/causal_benchmark_3_dynamic.yaml \
  --master-csv evaluation_results/causal_benchmark_3/dynamic_test/campaign_b_world3_dynamic_test_results.csv
```

---

## Node Parameters Configuration

Key configuration parameters (`config/default_tuner_config.yaml`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `./models/causal_tuner_models.pkl` | Path to pre-trained model artifact |
| `envelope_constants_path` | string | `./models/envelope_constants.json` | Path to feasibility envelope constants |
| `tuning_rate_hz` | double | `20.0` | Real-time control loop frequency (Hz) |
| `alpha` | double | `0.20` | Quantile level for lower/upper confidence bounds |
| `dry_run` | bool | `true` | When true, proposed parameter changes are logged only |
