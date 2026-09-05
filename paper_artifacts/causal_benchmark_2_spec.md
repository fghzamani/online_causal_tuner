# `causal_benchmark_2` — specification for a second evaluation world

**Status:** design specification. Write this document, and commit it, *before*
running the online tuner in the new world. See §1.

---

## 1. Scientific standing of this exercise

You are designing a new environment after observing that Envelope-Only ties your
full method. That is a fork in the road, and which branch you take decides
whether the new world strengthens the paper or destroys it.

**The illegitimate version.** Build a world, run everything, keep it if your
method wins, and report it as "the evaluation world." This is
hypothesizing-after-results-are-known. A reviewer who sees one world in the
paper, and a prior arXiv version or a repository history with another, will
conclude the environment was selected on the outcome. That conclusion is fatal
in a causal-inference paper, because environment selection on the outcome is
precisely the bias the paper claims to eliminate.

**The legitimate version, which is what this document is for.** The argument is
not "I need a world where I win." It is:

> The first world varies along one configuration dimension only. Every passage
> in it either admits both arm states or neither, so the software parameters
> have no work to do and the context-dependence claim is untestable there. A
> second world is required whose *induced optimal configuration varies
> spatially in more than the arm dimension*.

That argument is derivable from the parameter space and the robot's geometry
without reference to any result, which is what makes it admissible. Three
obligations follow, and all three are load-bearing:

1. **Specify the world by mechanism before running the tuner.** Every geometric
   choice below is justified by which parameter it makes binding and why. No
   dimension is chosen because a particular method handles it well.
2. **Validate with a static-configuration oracle sweep (§6), not with your
   method.** The pass criteria are stated in §6.2 and fixed in advance. If the
   world fails them, fix the world; do not proceed and hope.
3. **Report both worlds.** `causal_benchmark` becomes World 1 and is reported in
   full, including the Envelope-Only tie. World 2 is reported as a second
   environment designed to exercise the software dimensions, with this rationale
   stated in the paper. Two worlds with a stated design difference is a
   contribution. One world with a buried predecessor is misconduct.

If you cannot fit both worlds in the page budget, report World 1 in the paper
and World 2 in a supplementary section — not the other way round.

---

## 2. Robot geometry constants

All widths below derive from these. **Resolve the first row before generating
anything**: the paper and the deployed costmap polygon disagree, and every
clearance band in this document shifts by 5 cm depending on which is right.

| Quantity | Symbol | Value | Source |
|---|---|---|---|
| Tucked circumscribed radius | $r^{\text{tuck}}_{\text{circ}}$ | **0.325 m** (paper) / 0.275 m (costmap polygon) | §V, `nav2_params.yaml` |
| Tucked inscribed radius | $r^{\text{tuck}}_{\text{insc}}$ | ≈ 0.27 m | polygon |
| Carry circumscribed radius | $r^{\text{carry}}_{\text{circ}}$ | 0.847 m | §V |
| Map resolution | $\rho$ | 0.05 m | `causal_benchmark.yaml` |
| Goal xy tolerance | — | 0.35 m | `general_goal_checker` |
| Inflation range (tuned) | $C^{\text{inf}}$ | 0.15 – 0.60 m | Table II |
| Speed range (tuned) | $C^{\text{spd}}$ | 30 – 95 % of 0.7 m/s = 0.21 – 0.665 m/s | Table II |
| MPPI lookahead | — | $56 \times 0.05 \times v_{\max}$ = 1.96 m at full speed | `nav2_params.yaml` |

Use the **larger** value (0.325) for design margins. If the true radius is
0.275 the passages become easier, which is a safe direction; the reverse is not.

### 2.1 Derived clearance bands

A free gap of width $W$ between two obstacle faces:

| Band | Width | Meaning |
|---|---|---|
| **B0** impassable | $W < 0.70$ | Below $2r^{\text{tuck}}_{\text{circ}} + 2\rho$. Do not place. |
| **B1** tucked-only, inflation-binding | $0.75 \le W \le 0.95$ | Free centre after inflation is $W - 2C^{\text{inf}}$. At $C^{\text{inf}}{=}0.15$ this is 0.45–0.65 m; at $C^{\text{inf}}{=}0.60$ it is negative and the whole gap carries near-lethal cost. **Inflation choice binds here.** |
| **B2** tucked-only, inflation-free | $1.00 \le W \le 1.60$ | Tucked passes at any inflation. Carry blocked. Arm dimension binds, nothing else. This is the *entire* content of World 1. |
| **B3** carry-admissible | $W \ge 1.85$ | $2r^{\text{carry}}_{\text{circ}} + 0.15$. Carry passes. Needed so the transport task is actually performed. |

World 1 contains only B2. **World 2 must contain B1, B2 and B3, and must contain
at least one pair of passages requiring opposite settings of the same
parameter.** That opposition is the whole design; see §4.

---

## 3. World envelope and conventions

Keep the conventions of `causal_benchmark` so the existing toolchain works
unchanged.

```
bounds        x ∈ [-6, 6], y ∈ [-16, 16]   (12 m × 32 m)
walls         0.20 m thick, 2.0 m tall, at x = ±6 and y = ±16
map yaml      resolution 0.0500
              origin [-6.0000, -16.0000, 0.0000]
              mode trinary, negate 0, occupied_thresh 0.65, free_thresh 0.25
pgm size      240 × 640 px
obstacles     box primitives, static, floor-mounted (bottom z = 0)
naming        zone-prefixed: z1_pinch_west, z2_pole_03, ...
```

**Every obstacle must reach the floor and rise to at least 1.0 m.** The LiDAR
plane is around 0.2 m. An obstacle that starts above the scan plane is invisible
to $R_t$ and makes the trial uninterpretable — you would be measuring the
planar-clearance limitation, not configuration adaptation. Keep that
limitation as a stated limitation, not as an uncontrolled variable.

---

## 4. Zone design

Six zones, south to north. Missions run between pose-pool samples, so most
routes traverse three or more zones.

The design principle is **opposition**: for each software parameter that the
paper claims is context-dependent, there are two zones that require *opposite*
settings of it. This is what makes "no single static configuration is adequate"
true by construction of the environment rather than by luck, and it is
verifiable independently of any method (§6).

### Z1 — Carry hall (south), $y \in [-16, -11]$

Open, 11.6 m wide. Contains half the pose pool. Exit doorway at $y = -11$ of
width **2.60 m** (band B3), centred at $x = 0$.

```
z1_door_west   pose (-3.65, -11.00, 1.00)   size (4.70, 0.20, 2.00)   # spans x -6.00 .. -1.30
z1_door_east   pose ( 3.65, -11.00, 1.00)   size (4.70, 0.20, 2.00)   # spans x  1.30 ..  6.00
                                                                      # gap = 2.60 m, centred x = 0
```

*Purpose.* Carry is admissible here, so the $\omega$ bonus is exercised and
`carry_distance_m` accumulates. World 1 has no B3 passage at all, which is why
carry distance was 1.03 m of an 8.54 m path. Without a zone like this the
transport task is nominal.

### Z2 — Pinch (inflation must be LOW), $y \in [-11, -7]$

A single gap of width **0.85 m** (band B1) at $x = -2.0$, in a divider at
$y = -7$.

```
z2_pinch_west  pose (-4.2125, -7.00, 1.00)  size (3.575, 0.20, 2.00)  # spans x -6.000 .. -2.425
z2_pinch_east  pose ( 2.2125, -7.00, 1.00)  size (7.575, 0.20, 2.00)  # spans x -1.575 ..  6.000
                                                                      # gap = 0.850 m, centred x = -2.00
```

*Purpose and mechanism.* Free centre width is $0.85 - 2C^{\text{inf}}$: 0.55 m at
$C^{\text{inf}}{=}0.15$, 0.25 m at 0.30, negative at 0.45 and 0.60. The robot's
inscribed diameter is 0.54 m. So at low inflation the gap has a genuinely free
corridor; at 0.30 the robot must traverse inflated cells at cost; at 0.45+ the
`SmacPlannerLattice` `cost_penalty` makes the passage prohibitive and the
planner will prefer to fail. **This zone requires $C^{\text{inf}} \le 0.30$ and
$C^{\text{arm}} = $ tucked.**

Note this is the passage Envelope-Only cannot adapt to: it pins
`inflation_radius` at 0.30 (`ENVELOPE_ONLY_SOFTWARE`). That is a *consequence* of
the design, not its motivation — the zone exists because inflation is a claimed
context-dependent parameter and World 1 never makes it binding. State it that
way in the paper, and place the zone before you look at any result.

### Z3 — Clutter field (inflation must be HIGH), $y \in [-7, -2]$

Nine free-standing square posts, 0.20 × 0.20 m footprint, 1.20 m tall, on a
jittered grid with nominal spacing 1.60 m.

```
z3_post_01 (-4.10, -6.10)   z3_post_02 (-2.30, -5.40)   z3_post_03 (-0.40, -6.30)
z3_post_04 ( 1.50, -5.60)   z3_post_05 ( 3.60, -6.20)   z3_post_06 (-3.40, -3.90)
z3_post_07 (-1.20, -3.20)   z3_post_08 ( 0.90, -4.10)   z3_post_09 ( 3.10, -3.40)
   all: size (0.20, 0.20, 1.20), z = 0.60
```

*Purpose and mechanism.* Free lanes between posts are 1.2–1.6 m, comfortably
B2, so the passage is never blocked. What varies is *margin*. At
$C^{\text{inf}} = 0.15$ the inflated halo is smaller than the robot's own
inscribed radius, so MPPI's optimal trajectory passes within centimetres of the
posts and small tracking error produces contact. At $C^{\text{inf}} \ge 0.45$ the
lanes carry a cost gradient that centres the robot. **This zone requires
$C^{\text{inf}} \ge 0.45$, the opposite of Z2**, and rewards a high
`CostCritic` weight.

Posts rather than walls because thin obstacles are where inflation choice
actually shows: a long wall is avoided by the path itself, a post is avoided
only by margin.

### Z4 — Blind bend (speed must be LOW), $y \in [-2, 3]$

An L-bend corridor with a 90° turn and no line of sight through it. Vertical
leg 1.10 m wide, horizontal leg 1.00 m wide (both band B2), so the turn tightens
as the robot commits to it.

```
z4_wall_a  pose (-1.65,  0.50, 1.00)   size (0.20, 5.00, 2.00)   # inner face x = -1.55
z4_wall_b  pose (-0.35, -0.60, 1.00)   size (0.20, 2.80, 2.00)   # inner face x = -0.45
                                                                 # vertical leg gap = 1.10 m
z4_wall_c  pose ( 1.60,  0.85, 1.00)   size (4.30, 0.20, 2.00)   # inner face y = 0.95
z4_wall_d  pose ( 1.60,  2.05, 1.00)   size (4.30, 0.20, 2.00)   # inner face y = 1.95
                                                                 # horizontal leg gap = 1.00 m
```

*Purpose and mechanism.* MPPI looks ahead $56 \times 0.05 \times v_{\max}$, which
is 1.96 m at $C^{\text{spd}} = 95\%$ and 0.62 m at 30%. The corridor leg before
the turn is 2.8 m and the turn narrows from 1.10 m to 1.00 m at the corner. Entering at 0.665 m/s
the controller's horizon extends past the corner into space the LiDAR cannot
see, and the sampled trajectory set contains no admissible turn; entering at
0.21 m/s it does. **This zone requires $C^{\text{spd}} \le 50\%$.** It is also
where $R^{\text{ttc}}$ and $R^{\text{clear}}$ should carry signal, so it tests
the risk state as well as the policy.

### Z5 — Long hall (speed must be HIGH), $y \in [3, 11]$

Open, 11.6 m wide, 8 m long, no obstacles. Entry doorway at $y = 3$ of width
**2.20 m** (band B3) centred at $x = 1.5$.

```
z5_door_west   pose (-2.80, 3.00, 1.00)   size (6.40, 0.20, 2.00)   # spans x -6.00 .. 0.40
z5_door_east   pose ( 4.30, 3.00, 1.00)   size (3.40, 0.20, 2.00)   # spans x  2.60 .. 6.00
                                                                    # gap = 2.20 m, centred x = 1.50
```

*Purpose and mechanism.* **This zone requires $C^{\text{spd}} \ge 70\%$**, and it
requires it through the mission time budget, not through geometry: at 0.21 m/s
an 8 m hall costs 38 s, and a mission crossing Z1→Z5 exceeds the 240 s timeout
if the tuner holds a low speed limit throughout. Set the timeout deliberately
(§5) so this is true; do not leave it to chance. B3 doorway so carry is
admissible again after Z4 forced retraction — this is what produces arm
switching within a single mission, which World 1 almost never does (mean 1.17
arm switches).

### Z6 — Goal room (north), $y \in [11, 16]$

Open, contains the other half of the pose pool. Two cabinets against the north
wall as terminal-approach obstacles:

```
z6_cab_west   pose (-4.20, 14.60, 0.50)   size (1.20, 0.80, 1.00)
z6_cab_east   pose ( 4.20, 14.60, 0.50)   size (1.20, 0.80, 1.00)
```

### 4.1 Opposition summary

This table is the claim the world makes. It should appear in the paper.

| Parameter | Zone requiring LOW | Zone requiring HIGH | Separation |
|---|---|---|---|
| $C^{\text{inf}}$ | Z2 pinch ($\le 0.30$) | Z3 clutter ($\ge 0.45$) | full tuned range |
| $C^{\text{spd}}$ | Z4 blind bend ($\le 50\%$) | Z5 long hall ($\ge 70\%$) | full tuned range |
| $C^{\text{arm}}$ | Z2, Z4 (tucked) | Z1, Z5 (carry admissible) | both states used |

No static assignment satisfies both columns. That is the property World 1 lacks
and it is what makes the context-dependence claim testable.

---

## 5. Protocol parameters

| Parameter | Value | Rationale |
|---|---|---|
| Mission timeout | **180 s** | Long enough for a Z1→Z6 traverse at moderate speed (~34 m route at 0.4 m/s ≈ 85 s plus recovery), short enough that holding 30% throughout fails. Fix this before running and record it in every trial JSON. |
| Pose pool | 30 start–goal pairs | Stratified: 10 pairs crossing ≥ 4 zones, 10 crossing 2–3, 10 within-zone. Report the stratum with each result. |
| Repeats | **3 per (strategy, pair)** | Still absent from both campaigns. Without it there is no within-cell variance and the paired tests are the only inference available. |
| Goal clearance in sampler | ≥ 0.70 m | Existing `--goal-clearance` default. Verify no sampled goal lands in Z2 or Z4 corridors, where the tolerance cannot be met. |
| Fixed critics | unchanged | GoalCritic 20, GoalAngleCritic 10, PreferForward threshold 1.5, identical across strategies and across both worlds. |

---

## 6. Validation before the tuner ever runs

### 6.1 Static-configuration oracle sweep

Run **static configurations only** — no learned model, no tuner, no
Envelope-Only. A coarse grid is sufficient:

```
C_arm  ∈ {tucked, carry}                    2
C_spd  ∈ {30, 50, 70, 95} %                 4
C_inf  ∈ {0.15, 0.30, 0.45, 0.60} m         4
                                    = 32 configurations
```

over 10 pose pairs, 1 repeat = 320 trials. At the current throughput that is a
few hours and it is the cheapest insurance available.

For each configuration record success, collision, planning failure and per-zone
traversal success (which zones the robot entered and left).

### 6.2 Pass criteria — fix these now, do not adjust them after seeing results

The world is usable if and only if all four hold:

1. **No dominant static configuration.** The best single configuration by
   overall success rate achieves **< 0.75**. (World 1 fails this: fixed tucked
   reaches 0.93.)
2. **Spatial opposition realized.** The configuration with the highest Z2
   traversal rate and the configuration with the highest Z3 traversal rate
   differ in $C^{\text{inf}}$ by at least one grid step, and likewise for
   $C^{\text{spd}}$ between Z4 and Z5.
3. **Both arm states used.** At least one pose pair is traversable carrying and
   at least one requires retraction; carry is admissible over ≥ 40% of the
   nominal route length.
4. **Not degenerate.** At least one static configuration achieves ≥ 0.40 overall.
   If everything fails, the world is impossible rather than discriminating, and
   an impossible world proves nothing.

If a criterion fails, adjust the geometry it targets — widen Z2, respace Z3,
lengthen Z5 — and re-run the sweep. Log every adjustment and its reason. Those
adjustments are legitimate because none of them consults your method's
performance. **Do not run the online tuner or Envelope-Only until all four
criteria pass.**

### 6.3 Characterize both worlds

Report these for World 1 and World 2 side by side, so the reader sees the
difference is measured rather than asserted. These follow the environment
characterization used in constrained-navigation benchmarking (BARN, Xiao et al.
2022) and are computed from the occupancy grid, not from any trial:

- distribution of minimum corridor width along the nominal route (report the
  5th, 50th and 95th percentiles);
- fraction of route length in each band B1 / B2 / B3;
- obstacle dispersion (mean distance from each free cell to the nearest lethal
  cell);
- average visibility (mean fraction of the free space with line of sight from a
  free cell), which is what distinguishes Z4 from Z5.

---

## 7. What this world can and cannot buy you

**Can.** If the sweep passes §6.2, the world contains passages that no single
static configuration handles, in two software dimensions plus the arm. A
method that adapts along those dimensions should then separate from
Envelope-Only, which holds inflation at 0.30 and speed at 0.55 by
construction and therefore cannot satisfy Z2 and Z3 simultaneously.

**Cannot.** It cannot retroactively fix World 1. Report World 1 as it stands,
including the tie, and present World 2 as a second environment built to a stated
specification. Two worlds with different induced difficulty, both reported, is a
stronger paper than one world that happens to favour you — and it is the only
version that survives a reviewer who checks.

**Also cannot.** It cannot substitute for repeats. Thirty pose pairs at one
repeat in a harder world still gives you no within-cell variance, and a harder
world has *higher* outcome variance, not lower. Budget the three repeats first;
if you must choose between three repeats in World 1 and one repeat in World 2,
choose the repeats.

---

## 8. Deadline reality

Generating the world, running the 320-trial sweep, iterating the geometry, and
then running 6 strategies × 30 pairs × 3 repeats is on the order of a week of
wall-clock with no failures. Today is 5 September; the deadline is the 15th.

If that does not fit, the fallback is not to rush it. It is to report World 1
honestly with the Envelope-Only tie and the framing in §7 — the causal analysis
identified the arm dimension, and a policy acting on that identification
recovers the full method's performance — and to name the second world as
designed-but-not-yet-run future work with this specification as the plan. That
is a publishable paper. A hurried second world with an unvalidated protocol is
not.
