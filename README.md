# Fascia Robot — Phase 1–3 mechanics validation

This repository contains the controlled Unitree G1 MuJoCo baseline and a minimal
Phase 3 massless, passive, connected torso exofascia. Phase 3 validates network
mechanics and load transmission; it does not claim performance improvement.

## What is implemented

- Unmodified, BSD-licensed Unitree G1 29-DoF MJCF from MuJoCo Menagerie, pinned
  to commit `da76818e269b82289eba39808e2fb91d679d6994`.
- Deterministic 70° left-arm raise using the model's supplied position
  actuators and a minimum-jerk command trajectory.
- Deterministic 9.6 N·s lateral shoulder perturbation (120 N for 80 ms).
- Per-step CSV telemetry: position, velocity, acceleration, finite-difference
  jerk, actuator torque, power, cumulative absolute mechanical work, COM,
  torso quaternion, commands, and external force.
- Derived JSON metrics and PNG baseline plots.
- A 32-node, 84-edge torso network with circumferential, longitudinal,
  alternating diagonal, and long cross-body connections.
- Eight body-fixed attachment nodes and 24 massless free nodes solved at
  quasi-static equilibrium.
- Per-node, per-edge, per-attachment, force-balance, energy, and solver telemetry.
- Static contralateral propagation and dynamic shoulder-perturbation diagnostics.

## Setup

MuJoCo 3.3.7 does not provide a Python 3.14 wheel. Use Python 3.11–3.13:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-build-isolation
```

## Run

```bash
.venv/bin/python -m fascia_robot.cli all
.venv/bin/python -m fascia_robot.cli arm_raise
.venv/bin/python -m fascia_robot.cli perturbation
.venv/bin/python -m fascia_robot.cli fascia_static
.venv/bin/python -m fascia_robot.cli perturbation_fascia
.venv/bin/python -m fascia_robot.cli phase4
.venv/bin/pytest
```

Outputs are written to `results/baseline/<experiment>/`:

- `telemetry.csv`: auditable raw samples
- `summary.json`: configuration, provenance, metric definitions and results
- `baseline.png`: experiment plot

All experiment parameters are editable in `configs/experiments.yaml`.
The fixed, untuned Phase 3 mechanics parameters are in
`configs/fascia_phase3.yaml`.

Phase 3 outputs are written below `results/phase3/`. The static test displaces
the left shoulder boundary by 10 mm and verifies that forces reach the right
torso and pelvis through a single connected graph. The dynamic run uses the
unchanged Phase 2 impulse, controller, timestep, initial state, and seed.

## Scientific scope and limitations

The original morphology, masses, collision geometry, force limits, stand
keyframe, and actuator definitions are unchanged. The controller is the model's
simple joint-position controller, not Unitree's proprietary whole-body balance
or locomotion controller. Menagerie's own model README marks these position
actuators as needing tuning. Consequently these Phase 2 experiments validate
the data pipeline and establish a reproducible open baseline; they do not claim
hardware-equivalent G1 performance.

"Energy" is absolute mechanical actuator work,
`integral(sum(abs(torque * joint_velocity))) dt`. It is not battery/electrical
energy because the public MJCF has no motor-efficiency or electrical model.
Jerk is the numerical time derivative of MuJoCo generalized acceleration and is
therefore timestep- and solver-sensitive. The first and last samples should be
excluded from high-confidence derivative analysis. Reported aggregate metrics
exclude the configured settling warmup, while the raw CSV retains every sample.

The web is an idealized massless axial network. Its free nodes are solved
quasi-statically, so it has no material inertia or finite wave speed. Phase 4
edges are tension-only, using a 10 µN smooth unilateral transition for stable
equilibrium solves. Stiffness and damping were not optimized against baseline
performance.

## Phase 4 controlled comparison

Phase 4 converts edges to tension-only behavior and uses a fixed 5% assembly
pretension selected solely by a pre-performance topology-engagement check. It
runs arm raise, symmetric squat, and lateral perturbation under blinded labels.
Both conditions use identical model state, controller, targets, impulse,
timestep, duration, and seed. Parameters are decoded only after all six trials.

Paired tables, bar plots, time histories, raw telemetry, the configuration hash,
and the decoded protocol are written to `results/phase4/`. Classifications are
descriptive changes from one deterministic simulation pair, not statistical
significance or evidence of hardware performance.

## Anatomical Exofascia v1 — Phase A

Phase A implements geometry only, following `ANATOMICAL_EXOFASCIA_V1.md`:

1. A coarse whole-body superficial mesh
2. Selected biomechanical attachment zones
3. Automated topology checks and a diagnostic visualization

The mesh contains 208 nodes and 560 edges across torso/neck, arms, forearms,
pelvis, thighs, lower legs, ankles, and feet. Forty-four nodes are attachment
nodes and 164 remain internal. All seven components form one connected graph.

```bash
.venv/bin/python -m fascia_robot.cli anatomical_phase_a
```

Outputs are under `results/anatomical_v1/phase_a/`. Mechanics are deliberately
disabled. Deep regional sleeves, reinforced pathways, anisotropic materials,
sliding interfaces, nonlinear materials, propagation experiments, performance
tests, and optimization remain deferred to their specified later phases.

## Anatomical Exofascia v1 — Phase B

Phase B adds geometry in the document's required order: deep trunk fascia,
bilateral arm sleeves, pelvic/thigh/lower-leg sleeves, and bilateral plantar
sheets. The combined Phase A+B topology contains 412 nodes and 1,054 edges;
204 nodes and 494 edges belong to the ten new deep regional structures.

```bash
.venv/bin/python -m fascia_robot.cli anatomical_phase_b
```

The diagnostic image and machine-readable summary are written to
`results/anatomical_v1/phase_b/`. Phase A geometry and its 44 attachment nodes
remain unchanged. The deep structures are deliberately separate geometry
layers: mechanics and superficial/deep shear coupling are not introduced ahead
of their specified phases. Phase C reinforced pathways and all optimization or
performance claims remain deferred.

## Anatomical Exofascia v1 — Phase C

Phase C overlays four configurable reinforced routing families on the unchanged
Phase A+B nodes: bilateral posterior shoulder-to-opposite-pelvis paths,
pelvis-to-lateral-knee paths, calf/Achilles/plantar paths, and chest/scapula-to-
forearm continuity. The combined topology contains 412 nodes and 1,102 edges,
including 48 reinforced pathway edges with explicit evidence labels.

```bash
.venv/bin/python -m fascia_robot.cli anatomical_phase_c
```

Outputs are written to `results/anatomical_v1/phase_c/`. These routes are
biomimetic engineering abstractions, not claims that every pathway has proven
physiological functional significance. Phase C assigns topology only. Phase D
anisotropy, shear interfaces, nonlinear tension-only mechanics, and passivity
validation have deliberately not been introduced yet.

## Anatomical Exofascia v1 — Phase D

Phase D assigns region- and direction-specific material parameters to every
axial edge, adds 204 weak superficial/deep layer-pair interfaces, and implements
a passive tension-only toe-region law with projected Kelvin–Voigt damping. The
interfaces are central-force spring-dampers: they permit finite relative glide,
constrain separation, and preserve internal linear and angular momentum.

```bash
.venv/bin/python -m fascia_robot.cli anatomical_phase_d
```

The full Phase D topology contains 412 nodes and 1,306 elements. Material-law
plots and a validation summary are written to `results/anatomical_v1/phase_d/`.
All initial values are editable in `configs/fascia_materials.yaml`; they have not
been tuned against performance. The mechanics are currently a stateless passive
force evaluator, ready for the prescribed Phase E quasi-static propagation
tests. No MuJoCo performance comparison or improvement claim is made here.

## Anatomical Exofascia v1 — Phase E diagnostic status

Phase E executes five deterministic quasi-static propagation cases: left
shoulder displacement, pelvic rotation, ankle dorsiflexion, arm elevation, and
torso twist. A fixed 1% assembly pretension is used only to engage the
tension-only topology; it was not selected from performance outcomes.

```bash
.venv/bin/python -m fascia_robot.cli anatomical_phase_e
```

The current architecture **does not pass Phase E**. Solvers converge and
internal force/torque balance is preserved, but not every expected distant deep
region exceeds the configured 1 mN change-in-tension threshold. The deep sleeves
can relax toward slack because they have no direct attachment zones and the
weak central layer interfaces primarily constrain separation. Results and the
failure classification are under `results/anatomical_v1/phase_e/`. Performance
testing and optimization must remain blocked until this mechanical connectivity
issue is revised and Phase E passes.
