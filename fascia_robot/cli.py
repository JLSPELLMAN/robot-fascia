from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import PROJECT_ROOT, load_config, run_arm_raise, run_perturbation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic G1 baseline experiments (no fascia).")
    parser.add_argument("experiment", choices=["arm_raise", "perturbation", "all", "fascia_static", "perturbation_fascia", "phase4", "anatomical_phase_a", "anatomical_phase_b", "anatomical_phase_c", "anatomical_phase_d", "anatomical_phase_e"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    results = {}
    if args.experiment == "anatomical_phase_e":
        from .experiments.fascia_propagation import run_phase_e
        results["anatomical_phase_e"] = run_phase_e(output=args.output or PROJECT_ROOT / "results/anatomical_v1/phase_e")
    if args.experiment == "anatomical_phase_d":
        from .experiments.fascia_layer_mechanics import run_phase_d
        results["anatomical_phase_d"] = run_phase_d(output=args.output or PROJECT_ROOT / "results/anatomical_v1/phase_d")
    if args.experiment == "anatomical_phase_c":
        from .experiments.fascia_reinforced_pathways import run_phase_c
        results["anatomical_phase_c"] = run_phase_c(output=args.output or PROJECT_ROOT / "results/anatomical_v1/phase_c")
    if args.experiment == "anatomical_phase_b":
        from .experiments.fascia_deep_regional_geometry import run_phase_b
        results["anatomical_phase_b"] = run_phase_b(output=args.output or PROJECT_ROOT / "results/anatomical_v1/phase_b")
    if args.experiment == "anatomical_phase_a":
        from .experiments.fascia_fullbody_geometry import run_phase_a
        results["anatomical_phase_a"] = run_phase_a(output=args.output or PROJECT_ROOT / "results/anatomical_v1/phase_a")
    if args.experiment == "phase4":
        from .phase4 import run_phase4
        results["phase4"] = run_phase4(output=args.output or PROJECT_ROOT / "results/phase4")
    if args.experiment == "fascia_static":
        from .experiments.fascia_static_test import run_static_test
        results["fascia_static"] = run_static_test(output=args.output or PROJECT_ROOT / "results/phase3/static_deformation")
    if args.experiment == "perturbation_fascia":
        from .experiments.perturbation_fascia import run_perturbation_fascia
        results["perturbation_fascia"] = run_perturbation_fascia(experiment_config=args.config, output=args.output or PROJECT_ROOT / "results/phase3/perturbation_fascia")
    baseline_output = args.output or PROJECT_ROOT / "results/baseline"
    if args.experiment in ("arm_raise", "all"):
        results["arm_raise"] = run_arm_raise(config, baseline_output)
    if args.experiment in ("perturbation", "all"):
        results["perturbation"] = run_perturbation(config, baseline_output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
