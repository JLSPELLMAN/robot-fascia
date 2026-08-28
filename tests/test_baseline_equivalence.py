import json

from fascia_robot.runner import PROJECT_ROOT, load_config, run_arm_raise, run_perturbation


def test_fascia_disabled_reproduces_frozen_phase2_metrics(tmp_path):
    config = load_config(PROJECT_ROOT / "configs/experiments.yaml")
    expected_arm = json.loads((PROJECT_ROOT / "results/baseline/arm_raise/summary.json").read_text())["metrics"]
    expected_push = json.loads((PROJECT_ROOT / "results/baseline/perturbation/summary.json").read_text())["metrics"]
    actual_arm = run_arm_raise(config, tmp_path)
    actual_push = run_perturbation(config, tmp_path)
    assert actual_arm == expected_arm
    assert actual_push == expected_push
