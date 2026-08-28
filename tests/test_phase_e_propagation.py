from __future__ import annotations

import json

import numpy as np

from fascia_robot.experiments.fascia_propagation import run_phase_e


def test_phase_e_pipeline_is_deterministic_balanced_and_flags_current_failure(tmp_path):
    first = run_phase_e(tmp_path/"first")
    second = run_phase_e(tmp_path/"second")
    assert first["topology_hash"] == second["topology_hash"]
    assert first["case_count"] == 5 and first["all_solvers_converged"]
    assert first["validation_status"] == "failed_expected_distant_region_engagement"
    assert not first["all_expected_regions_engaged"]
    for name, case in first["cases"].items():
        repeated = second["cases"][name]
        assert case["moved_attachment_nodes"]
        assert case["success"]
        assert case["free_force_residual_n"] < 1e-4
        assert np.linalg.norm(case["net_internal_force_n"]) < 1e-12
        assert np.linalg.norm(case["net_internal_torque_nm"]) < 1e-12
        assert np.isclose(case["stored_energy_j"], repeated["stored_energy_j"], rtol=0, atol=1e-12)
        assert case["all_expected_regions_engaged"] == repeated["all_expected_regions_engaged"]
    assert json.loads((tmp_path/"first"/"summary.json").read_text())["validation_status"] == first["validation_status"]
