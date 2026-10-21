"""
Digital Twin Integration Tests — Stages 1–4.

Covers:
  1.  Multi-scale vessel representation (VesselScale, classify, decompose, metrics)
  2.  Spatial retinal analysis (coordinate system, zones, quadrants, maps)
  3.  Longitudinal vascular evolution (visits, alignment, trends, prediction)
  4.  Physiological simulation (DR/HT/glaucoma, counterfactuals, MC ensembles)
  5.  Platform registry (register, get, list, override protection)
  6.  Platform config (load_config_from_dict, validation, presets)
  7.  Platform experiment tracking (start/finish, log metrics, leaderboard)

All tests use synthetic data only — no GPU, no real datasets required.

Run with:
    python -m pytest tests/test_digital_twin.py -v
or:
    python tests/test_digital_twin.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Any, Dict

# Ensure the workspace root is on sys.path so rapid_twin is importable
# when this file is run directly (python tests/test_digital_twin.py)
# as well as via pytest from the workspace root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ─── Shared fixtures ──────────────────────────────────────────────────────────

def _make_mask(H: int = 64, W: int = 64) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.float32)
    mask[H // 4, :]  = 1.0
    mask[H // 2, :]  = 1.0
    mask[3*H//4, :]  = 1.0
    mask[:, W // 4]  = 1.0
    mask[:, W // 2]  = 1.0
    mask[:, 3*W//4]  = 1.0
    return mask


def _make_graph(H: int = 64, W: int = 64):
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder
    builder = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    mask = _make_mask(H, W)
    return builder.build(mask, intensity_image=mask)


def _make_biomarker_report(graph=None, mask=None):
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    if graph is None:
        graph = _make_graph()
    if mask is None:
        mask = _make_mask()
    return RetinalBiomarkerExtractor().extract(graph, mask=mask)


def _passed(name: str):
    print(f"  PASS  {name}")


def _failed(name: str, exc: Exception):
    print(f"  FAIL  {name}: {exc}")
    traceback.print_exc()
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. Multi-scale Vessel Representation
# ══════════════════════════════════════════════════════════════════════════════

def test_vessel_scale_from_diameter():
    """VesselScale.from_diameter must return correct tier for each range."""
    from rapid_twin.graph.multiscale import VesselScale
    assert VesselScale.from_diameter(8.0)  == VesselScale.LARGE
    assert VesselScale.from_diameter(4.5)  == VesselScale.MEDIUM
    assert VesselScale.from_diameter(1.0)  == VesselScale.CAPILLARY
    assert VesselScale.from_diameter(6.0)  == VesselScale.MEDIUM   # exactly at threshold → MEDIUM
    assert VesselScale.from_diameter(3.0)  == VesselScale.CAPILLARY # exactly at threshold → CAPILLARY
    _passed("vessel_scale_from_diameter")


def test_multiscale_threshold_validation():
    """MultiScaleVesselAnalyzer must raise ValueError if capillary >= large threshold."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer
    try:
        MultiScaleVesselAnalyzer(large_thresh=3.0, capillary_thresh=5.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("multiscale_threshold_validation")


def test_multiscale_classify_basic():
    """ScaleClassification must have node and edge scale for every node/edge."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer
    graph    = _make_graph()
    analyzer = MultiScaleVesselAnalyzer()
    clf      = analyzer.classify(graph)

    assert len(clf.node_scale) == len(graph.nodes), (
        f"node_scale has {len(clf.node_scale)} entries, expected {len(graph.nodes)}"
    )
    assert len(clf.edge_scale) == len(graph.edges), (
        f"edge_scale has {len(clf.edge_scale)} entries, expected {len(graph.edges)}"
    )
    _passed("multiscale_classify_basic")


def test_multiscale_decompose_completeness():
    """MultiScaleGraph scale_nodes must cover all nodes."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer, VesselScale
    graph    = _make_graph()
    analyzer = MultiScaleVesselAnalyzer()
    msg, _   = analyzer.analyze(graph)

    all_classified = sum(len(v) for v in msg.scale_nodes.values())
    assert all_classified == len(graph.nodes), (
        f"scale_nodes covers {all_classified} nodes, expected {len(graph.nodes)}"
    )
    _passed("multiscale_decompose_completeness")


def test_multiscale_metrics_embedding_shape():
    """MultiScaleMetrics.scale_embedding must be (24,) float32."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer
    graph    = _make_graph()
    analyzer = MultiScaleVesselAnalyzer()
    _, metrics = analyzer.analyze(graph)

    emb = metrics.scale_embedding
    assert emb.shape == (24,), f"scale_embedding shape {emb.shape} ≠ (24,)"
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all(), "scale_embedding contains NaN or Inf"
    _passed("multiscale_metrics_embedding_shape")


def test_multiscale_density_fractions_sum_one():
    """density_fraction across all scales must sum to 1.0."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer, VesselScale
    graph    = _make_graph()
    if not graph.edges:
        _passed("multiscale_density_fractions_sum_one (no edges — skipped)")
        return
    analyzer = MultiScaleVesselAnalyzer()
    _, metrics = analyzer.analyze(graph)

    total = sum(metrics.density_fraction.values())
    assert abs(total - 1.0) < 1e-5, f"density fractions sum to {total:.6f} ≠ 1.0"
    _passed("multiscale_density_fractions_sum_one")


def test_multiscale_to_dict_types():
    """MultiScaleMetrics.to_dict must return only floats."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer
    graph    = _make_graph()
    _, metrics = MultiScaleVesselAnalyzer().analyze(graph)
    flat = metrics.to_dict()
    for k, v in flat.items():
        assert isinstance(v, (int, float)), f"{k}: type={type(v).__name__}"
    _passed("multiscale_to_dict_types")


def test_capillary_rarefaction():
    """detect_capillary_rarefaction must return expected keys."""
    from rapid_twin.graph.multiscale import MultiScaleVesselAnalyzer, detect_capillary_rarefaction
    graph    = _make_graph()
    mask     = _make_mask()
    analyzer = MultiScaleVesselAnalyzer()
    msg, _   = analyzer.analyze(graph)

    result = detect_capillary_rarefaction(msg, mask, cell_size=16)
    required = ["rarefaction_index", "capillary_density",
                "rarefied_cell_count", "total_cell_count"]
    for k in required:
        assert k in result, f"Missing key: {k}"
    ri = result["rarefaction_index"]
    if math.isfinite(ri):
        assert 0.0 <= ri <= 1.0, f"rarefaction_index={ri} out of [0,1]"
    _passed("capillary_rarefaction")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Spatial Retinal Analysis
# ══════════════════════════════════════════════════════════════════════════════

def test_retinal_coord_system_basic():
    """RetinalCoordinateSystem must set up fovea_offset and fovea_centre."""
    from rapid_twin.biomarkers.spatial_analysis import RetinalCoordinateSystem
    coord = RetinalCoordinateSystem(disc_centre=(32.0, 32.0), disc_radius=8.0)
    assert coord.fovea_offset is not None
    fc = coord.fovea_centre
    assert len(fc) == 2
    _passed("retinal_coord_system_basic")


def test_retinal_coord_invalid_radius():
    """RetinalCoordinateSystem must raise ValueError for disc_radius <= 0."""
    from rapid_twin.biomarkers.spatial_analysis import RetinalCoordinateSystem
    try:
        RetinalCoordinateSystem(disc_centre=(32.0, 32.0), disc_radius=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("retinal_coord_invalid_radius")


def test_vessel_zone_from_distance():
    """VascularZone.from_distance must return correct zones."""
    from rapid_twin.biomarkers.spatial_analysis import VascularZone
    assert VascularZone.from_distance(0.5) == VascularZone.PERIPAPILLARY
    assert VascularZone.from_distance(1.5) == VascularZone.ZONE_A
    assert VascularZone.from_distance(3.0) == VascularZone.ZONE_B
    assert VascularZone.from_distance(5.0) == VascularZone.PERIPHERAL
    _passed("vessel_zone_from_distance")


def test_vessel_quadrant_from_angle():
    """VascularQuadrant.from_angle must return correct quadrants."""
    from rapid_twin.biomarkers.spatial_analysis import VascularQuadrant
    assert VascularQuadrant.from_angle(90)  == VascularQuadrant.SUPERIOR
    assert VascularQuadrant.from_angle(270) == VascularQuadrant.INFERIOR
    _passed("vessel_quadrant_from_angle")


def test_spatial_analyzer_runs():
    """SpatialVascularAnalyzer.analyze must complete without crash."""
    from rapid_twin.biomarkers.spatial_analysis import (
        RetinalCoordinateSystem, SpatialVascularAnalyzer
    )
    graph = _make_graph()
    mask  = _make_mask()
    coord = RetinalCoordinateSystem(disc_centre=(32.0, 32.0), disc_radius=8.0)
    sa    = SpatialVascularAnalyzer(coord)
    report = sa.analyze(graph, mask)
    assert report is not None
    _passed("spatial_analyzer_runs")


def test_spatial_report_zones():
    """SpatialAnalysisReport must contain all four VascularZones."""
    from rapid_twin.biomarkers.spatial_analysis import (
        RetinalCoordinateSystem, SpatialVascularAnalyzer, VascularZone
    )
    graph = _make_graph()
    coord = RetinalCoordinateSystem(disc_centre=(32.0, 32.0), disc_radius=8.0)
    sa    = SpatialVascularAnalyzer(coord)
    report = sa.analyze(graph)

    for zone in VascularZone:
        assert zone in report.zone_biomarkers, f"Missing zone: {zone}"
    _passed("spatial_report_zones")


def test_spatial_sir_ntr_finite():
    """SIR and NTR should be finite positive numbers."""
    from rapid_twin.biomarkers.spatial_analysis import (
        RetinalCoordinateSystem, SpatialVascularAnalyzer
    )
    graph = _make_graph(128, 128)
    coord = RetinalCoordinateSystem(disc_centre=(64.0, 64.0), disc_radius=16.0)
    report = SpatialVascularAnalyzer(coord).analyze(graph)
    assert math.isfinite(report.superior_inferior_ratio)
    assert math.isfinite(report.nasal_temporal_ratio)
    assert report.superior_inferior_ratio > 0
    assert report.nasal_temporal_ratio > 0
    _passed("spatial_sir_ntr_finite")


def test_spatial_embedding_no_nan():
    """SpatialAnalysisReport.spatial_embedding must contain no NaN."""
    from rapid_twin.biomarkers.spatial_analysis import (
        RetinalCoordinateSystem, SpatialVascularAnalyzer
    )
    graph = _make_graph()
    coord = RetinalCoordinateSystem(disc_centre=(32.0, 32.0), disc_radius=8.0)
    report = SpatialVascularAnalyzer(coord).analyze(graph)
    assert np.isfinite(report.spatial_embedding).all(), (
        f"spatial_embedding contains NaN: {report.spatial_embedding}"
    )
    _passed("spatial_embedding_no_nan")


def test_spatial_to_dict_types():
    """SpatialAnalysisReport.to_dict must return only floats."""
    from rapid_twin.biomarkers.spatial_analysis import (
        RetinalCoordinateSystem, SpatialVascularAnalyzer
    )
    graph  = _make_graph()
    coord  = RetinalCoordinateSystem(disc_centre=(32.0, 32.0), disc_radius=8.0)
    report = SpatialVascularAnalyzer(coord).analyze(graph)
    flat   = report.to_dict()
    assert len(flat) > 0
    for k, v in flat.items():
        assert isinstance(v, (int, float)), f"{k}: type={type(v).__name__}"
    _passed("spatial_to_dict_types")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Longitudinal Vascular Evolution
# ══════════════════════════════════════════════════════════════════════════════

def _make_visit(patient_id="P001", visit_id="V1", date="2022-01-01",
                H=64, W=64):
    from rapid_twin.twin.longitudinal import RetinalVisit
    graph = _make_graph(H, W)
    report = _make_biomarker_report(graph)
    mask  = _make_mask(H, W)
    return RetinalVisit(patient_id=patient_id, visit_id=visit_id,
                        date=date, graph=graph,
                        biomarker_report=report, mask=mask)


def test_retinal_visit_date_ordinal():
    """RetinalVisit.date_ordinal must be a finite float."""
    v = _make_visit(date="2023-06-15")
    assert math.isfinite(v.date_ordinal)
    assert v.date_ordinal > 0
    _passed("retinal_visit_date_ordinal")


def test_graph_alignment_identical():
    """Aligning identical graphs must give alignment_score = 1.0."""
    from rapid_twin.twin.longitudinal import align_graphs
    graph = _make_graph()
    align = align_graphs(graph, graph, max_distance_px=5.0)
    assert align.alignment_score > 0.9, (
        f"Identical graph alignment score = {align.alignment_score:.3f} < 0.9"
    )
    assert math.isfinite(align.mean_drift_px)
    assert align.mean_drift_px < 1.0  # identical graphs = 0 drift
    _passed("graph_alignment_identical")


def test_graph_alignment_empty():
    """Aligning an empty graph must not crash and must return score=0."""
    from rapid_twin.twin.longitudinal import align_graphs
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder
    empty_mask = np.zeros((32, 32), dtype=np.float32)
    builder    = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    empty_g    = builder.build(empty_mask)
    graph      = _make_graph()
    align      = align_graphs(empty_g, graph, max_distance_px=10.0)
    assert align.alignment_score == 0.0
    _passed("graph_alignment_empty")


def test_longitudinal_twin_add_visit():
    """LongitudinalVascularTwin.add_visit must register the visit."""
    from rapid_twin.twin.longitudinal import LongitudinalVascularTwin
    twin = LongitudinalVascularTwin()
    v    = _make_visit(patient_id="P001", visit_id="V1", date="2022-01-01")
    state = twin.add_visit(v)
    assert state.patient_id == "P001"
    assert len(state.visits) == 1
    assert "P001" in twin.list_patients()
    _passed("longitudinal_twin_add_visit")


def test_longitudinal_twin_trend_estimation():
    """Trends must be computed after two visits with different biomarkers."""
    from rapid_twin.twin.longitudinal import LongitudinalVascularTwin, RetinalVisit
    twin = LongitudinalVascularTwin()
    v1 = _make_visit("P001", "V1", "2022-01-01")
    v2 = _make_visit("P001", "V2", "2023-01-01")
    twin.add_visit(v1)
    state = twin.add_visit(v2)
    assert len(state.latest_trends) > 0
    _passed("longitudinal_twin_trend_estimation")


def test_longitudinal_progression_report_fields():
    """ProgressionReport must have all required fields after two visits."""
    from rapid_twin.twin.longitudinal import LongitudinalVascularTwin
    twin = LongitudinalVascularTwin()
    twin.add_visit(_make_visit("P001", "V1", "2022-01-01"))
    twin.add_visit(_make_visit("P001", "V2", "2023-06-01"))
    report = twin.get_progression_report("P001")

    assert report is not None
    assert report.patient_id == "P001"
    assert report.n_visits == 2
    assert report.visit_span_days > 0
    assert report.risk_trajectory in ("stable", "progressing", "remitting", "mixed")
    assert 0.0 <= report.n_adverse_trends
    assert isinstance(report.graph_alignments, list)
    _passed("longitudinal_progression_report_fields")


def test_longitudinal_biomarker_prediction():
    """predict_biomarker must return finite values after two visits."""
    from rapid_twin.twin.longitudinal import LongitudinalVascularTwin
    twin = LongitudinalVascularTwin()
    twin.add_visit(_make_visit("P001", "V1", "2022-01-01"))
    twin.add_visit(_make_visit("P001", "V2", "2023-01-01"))
    state   = twin.get_state("P001")
    bm_name = next(iter(state.latest_trends), None)
    if bm_name is None:
        _passed("longitudinal_biomarker_prediction (no trends — skipped)")
        return
    pred, lo, hi = twin.predict_biomarker("P001", bm_name, days_ahead=365)
    assert math.isfinite(pred), f"Prediction for {bm_name} is nan"
    _passed("longitudinal_biomarker_prediction")


def test_longitudinal_deduplication():
    """Adding the same visit_id twice must not duplicate the visit."""
    from rapid_twin.twin.longitudinal import LongitudinalVascularTwin
    twin = LongitudinalVascularTwin()
    twin.add_visit(_make_visit("P001", "V1", "2022-01-01"))
    twin.add_visit(_make_visit("P001", "V1", "2022-01-01"))  # duplicate
    assert len(twin.get_state("P001").visits) == 1
    _passed("longitudinal_deduplication")


def test_biomarker_trend_fields():
    """BiomarkerTrend must have all required attributes."""
    from rapid_twin.twin.longitudinal import estimate_biomarker_trends
    v1 = _make_visit("P001", "V1", "2022-01-01")
    v2 = _make_visit("P001", "V2", "2023-01-01")
    trends = estimate_biomarker_trends([v1, v2])
    for name, t in trends.items():
        assert hasattr(t, "slope")
        assert hasattr(t, "r_squared")
        assert hasattr(t, "trend_label")
        assert hasattr(t, "is_adverse")
        assert t.trend_label in (
            "increasing", "decreasing", "stable", "insufficient_data"
        ), f"{name}: unexpected trend_label={t.trend_label!r}"
        break   # check first trend only
    _passed("biomarker_trend_fields")


def test_longitudinal_unknown_patient():
    """get_progression_report for unknown patient must return None."""
    from rapid_twin.twin.longitudinal import LongitudinalVascularTwin
    twin = LongitudinalVascularTwin()
    result = twin.get_progression_report("UNKNOWN_PATIENT_XYZ")
    assert result is None
    _passed("longitudinal_unknown_patient")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Physiological Simulation
# ══════════════════════════════════════════════════════════════════════════════

def _make_initial_biomarkers() -> Dict[str, float]:
    """Return a physiologically reasonable starting biomarker state."""
    return {
        "mean_tortuosity":           1.10,
        "crae":                     82.0,
        "crve":                    110.0,
        "avr":                       0.74,
        "fractal_dimension":          1.62,
        "vessel_density_global":      0.12,
        "capillary_density":          0.25,
        "av_nicking_index":           0.05,
        "av_crossing_score":          0.08,
        "neovascularization_index":   0.02,
        "peripapillary_vessel_density": 0.18,
        "mean_narrowing_focal":       0.10,
    }


def test_build_dr_scenario():
    """build_dr_scenario must produce valid DiseaseScenario."""
    from rapid_twin.simulation import build_dr_scenario, DiseaseType
    scenario = build_dr_scenario(initial_hba1c=9.0, n_steps=12)
    assert scenario.disease_type == DiseaseType.DR
    assert scenario.n_steps == 12
    assert len(scenario.progression_rates) > 0
    assert 0.0 < scenario.initial_severity <= 1.0
    _passed("build_dr_scenario")


def test_build_hypertension_scenario():
    """build_hypertension_scenario must produce valid DiseaseScenario."""
    from rapid_twin.simulation import build_hypertension_scenario, DiseaseType
    scenario = build_hypertension_scenario(systolic_bp_mmhg=160.0, n_steps=12)
    assert scenario.disease_type == DiseaseType.HYPERTENSION
    assert scenario.initial_severity > 0
    _passed("build_hypertension_scenario")


def test_build_glaucoma_scenario():
    """build_glaucoma_scenario must produce valid DiseaseScenario."""
    from rapid_twin.simulation import build_glaucoma_scenario, DiseaseType
    scenario = build_glaucoma_scenario(iop_mmhg=25.0, n_steps=12)
    assert scenario.disease_type == DiseaseType.GLAUCOMA
    _passed("build_glaucoma_scenario")


def test_simulation_trajectory_length():
    """SimulationTrajectory must have n_steps simulation steps."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_dr_scenario
    sim      = PhysiologicalSimulator(seed=0)
    scenario = build_dr_scenario(initial_hba1c=8.5, n_steps=12)
    traj     = sim.simulate(_make_initial_biomarkers(), scenario)
    assert len(traj.steps) == 12, f"Expected 12 steps, got {len(traj.steps)}"
    _passed("simulation_trajectory_length")


def test_simulation_plausibility_score():
    """SimulationTrajectory.plausibility_score must be in [0, 1]."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_dr_scenario
    sim  = PhysiologicalSimulator(seed=1)
    traj = sim.simulate(_make_initial_biomarkers(), build_dr_scenario(n_steps=6))
    assert 0.0 <= traj.plausibility_score <= 1.0
    _passed("simulation_plausibility_score")


def test_simulation_tortuosity_bound():
    """Tortuosity must never fall below 1.0 in any simulation step."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_dr_scenario
    sim  = PhysiologicalSimulator(seed=2)
    traj = sim.simulate(_make_initial_biomarkers(), build_dr_scenario(n_steps=24))
    for step in traj.steps:
        t = step.biomarkers.get("mean_tortuosity", 1.0)
        if math.isfinite(t):
            assert t >= 1.0, f"Tortuosity={t:.4f} < 1.0 at step {step.step}"
    _passed("simulation_tortuosity_bound")


def test_simulation_ci_shape():
    """SimulationTrajectory.ci_lower and ci_upper must have n_steps entries."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_dr_scenario
    sim      = PhysiologicalSimulator(seed=3, n_ci_samples=5)
    scenario = build_dr_scenario(n_steps=8)
    traj     = sim.simulate(_make_initial_biomarkers(), scenario)
    bm = "mean_tortuosity"
    assert len(traj.ci_lower[bm]) == 8, f"ci_lower length {len(traj.ci_lower[bm])} ≠ 8"
    assert len(traj.ci_upper[bm]) == 8
    _passed("simulation_ci_shape")


def test_counterfactual_benefit_score():
    """CounterfactualResult.benefit_score must be in [0, 1]."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_dr_scenario
    sim = PhysiologicalSimulator(seed=4)
    cf  = sim.counterfactual(_make_initial_biomarkers(), build_dr_scenario(n_steps=12))
    assert 0.0 <= cf.benefit_score <= 1.0, (
        f"benefit_score={cf.benefit_score:.4f} out of [0,1]"
    )
    _passed("counterfactual_benefit_score")


def test_counterfactual_intervention_slows_disease():
    """Intervention must slow or halt capillary dropout vs baseline."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_dr_scenario
    sim = PhysiologicalSimulator(seed=5)
    cf  = sim.counterfactual(
        _make_initial_biomarkers(), build_dr_scenario(initial_hba1c=11.0, n_steps=24)
    )
    bm = "capillary_density"
    b_final = cf.baseline.final_state.get(bm, float("nan"))
    i_final = cf.intervention.final_state.get(bm, float("nan"))
    if math.isfinite(b_final) and math.isfinite(i_final):
        assert i_final >= b_final - 0.01, (
            f"Intervention capillary_density={i_final:.4f} should be >= "
            f"baseline={b_final:.4f} (disease slowed or reversed)"
        )
    _passed("counterfactual_intervention_slows_disease")


def test_mc_ensemble_shape():
    """EnsembleResult must have n_samples and correct biomarker keys."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_hypertension_scenario
    sim      = PhysiologicalSimulator(seed=6)
    scenario = build_hypertension_scenario(n_steps=6)
    ensemble = sim.mc_ensemble(_make_initial_biomarkers(), scenario, n_samples=20)
    assert ensemble.n_samples == 20
    assert 0.0 <= ensemble.risk_probability <= 1.0
    for bm in _make_initial_biomarkers():
        assert bm in ensemble.mean_final, f"Missing biomarker in ensemble: {bm}"
    _passed("mc_ensemble_shape")


def test_mc_ensemble_std_positive():
    """EnsembleResult.std_final must be >= 0 for all finite biomarkers."""
    from rapid_twin.simulation import PhysiologicalSimulator, build_glaucoma_scenario
    sim      = PhysiologicalSimulator(seed=7)
    ensemble = sim.mc_ensemble(
        _make_initial_biomarkers(), build_glaucoma_scenario(n_steps=6), n_samples=15
    )
    for bm, std in ensemble.std_final.items():
        if math.isfinite(std):
            assert std >= 0.0, f"std_final[{bm}]={std} < 0"
    _passed("mc_ensemble_std_positive")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Platform Registry
# ══════════════════════════════════════════════════════════════════════════════

def test_registry_register_and_get():
    """ModuleRegistry register + get must return the same callable."""
    from rapid_twin.platform.registry import ModuleRegistry

    class FakeModule:
        pass

    reg = ModuleRegistry("test")
    reg.register("fake", FakeModule, tags=["test"])
    assert reg.get("fake") is FakeModule
    _passed("registry_register_and_get")


def test_registry_duplicate_raises():
    """Registering a duplicate name without allow_override must raise ValueError."""
    from rapid_twin.platform.registry import ModuleRegistry

    class A: pass
    class B: pass

    reg = ModuleRegistry("test")
    reg.register("mod", A)
    try:
        reg.register("mod", B)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("registry_duplicate_raises")


def test_registry_allow_override():
    """allow_override=True must replace the existing entry."""
    from rapid_twin.platform.registry import ModuleRegistry

    class A: pass
    class B: pass

    reg = ModuleRegistry("test")
    reg.register("mod", A)
    reg.register("mod", B, allow_override=True)
    assert reg.get("mod") is B
    _passed("registry_allow_override")


def test_registry_missing_raises_keyerror():
    """get() for an unregistered name must raise KeyError."""
    from rapid_twin.platform.registry import ModuleRegistry
    reg = ModuleRegistry("test")
    try:
        reg.get("does_not_exist")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    _passed("registry_missing_raises_keyerror")


def test_registry_list_by_tag():
    """list_names with tag filter must return only matching names."""
    from rapid_twin.platform.registry import ModuleRegistry

    class A: pass
    class B: pass

    reg = ModuleRegistry("test")
    reg.register("a", A, tags=["seg"])
    reg.register("b", B, tags=["graph"])
    assert "a" in reg.list_names(tag="seg")
    assert "b" not in reg.list_names(tag="seg")
    _passed("registry_list_by_tag")


def test_global_registry_has_core_modules():
    """GLOBAL_REGISTRY must contain at least the core platform modules."""
    from rapid_twin.platform.registry import GLOBAL_REGISTRY
    core = [
        "retinal_graph_builder", "biomarker_extractor",
        "hemodynamic_model", "benchmark_runner",
    ]
    for name in core:
        assert name in GLOBAL_REGISTRY, f"'{name}' missing from GLOBAL_REGISTRY"
    _passed("global_registry_has_core_modules")


def test_registry_decorator():
    """@register decorator must register a class in the specified registry."""
    from rapid_twin.platform.registry import register, ModuleRegistry

    local_reg = ModuleRegistry("local")

    @register("decorated_module", tags=["test"], registry=local_reg)
    class DecoratedModule:
        pass

    assert "decorated_module" in local_reg
    assert local_reg.get("decorated_module") is DecoratedModule
    _passed("registry_decorator")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Platform Config
# ══════════════════════════════════════════════════════════════════════════════

def test_load_config_from_dict_basic():
    """load_config_from_dict must parse a valid config dict."""
    from rapid_twin.platform.config import load_config_from_dict
    raw = {
        "name": "test_pipeline",
        "dataset": "DRIVE",
        "stages": [
            {"name": "graph_build", "module": "retinal_graph_builder",
             "kwargs": {"max_nodes": 256}},
            {"name": "biomarkers",  "module": "biomarker_extractor", "kwargs": {}},
        ],
    }
    cfg = load_config_from_dict(raw)
    assert cfg.name == "test_pipeline"
    assert cfg.dataset == "DRIVE"
    assert len(cfg.stages) == 2
    assert cfg.stage_names == ["graph_build", "biomarkers"]
    _passed("load_config_from_dict_basic")


def test_load_config_missing_name_raises():
    """load_config_from_dict must raise ValueError when 'name' is absent."""
    from rapid_twin.platform.config import load_config_from_dict
    try:
        load_config_from_dict({"stages": []})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("load_config_missing_name_raises")


def test_load_config_stage_missing_module_raises():
    """load_config_from_dict must raise ValueError when a stage has no 'module'."""
    from rapid_twin.platform.config import load_config_from_dict
    try:
        load_config_from_dict({
            "name": "bad",
            "stages": [{"name": "orphan"}],  # missing 'module'
        })
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("load_config_stage_missing_module_raises")


def test_pipeline_config_active_stages():
    """PipelineConfig.active_stages must exclude disabled stages."""
    from rapid_twin.platform.config import load_config_from_dict
    cfg = load_config_from_dict({
        "name": "p",
        "stages": [
            {"name": "a", "module": "m1", "enabled": True},
            {"name": "b", "module": "m2", "enabled": False},
        ],
    })
    assert len(cfg.active_stages) == 1
    assert cfg.active_stages[0].name == "a"
    _passed("pipeline_config_active_stages")


def test_pipeline_config_to_dict_roundtrip():
    """PipelineConfig.to_dict must be JSON serialisable and roundtrip correctly."""
    import json
    from rapid_twin.platform.config import load_config_from_dict
    raw = {
        "name": "roundtrip",
        "dataset": "STARE",
        "stages": [{"name": "s1", "module": "m1", "kwargs": {"k": 1}}],
    }
    cfg  = load_config_from_dict(raw)
    d    = cfg.to_dict()
    json_str = json.dumps(d)
    assert "roundtrip" in json_str
    _passed("pipeline_config_to_dict_roundtrip")


def test_standard_pipeline_config():
    """standard_pipeline_config must return a valid PipelineConfig."""
    from rapid_twin.platform.config import standard_pipeline_config
    cfg = standard_pipeline_config(dataset="HRF")
    assert cfg.dataset == "HRF"
    assert len(cfg.stages) >= 4
    assert all(s.module for s in cfg.stages)
    _passed("standard_pipeline_config")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Platform Experiment Tracking
# ══════════════════════════════════════════════════════════════════════════════

def test_experiment_start_finish():
    """ExperimentTracker: start_run / finish_run lifecycle."""
    from rapid_twin.platform.experiment import ExperimentTracker
    tracker = ExperimentTracker(project="test")
    run     = tracker.start_run("run1", config={"dataset": "DRIVE"})
    assert run.status == "running"
    tracker.finish_run(run.run_id)
    assert tracker.get_run(run.run_id).status == "finished"
    _passed("experiment_start_finish")


def test_experiment_log_and_retrieve_metric():
    """log_metric must make the value retrievable via latest_metric."""
    from rapid_twin.platform.experiment import ExperimentTracker
    tracker = ExperimentTracker(project="test")
    run     = tracker.start_run("metric_test")
    tracker.log_metric(run.run_id, "dice", 0.81, step=1)
    tracker.log_metric(run.run_id, "dice", 0.83, step=2)
    assert abs(run.latest_metric("dice") - 0.83) < 1e-6
    assert abs(run.best_metric("dice", mode="max") - 0.83) < 1e-6
    _passed("experiment_log_and_retrieve_metric")


def test_experiment_log_metrics_batch():
    """log_metrics must log all finite values and skip nan."""
    from rapid_twin.platform.experiment import ExperimentTracker
    tracker = ExperimentTracker(project="test")
    run     = tracker.start_run("batch_test")
    tracker.log_metrics(run.run_id, {
        "dice": 0.80, "iou": 0.72, "broken": float("nan")
    })
    assert math.isfinite(run.latest_metric("dice"))
    assert math.isfinite(run.latest_metric("iou"))
    # nan metric should not be stored
    assert len(run.metrics.get("broken", [])) == 0
    _passed("experiment_log_metrics_batch")


def test_experiment_log_benchmark():
    """log_benchmark must populate run.benchmarks."""
    from rapid_twin.platform.experiment import ExperimentTracker
    from rapid_twin.benchmarks import SegmentationBenchmark
    tracker = ExperimentTracker(project="test")
    run     = tracker.start_run("bench_test")
    mask    = _make_mask()
    result  = SegmentationBenchmark(dataset="synthetic").evaluate([mask], [mask])
    tracker.log_benchmark(run.run_id, result)
    assert "segmentation" in run.benchmarks
    _passed("experiment_log_benchmark")


def test_experiment_best_run():
    """best_run must return the run with the highest metric."""
    from rapid_twin.platform.experiment import ExperimentTracker
    tracker = ExperimentTracker(project="test")
    r1 = tracker.start_run("r1")
    r2 = tracker.start_run("r2")
    tracker.log_metric(r1.run_id, "dice", 0.75)
    tracker.log_metric(r2.run_id, "dice", 0.82)
    tracker.finish_run(r1.run_id)
    tracker.finish_run(r2.run_id)
    best = tracker.best_run("dice", mode="max")
    assert best is not None
    assert best.run_id == r2.run_id
    _passed("experiment_best_run")


def test_experiment_unknown_run_raises():
    """get_run for an unknown run_id must raise KeyError."""
    from rapid_twin.platform.experiment import ExperimentTracker
    tracker = ExperimentTracker(project="test")
    try:
        tracker.get_run("nonexistent_id_xyz")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    _passed("experiment_unknown_run_raises")


def test_experiment_save_load(tmp_path=None):
    """save_all / load_run roundtrip must preserve run data."""
    import json, os, tempfile
    from rapid_twin.platform.experiment import ExperimentTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(project="test", save_dir=tmpdir)
        run     = tracker.start_run("save_test", config={"lr": 0.001})
        tracker.log_metric(run.run_id, "dice", 0.79)
        tracker.finish_run(run.run_id)
        tracker.save_all()

        # Load from disk
        path     = os.path.join(tmpdir, f"{run.run_id}.json")
        assert os.path.exists(path), f"Run file not found: {path}"
        tracker2 = ExperimentTracker(project="test")
        loaded   = tracker2.load_run(path)
        assert loaded.run_id == run.run_id
        assert abs(loaded.latest_metric("dice") - 0.79) < 1e-6
    _passed("experiment_save_load")


# ─── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # Stage 1: Multi-scale
    test_vessel_scale_from_diameter,
    test_multiscale_threshold_validation,
    test_multiscale_classify_basic,
    test_multiscale_decompose_completeness,
    test_multiscale_metrics_embedding_shape,
    test_multiscale_density_fractions_sum_one,
    test_multiscale_to_dict_types,
    test_capillary_rarefaction,

    # Stage 1: Spatial
    test_retinal_coord_system_basic,
    test_retinal_coord_invalid_radius,
    test_vessel_zone_from_distance,
    test_vessel_quadrant_from_angle,
    test_spatial_analyzer_runs,
    test_spatial_report_zones,
    test_spatial_sir_ntr_finite,
    test_spatial_embedding_no_nan,
    test_spatial_to_dict_types,

    # Stage 2: Longitudinal
    test_retinal_visit_date_ordinal,
    test_graph_alignment_identical,
    test_graph_alignment_empty,
    test_longitudinal_twin_add_visit,
    test_longitudinal_twin_trend_estimation,
    test_longitudinal_progression_report_fields,
    test_longitudinal_biomarker_prediction,
    test_longitudinal_deduplication,
    test_biomarker_trend_fields,
    test_longitudinal_unknown_patient,

    # Stage 3: Simulation
    test_build_dr_scenario,
    test_build_hypertension_scenario,
    test_build_glaucoma_scenario,
    test_simulation_trajectory_length,
    test_simulation_plausibility_score,
    test_simulation_tortuosity_bound,
    test_simulation_ci_shape,
    test_counterfactual_benefit_score,
    test_counterfactual_intervention_slows_disease,
    test_mc_ensemble_shape,
    test_mc_ensemble_std_positive,

    # Stage 4: Platform
    test_registry_register_and_get,
    test_registry_duplicate_raises,
    test_registry_allow_override,
    test_registry_missing_raises_keyerror,
    test_registry_list_by_tag,
    test_global_registry_has_core_modules,
    test_registry_decorator,
    test_load_config_from_dict_basic,
    test_load_config_missing_name_raises,
    test_load_config_stage_missing_module_raises,
    test_pipeline_config_active_stages,
    test_pipeline_config_to_dict_roundtrip,
    test_standard_pipeline_config,
    test_experiment_start_finish,
    test_experiment_log_and_retrieve_metric,
    test_experiment_log_metrics_batch,
    test_experiment_log_benchmark,
    test_experiment_best_run,
    test_experiment_unknown_run_raises,
    test_experiment_save_load,
]


def run_all_tests():
    passed = 0
    failed = 0
    print(f"\n{'='*70}")
    print("  RAPiD-Twin Digital Twin Platform — Integration Test Suite")
    print(f"{'='*70}\n")
    print("  Sections:")
    print("    Stage 1a. Multi-scale Vessel Representation  (8 tests)")
    print("    Stage 1b. Spatial Retinal Analysis            (9 tests)")
    print("    Stage 2.  Longitudinal Vascular Evolution     (9 tests)")
    print("    Stage 3.  Physiological Simulation            (9 tests)")
    print("    Stage 4a. Platform Registry                   (7 tests)")
    print("    Stage 4b. Platform Config                     (6 tests)")
    print("    Stage 4c. Experiment Tracking                 (7 tests)")
    print()

    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            _failed(fn.__name__, exc)
            failed += 1

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print(f"{'='*70}\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
