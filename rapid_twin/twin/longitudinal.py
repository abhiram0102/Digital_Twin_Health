"""
Longitudinal Vascular Evolution — Stage 2.

Tracks retinal vascular structure, hemodynamics, and biomarkers across
multiple clinical visits for individual patients, building a persistent
per-patient retinal digital twin state.

Provides:
  RetinalVisit             — snapshot at one clinical visit
  GraphAlignment           — cross-visit node correspondence
  BiomarkerTrend           — linear trend estimate for one biomarker
  ProgressionReport        — full longitudinal summary per patient
  PatientTwinState         — persistent state of one patient's retinal twin
  LongitudinalVascularTwin — patient registry + all longitudinal analytics

Clinical design decisions:
  - Graph alignment uses nearest-neighbour node matching (no full image
    registration required), with a spatial tolerance of 10% of disc radius.
  - Biomarker trends fit a linear model (slope + R²) over visit dates.
    Sparse visits (< 2 data points) return nan trends.
  - Risk trajectory is classified from a weighted combination of biomarker
    trends: three or more key biomarkers with adverse slopes → "progressing".
  - All uncertainties are derived from regression residuals (95% PI).

Design contracts:
  - Never mutates an existing RetinalVisit — all updates produce new objects.
  - Patient IDs and visit IDs are arbitrary strings; no ordering assumed.
  - Dates are stored as float (days from epoch) for arithmetic; ISO strings
    are accepted for input convenience.
  - All outputs carry explicit nan for insufficient data — no silent defaults.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Visit snapshot ───────────────────────────────────────────────────────────

def _to_days(d: Any) -> float:
    """Convert a date / ISO string / float-days to float days-since-epoch."""
    if isinstance(d, (int, float)):
        return float(d)
    if isinstance(d, (datetime, date)):
        return float(d.toordinal())
    if isinstance(d, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return float(datetime.strptime(d, fmt).toordinal())
            except ValueError:
                continue
    raise ValueError(f"Cannot parse date: {d!r}")


@dataclass
class RetinalVisit:
    """
    Single clinical visit snapshot for one patient.

    Attributes:
        patient_id:       unique patient identifier string.
        visit_id:         unique visit identifier (e.g., "V1", "2024-03-15").
        date:             visit date — ISO string, date object, or float days.
        graph:            VascularGraph at this visit.
        biomarker_report: BiomarkerReport from RetinalBiomarkerExtractor.
        mask:             (H, W) binary float32 vessel mask.
        metadata:         arbitrary key-value pairs (e.g., camera, operator, laterality).
    """
    patient_id:       str
    visit_id:         str
    date:             Any         # str / date / float
    graph:            Any         # VascularGraph
    biomarker_report: Any         # BiomarkerReport
    mask:             Optional[np.ndarray] = None
    metadata:         Dict[str, Any] = field(default_factory=dict)

    @property
    def date_ordinal(self) -> float:
        """Date as float (days since epoch) for regression arithmetic."""
        return _to_days(self.date)


# ─── Graph alignment ──────────────────────────────────────────────────────────

@dataclass
class GraphAlignment:
    """
    Correspondence between nodes in two VascularGraphs.

    Attributes:
        matched_pairs:    list of (node_id_a, node_id_b, distance_px) tuples.
        alignment_score:  fraction of nodes in the smaller graph that were matched.
        mean_drift_px:    mean spatial drift of matched node pairs in pixels.
        unmatched_a:      node IDs in graph A with no match in B (appeared/disappeared).
        unmatched_b:      node IDs in graph B with no match in A.
    """
    matched_pairs:  List[Tuple[int, int, float]]
    alignment_score: float
    mean_drift_px:   float
    unmatched_a:    List[int]
    unmatched_b:    List[int]


def align_graphs(
    graph_a,
    graph_b,
    max_distance_px: float = 15.0,
) -> GraphAlignment:
    """
    Align nodes between two VascularGraphs by nearest-position matching.

    Uses a greedy bipartite nearest-neighbour strategy:
      1. For each node in A, find the closest unmatched node in B.
      2. Accept if distance < max_distance_px.

    Args:
        graph_a:         VascularGraph (visit A).
        graph_b:         VascularGraph (visit B).
        max_distance_px: maximum allowable node displacement for a match.

    Returns:
        GraphAlignment.
    """
    if not graph_a.nodes or not graph_b.nodes:
        return GraphAlignment(
            matched_pairs=[], alignment_score=0.0, mean_drift_px=float("nan"),
            unmatched_a=[n.node_id for n in graph_a.nodes],
            unmatched_b=[n.node_id for n in graph_b.nodes],
        )

    pos_a = np.array([[n.position[0], n.position[1]] for n in graph_a.nodes])
    pos_b = np.array([[n.position[0], n.position[1]] for n in graph_b.nodes])
    ids_a = [n.node_id for n in graph_a.nodes]
    ids_b = [n.node_id for n in graph_b.nodes]

    # Distance matrix (N_a, N_b)
    diff    = pos_a[:, None, :] - pos_b[None, :, :]       # (N_a, N_b, 2)
    dists   = np.sqrt((diff ** 2).sum(axis=-1))            # (N_a, N_b)

    matched_pairs: List[Tuple[int, int, float]] = []
    matched_b: set = set()

    # Sort A nodes by their minimum distance to any B node (stable greedy)
    a_order = np.argsort(dists.min(axis=1))

    for ai in a_order:
        # Find closest unmatched B node
        row = dists[ai].copy()
        row[list(matched_b)] = np.inf
        bi  = int(np.argmin(row))
        d   = float(row[bi])
        if d < max_distance_px:
            matched_pairs.append((ids_a[ai], ids_b[bi], d))
            matched_b.add(bi)

    matched_a_ids = {p[0] for p in matched_pairs}
    matched_b_ids = {p[1] for p in matched_pairs}

    unmatched_a = [ids_a[i] for i in range(len(ids_a)) if ids_a[i] not in matched_a_ids]
    unmatched_b = [ids_b[i] for i in range(len(ids_b)) if ids_b[i] not in matched_b_ids]

    n_smaller   = min(len(ids_a), len(ids_b))
    alignment   = len(matched_pairs) / max(n_smaller, 1)
    mean_drift  = float(np.mean([p[2] for p in matched_pairs])) if matched_pairs else float("nan")

    return GraphAlignment(
        matched_pairs=matched_pairs,
        alignment_score=alignment,
        mean_drift_px=mean_drift,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
    )


# ─── Biomarker trend estimation ───────────────────────────────────────────────

@dataclass
class BiomarkerTrend:
    """
    Linear trend of one scalar biomarker over time.

    Attributes:
        name:         biomarker name.
        slope:        units/day; positive = increasing over time.
        intercept:    predicted value at t=0 (baseline).
        r_squared:    coefficient of determination; 1.0 = perfect linear fit.
        ci_lower_slope: 95% CI lower bound on slope (from regression std error).
        ci_upper_slope: 95% CI upper bound on slope.
        n_observations: number of valid data points.
        trend_label:  "increasing" / "decreasing" / "stable" / "insufficient_data".
        is_adverse:   True if the slope represents physiological deterioration.
    """
    name:              str
    slope:             float
    intercept:         float
    r_squared:         float
    ci_lower_slope:    float
    ci_upper_slope:    float
    n_observations:    int
    trend_label:       str
    is_adverse:        bool


# Biomarkers where INCREASING value = adverse (e.g., tortuosity, violation rates)
_ADVERSE_IF_INCREASING = {
    "mean_tortuosity", "tortuosity_std", "mean_narrowing_focal",
    "mean_narrowing_diffuse_z", "murray_violation_rate", "caliber_jump_rate",
    "tapering_violation_rate", "av_nicking_index", "av_crossing_score",
    "microaneurysm_proxy", "capillary_dropout_index", "neovascularization_index",
}
# Biomarkers where DECREASING value = adverse (e.g., vessel density, perfusion)
_ADVERSE_IF_DECREASING = {
    "crae", "crve", "avr", "fractal_dimension", "vessel_density_global",
    "peripapillary_vessel_density", "capillary_density",
    "vessel_density_zone_a", "vessel_density_zone_b",
}


def _linear_trend(
    times:  np.ndarray,
    values: np.ndarray,
    z:      float = 1.96,   # 95% CI z-score
) -> Tuple[float, float, float, float, float]:
    """
    Fit y = a*t + b; return (slope, intercept, r², ci_lo, ci_hi).
    Returns nan-tuple if insufficient data.
    """
    n = len(times)
    if n < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan

    t  = times - times.mean()
    y  = values
    Stt = (t ** 2).sum()
    if Stt < 1e-12:
        return 0.0, float(y.mean()), float("nan"), 0.0, 0.0

    slope     = float((t * y).sum() / Stt)
    intercept = float(y.mean() - slope * times.mean())
    y_hat     = slope * times + intercept
    ss_res    = float(((y - y_hat) ** 2).sum())
    ss_tot    = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
    r_squared = float(max(0.0, min(1.0, r_squared)))

    # Standard error of slope
    s2    = ss_res / max(n - 2, 1)
    se    = math.sqrt(s2 / max(Stt, 1e-12))
    ci_lo = slope - z * se
    ci_hi = slope + z * se

    return slope, intercept, r_squared, ci_lo, ci_hi


def estimate_biomarker_trends(
    visits: List[RetinalVisit],
    slope_eps: float = 1e-5,
) -> Dict[str, BiomarkerTrend]:
    """
    Estimate linear trends for all biomarkers across a list of visits.

    Args:
        visits:    list of RetinalVisit sorted by date (or sortable).
        slope_eps: slope magnitude below which trend is labelled "stable".

    Returns:
        dict of biomarker_name → BiomarkerTrend.
    """
    if len(visits) < 2:
        warnings.warn("Trend estimation requires >= 2 visits.")
        return {}

    visits_sorted = sorted(visits, key=lambda v: v.date_ordinal)
    times = np.array([v.date_ordinal for v in visits_sorted], dtype=np.float64)

    # Collect per-biomarker time series
    all_series: Dict[str, List[Tuple[float, float]]] = {}
    for v in visits_sorted:
        flat = v.biomarker_report.to_dict()
        for name, val in flat.items():
            if math.isfinite(val):
                all_series.setdefault(name, []).append((v.date_ordinal, val))

    trends: Dict[str, BiomarkerTrend] = {}

    for name, ts in all_series.items():
        if len(ts) < 2:
            trends[name] = BiomarkerTrend(
                name=name, slope=float("nan"), intercept=float("nan"),
                r_squared=float("nan"), ci_lower_slope=float("nan"),
                ci_upper_slope=float("nan"), n_observations=len(ts),
                trend_label="insufficient_data", is_adverse=False,
            )
            continue

        ts_times  = np.array([t for t, _ in ts], dtype=np.float64)
        ts_values = np.array([v for _, v in ts], dtype=np.float64)

        slope, intercept, r2, ci_lo, ci_hi = _linear_trend(ts_times, ts_values)

        if not math.isfinite(slope):
            label = "insufficient_data"
        elif abs(slope) < slope_eps:
            label = "stable"
        elif slope > 0:
            label = "increasing"
        else:
            label = "decreasing"

        # Adverse direction
        if name in _ADVERSE_IF_INCREASING:
            adverse = slope > slope_eps
        elif name in _ADVERSE_IF_DECREASING:
            adverse = slope < -slope_eps
        else:
            adverse = False

        trends[name] = BiomarkerTrend(
            name=name, slope=slope, intercept=intercept,
            r_squared=float(r2),
            ci_lower_slope=ci_lo, ci_upper_slope=ci_hi,
            n_observations=len(ts),
            trend_label=label, is_adverse=adverse,
        )

    return trends


# ─── Progression report ───────────────────────────────────────────────────────

@dataclass
class ProgressionReport:
    """
    Full longitudinal summary for one patient across all visits.

    Attributes:
        patient_id:          patient identifier.
        n_visits:            total number of visits recorded.
        visit_span_days:     days from first to last visit.
        biomarker_trends:    dict of biomarker_name → BiomarkerTrend.
        n_adverse_trends:    count of biomarkers with adverse slopes.
        n_significant_trends: trends with |slope| > eps and R² > 0.5.
        risk_trajectory:     "stable" / "progressing" / "remitting" / "mixed".
        structural_change_score: aggregate structural change in [0, 1].
        graph_alignments:    cross-visit alignment reports (consecutive visits).
        mean_alignment_score: mean alignment quality across all visit pairs.
    """
    patient_id:              str
    n_visits:                int
    visit_span_days:         float
    biomarker_trends:        Dict[str, BiomarkerTrend]
    n_adverse_trends:        int
    n_significant_trends:    int
    risk_trajectory:         str
    structural_change_score: float
    graph_alignments:        List[GraphAlignment]
    mean_alignment_score:    float


def _classify_trajectory(
    trends: Dict[str, BiomarkerTrend],
    adverse_thresh: int  = 3,
    r2_thresh:      float = 0.40,
) -> str:
    """
    Classify overall disease trajectory from biomarker trend set.

    Returns: "progressing" / "remitting" / "mixed" / "stable".
    """
    n_adverse      = sum(1 for t in trends.values() if t.is_adverse and t.r_squared > r2_thresh)
    n_improving    = sum(1 for t in trends.values() if not t.is_adverse
                         and t.trend_label not in ("stable", "insufficient_data")
                         and t.r_squared > r2_thresh)

    if n_adverse >= adverse_thresh:
        return "progressing"
    if n_improving >= adverse_thresh and n_adverse == 0:
        return "remitting"
    if n_adverse > 0 and n_improving > 0:
        return "mixed"
    return "stable"


# ─── Patient twin state ───────────────────────────────────────────────────────

@dataclass
class PatientTwinState:
    """
    Persistent per-patient retinal digital twin state.

    Updated with each new visit.  Maintains a rolling summary that can be
    queried at any time without replaying all visits.

    Attributes:
        patient_id:    patient identifier.
        visits:        time-ordered list of RetinalVisit.
        latest_trends: most recently computed biomarker trends.
        latest_report: most recently computed ProgressionReport.
        risk_level:    "low" / "moderate" / "high" based on progression.
        last_updated:  ordinal day of last update.
    """
    patient_id:    str
    visits:        List[RetinalVisit] = field(default_factory=list)
    latest_trends: Dict[str, BiomarkerTrend] = field(default_factory=dict)
    latest_report: Optional[ProgressionReport] = None
    risk_level:    str = "unknown"
    last_updated:  float = 0.0

    def _compute_risk_level(self) -> str:
        if self.latest_report is None:
            return "unknown"
        traj = self.latest_report.risk_trajectory
        if traj == "progressing":
            n = self.latest_report.n_adverse_trends
            return "high" if n >= 5 else "moderate"
        if traj == "remitting":
            return "low"
        if traj == "mixed":
            return "moderate"
        return "low"


# ─── Longitudinal Vascular Twin ───────────────────────────────────────────────

class LongitudinalVascularTwin:
    """
    Patient registry and longitudinal analytics engine.

    Maintains a dict of PatientTwinState indexed by patient_id.  Each call to
    add_visit() updates the state and recomputes all trends for that patient.

    Usage:
        twin = LongitudinalVascularTwin(alignment_tolerance_px=12.0)

        twin.add_visit(RetinalVisit(
            patient_id="P001", visit_id="V1", date="2022-01-15",
            graph=g1, biomarker_report=r1, mask=m1
        ))
        twin.add_visit(RetinalVisit(
            patient_id="P001", visit_id="V2", date="2023-06-20",
            graph=g2, biomarker_report=r2, mask=m2
        ))

        report = twin.get_progression_report("P001")
        print(report.risk_trajectory, report.n_adverse_trends)
    """

    def __init__(
        self,
        alignment_tolerance_px: float = 12.0,
        adverse_trend_thresh:   int   = 3,
        r2_significance:        float = 0.40,
    ):
        self.alignment_tolerance   = alignment_tolerance_px
        self.adverse_trend_thresh  = adverse_trend_thresh
        self.r2_significance       = r2_significance
        self._patients: Dict[str, PatientTwinState] = {}

    # ── Visit registration ────────────────────────────────────────────────────

    def add_visit(self, visit: RetinalVisit) -> PatientTwinState:
        """
        Register a new visit and update the patient's twin state.

        Args:
            visit: RetinalVisit (patient_id must match existing or be new).

        Returns:
            Updated PatientTwinState for this patient.
        """
        pid = visit.patient_id
        if pid not in self._patients:
            self._patients[pid] = PatientTwinState(patient_id=pid)

        state = self._patients[pid]
        # Deduplicate: replace if same visit_id already registered
        state.visits = [v for v in state.visits if v.visit_id != visit.visit_id]
        state.visits.append(visit)
        state.visits.sort(key=lambda v: v.date_ordinal)
        state.last_updated = visit.date_ordinal

        # Recompute trends if ≥ 2 visits
        if len(state.visits) >= 2:
            state.latest_trends = estimate_biomarker_trends(state.visits)
            state.latest_report = self._build_report(pid, state)
            state.risk_level    = state._compute_risk_level()

        return state

    # ── Report building ───────────────────────────────────────────────────────

    def _build_report(
        self, pid: str, state: PatientTwinState
    ) -> ProgressionReport:
        visits  = state.visits
        trends  = state.latest_trends

        # Consecutive graph alignments
        alignments: List[GraphAlignment] = []
        for i in range(len(visits) - 1):
            ga = align_graphs(
                visits[i].graph, visits[i + 1].graph,
                self.alignment_tolerance,
            )
            alignments.append(ga)

        mean_align = float(np.mean([a.alignment_score for a in alignments])) if alignments else float("nan")

        n_adverse = sum(1 for t in trends.values() if t.is_adverse)
        n_sig     = sum(1 for t in trends.values()
                        if t.is_adverse and t.r_squared > self.r2_significance)

        trajectory = _classify_trajectory(trends, self.adverse_trend_thresh, self.r2_significance)

        # Structural change score: mean normalised alignment drift across visits
        drifts = [a.mean_drift_px for a in alignments if math.isfinite(a.mean_drift_px)]
        max_drift_ref = max(self.alignment_tolerance, 1)
        sc_score = float(np.clip(np.mean(drifts) / max_drift_ref, 0.0, 1.0)) if drifts else float("nan")

        span = (visits[-1].date_ordinal - visits[0].date_ordinal) if len(visits) > 1 else 0.0

        return ProgressionReport(
            patient_id=pid,
            n_visits=len(visits),
            visit_span_days=float(span),
            biomarker_trends=trends,
            n_adverse_trends=n_adverse,
            n_significant_trends=n_sig,
            risk_trajectory=trajectory,
            structural_change_score=sc_score,
            graph_alignments=alignments,
            mean_alignment_score=mean_align,
        )

    # ── Query interface ───────────────────────────────────────────────────────

    def get_state(self, patient_id: str) -> Optional[PatientTwinState]:
        """Return the current twin state for a patient, or None if unknown."""
        return self._patients.get(patient_id)

    def get_progression_report(self, patient_id: str) -> Optional[ProgressionReport]:
        """Return the latest ProgressionReport for a patient."""
        state = self._patients.get(patient_id)
        if state is None:
            return None
        return state.latest_report

    def list_patients(self) -> List[str]:
        """Return all registered patient IDs."""
        return list(self._patients.keys())

    def predict_biomarker(
        self,
        patient_id: str,
        biomarker:  str,
        days_ahead: float,
    ) -> Tuple[float, float, float]:
        """
        Extrapolate a biomarker value forward in time from the latest visit.

        Uses the fitted linear model (slope, intercept) to extrapolate.

        Args:
            patient_id:  patient identifier.
            biomarker:   biomarker name as it appears in BiomarkerReport.to_dict().
            days_ahead:  number of days ahead to predict.

        Returns:
            (predicted_value, ci_lower, ci_upper)
            All nan if biomarker trend is not available or insufficient data.
        """
        state = self._patients.get(patient_id)
        if state is None or biomarker not in state.latest_trends:
            return (float("nan"),) * 3

        trend = state.latest_trends[biomarker]
        if not math.isfinite(trend.slope) or not state.visits:
            return (float("nan"),) * 3

        t_latest = state.visits[-1].date_ordinal
        t_future = t_latest + days_ahead
        pred     = trend.slope * t_future + trend.intercept

        # CI width from slope uncertainty
        if math.isfinite(trend.ci_upper_slope) and math.isfinite(trend.ci_lower_slope):
            slope_err  = (trend.ci_upper_slope - trend.ci_lower_slope) / 2.0
            ci_half    = slope_err * abs(days_ahead)
            return pred, pred - ci_half, pred + ci_half

        return pred, float("nan"), float("nan")

    def print_summary(self, patient_id: str):
        """Print a concise longitudinal summary for a patient."""
        state = self._patients.get(patient_id)
        if state is None:
            print(f"[LongitudinalTwin] Patient '{patient_id}' not found.")
            return

        report = state.latest_report
        if report is None:
            print(f"[LongitudinalTwin] Patient {patient_id}: {len(state.visits)} visit(s) — insufficient for trend analysis.")
            return

        print(f"\n{'─'*60}")
        print(f"  Patient: {patient_id}   Risk: {state.risk_level.upper()}")
        print(f"  Visits: {report.n_visits}   Span: {report.visit_span_days:.0f} days")
        print(f"  Trajectory: {report.risk_trajectory.upper()}")
        print(f"  Adverse trends: {report.n_adverse_trends}   Significant: {report.n_significant_trends}")
        print(f"  Structural change score: {report.structural_change_score:.3f}")
        print(f"  Mean graph alignment: {report.mean_alignment_score:.3f}")
        print(f"\n  Key adverse trends:")
        for name, t in report.biomarker_trends.items():
            if t.is_adverse and math.isfinite(t.slope):
                print(f"    {name:<35}  slope={t.slope:+.6f}/day  R²={t.r_squared:.3f}  {t.trend_label}")
        print()
