"""
Predictive Physiological Simulator — Stage 3.

Converts RAPiD-Twin from a descriptive system into a predictive physiological
simulator.  Starting from a patient's current vascular state (biomarker report),
it simulates the expected evolution of retinal vascular parameters over time
under different disease trajectories and clinical interventions.

Provides:
  DiseaseType              — enum (DR, HYPERTENSION, GLAUCOMA, CUSTOM)
  DiseaseScenario          — disease parameters + progression rates per biomarker
  SimulationStep           — biomarker state at one time-point
  SimulationTrajectory     — full time series + confidence intervals
  CounterfactualResult     — baseline vs intervention comparison
  EnsembleResult           — Monte Carlo ensemble summary
  PhysiologicalSimulator   — orchestrator for all simulation modes

Disease models:
  DR (Diabetic Retinopathy):
    - Vessel caliber narrowing / microaneurysm accumulation
    - Progressive capillary dropout (capillary_density ↓)
    - Non-perfusion zone expansion (ischemic index ↑)
    - Neovascularisation at late stages (tortuosity ↑)

  Hypertension:
    - Arterial narrowing (CRAE ↓, AVR ↓)
    - AV nicking at crossings (av_nicking_index ↑)
    - Diffuse vessel narrowing (vessel_density ↓)

  Glaucoma:
    - Peripapillary vessel density loss (peripapillary_density ↓)
    - RNFL-associated vessel dropout (capillary_density ↓)
    - Inferior > superior loss asymmetry

Design contracts:
  • Simulation operates on biomarker flat-dicts, never on graph structures directly.
    This makes it independent of segmentation quality at simulation time.
  • Progression rates are per-biomarker per-day; time steps are configurable.
  • All simulated values are physically bounded (tortuosity ≥ 1, density ≥ 0, etc.).
  • Uncertainty derives from Monte Carlo noise around progression rates.
  • Every output carries explicit finite-value checks — no silent NaN propagation.
  • Counterfactual simulations subtract the intervention effect from disease rates.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ─── Disease types ────────────────────────────────────────────────────────────

class DiseaseType(Enum):
    DR           = "diabetic_retinopathy"
    HYPERTENSION = "hypertension"
    GLAUCOMA     = "glaucoma"
    CUSTOM       = "custom"


# ─── Scenario ─────────────────────────────────────────────────────────────────

@dataclass
class DiseaseScenario:
    """
    Definition of a disease progression scenario.

    Attributes:
        name:                  human-readable scenario name.
        disease_type:          DiseaseType enum.
        progression_rates:     dict of biomarker_name → rate/day (signed).
                               Positive = biomarker increases per day.
        rate_noise_scale:      standard deviation of noise applied to rates
                               (as fraction of |rate|) for Monte Carlo runs.
        intervention_factors:  dict of biomarker_name → multiplier (0.0 = halts
                               progression, 1.0 = no effect, negative = reverses).
        initial_severity:      float in [0, 1]; scales all progression rates.
        max_simulation_days:   hard cap on simulation duration.
        dt_days:               time-step size in days.
        n_steps:               number of simulation steps.
        description:           free-text clinical description.
    """
    name:               str
    disease_type:       DiseaseType
    progression_rates:  Dict[str, float]
    rate_noise_scale:   float = 0.15
    intervention_factors: Dict[str, float] = field(default_factory=dict)
    initial_severity:   float = 0.5
    max_simulation_days: float = 3650.0   # 10 years
    dt_days:            float = 30.0      # monthly steps
    n_steps:            int   = 60        # 5 years by default
    description:        str   = ""

    def effective_rates(
        self,
        apply_intervention: bool = False,
        noise_rng:          Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """
        Compute effective per-step progression rates.

        Args:
            apply_intervention: if True, multiply by intervention_factors.
            noise_rng:          if not None, add Gaussian noise to each rate.

        Returns:
            dict of biomarker_name → rate_per_step.
        """
        rates = {}
        for bm, r in self.progression_rates.items():
            effective = r * self.initial_severity * self.dt_days
            if apply_intervention and bm in self.intervention_factors:
                effective *= self.intervention_factors[bm]
            if noise_rng is not None:
                noise = noise_rng.normal(0.0, abs(effective) * self.rate_noise_scale)
                effective += noise
            rates[bm] = effective
        return rates


# ─── Output types ─────────────────────────────────────────────────────────────

@dataclass
class SimulationStep:
    """Biomarker state at one simulation time-point."""
    t_days:     float           # time since simulation start
    step:       int             # step index
    biomarkers: Dict[str, float]
    notes:      List[str] = field(default_factory=list)  # clinical flags


@dataclass
class SimulationTrajectory:
    """Full simulation time series with confidence intervals."""
    scenario:          DiseaseScenario
    initial_state:     Dict[str, float]
    steps:             List[SimulationStep]
    ci_lower:          Dict[str, List[float]]  # biomarker → per-step CI lower
    ci_upper:          Dict[str, List[float]]  # biomarker → per-step CI upper
    plausibility_score: float   # fraction of steps with physiologically valid values
    n_bounds_clamped:  int      # count of values clamped to physical bounds

    @property
    def final_state(self) -> Dict[str, float]:
        return self.steps[-1].biomarkers if self.steps else self.initial_state

    @property
    def biomarker_names(self) -> List[str]:
        return list(self.initial_state.keys())

    def get_trajectory(self, biomarker: str) -> List[float]:
        """Return time series of one biomarker across all steps."""
        return [s.biomarkers.get(biomarker, float("nan")) for s in self.steps]

    def get_times(self) -> List[float]:
        return [s.t_days for s in self.steps]


@dataclass
class CounterfactualResult:
    """
    Comparison between disease trajectory and counterfactual intervention.

    Attributes:
        baseline:     SimulationTrajectory without intervention.
        intervention: SimulationTrajectory with intervention applied.
        delta_final:  dict of biomarker_name → (final_baseline - final_intervention).
                      Positive delta = intervention reduces the biomarker (beneficial
                      for deteriorating biomarkers).
        benefit_score: fraction of adverse biomarkers where intervention helps.
        scenario:     the DiseaseScenario (with intervention_factors set).
    """
    baseline:     SimulationTrajectory
    intervention: SimulationTrajectory
    delta_final:  Dict[str, float]
    benefit_score: float
    scenario:     DiseaseScenario


@dataclass
class EnsembleResult:
    """
    Monte Carlo ensemble summary over N simulation runs.

    Attributes:
        n_samples:     number of MC runs.
        mean_final:    dict of biomarker_name → mean final value across runs.
        std_final:     dict of biomarker_name → std final value.
        ci_lower:      dict of biomarker_name → 5th percentile final value.
        ci_upper:      dict of biomarker_name → 95th percentile final value.
        risk_probability: probability of "high risk" state at final time-point.
        scenario:      scenario used.
    """
    n_samples:        int
    mean_final:       Dict[str, float]
    std_final:        Dict[str, float]
    ci_lower:         Dict[str, float]
    ci_upper:         Dict[str, float]
    risk_probability: float
    scenario:         DiseaseScenario


# ─── Physical bounds ──────────────────────────────────────────────────────────

# Minimum allowed values (physically impossible below these)
_BOUNDS_MIN: Dict[str, float] = {
    "mean_tortuosity":    1.0,    # tortuosity cannot be < 1
    "crae":               10.0,   # arteriolar caliber cannot collapse to zero
    "crve":               10.0,
    "avr":                0.3,    # AVR ≥ 0.3 in any living eye
    "vessel_density_global": 0.001,
    "fractal_dimension":  1.0,
    "capillary_density":  0.0,
    "av_nicking_index":   0.0,
}
_BOUNDS_MAX: Dict[str, float] = {
    "mean_tortuosity":    3.0,    # extreme tortuosity physiological max
    "avr":                1.5,    # AVR rarely exceeds 1.2 clinically
    "vessel_density_global": 1.0,
    "capillary_density":  1.0,
    "av_nicking_index":   1.0,
}


def _clamp(value: float, name: str) -> Tuple[float, bool]:
    """Apply physical bounds to a biomarker value. Returns (clamped, was_clamped)."""
    lo = _BOUNDS_MIN.get(name, -np.inf)
    hi = _BOUNDS_MAX.get(name, np.inf)
    clamped = float(np.clip(value, lo, hi))
    return clamped, (clamped != value)


def _add_clinical_flags(biomarkers: Dict[str, float]) -> List[str]:
    """Generate clinical warning flags for physiologically extreme values."""
    flags = []
    avr = biomarkers.get("avr", float("nan"))
    if math.isfinite(avr):
        if avr < 0.67:
            flags.append("AVR < 0.67 — Grade 2 arterial narrowing (Keith-Wagener)")
        elif avr < 0.75:
            flags.append("AVR 0.67–0.75 — Grade 1 arterial narrowing")

    tort = biomarkers.get("mean_tortuosity", float("nan"))
    if math.isfinite(tort) and tort > 1.5:
        flags.append(f"Tortuosity {tort:.2f} — elevated, possible neovascularisation")

    cap = biomarkers.get("capillary_density", float("nan"))
    if math.isfinite(cap) and cap < 0.1:
        flags.append("Capillary density < 10% — severe rarefaction / non-perfusion risk")

    return flags


# ─── Scenario presets ─────────────────────────────────────────────────────────

def build_dr_scenario(
    initial_hba1c:    float = 8.0,
    n_steps:          int   = 60,
    dt_days:          float = 30.0,
) -> DiseaseScenario:
    """
    Build a Diabetic Retinopathy progression scenario.

    Severity scales with HbA1c: higher HbA1c → faster progression.
    At HbA1c = 8%, approximately grade 1 NPDR at 5 years without control.

    Args:
        initial_hba1c: HbA1c % (6.5–14.0 range; 7.0 = well-controlled).
        n_steps:       simulation steps.
        dt_days:       days per step.

    Returns:
        DiseaseScenario.
    """
    severity = float(np.clip((initial_hba1c - 7.0) / 7.0, 0.1, 1.0))

    return DiseaseScenario(
        name=f"DR (HbA1c={initial_hba1c}%)",
        disease_type=DiseaseType.DR,
        progression_rates={
            "capillary_density":     -3.0e-5,   # dropout per day
            "mean_tortuosity":       +1.5e-5,   # tortuosity increase
            "fractal_dimension":     -8.0e-6,   # network complexity loss
            "vessel_density_global": -6.0e-6,
            "crae":                  -1.5e-4,   # arterial narrowing px/day
            "avr":                   -2.0e-5,
            "av_nicking_index":      +4.0e-5,
            "neovascularization_index": +2.0e-5,
        },
        rate_noise_scale=0.20,
        initial_severity=severity,
        n_steps=n_steps,
        dt_days=dt_days,
        description=(
            f"Diabetic retinopathy progression at HbA1c={initial_hba1c}%. "
            "Models capillary dropout, vessel narrowing, and neovascularisation."
        ),
        intervention_factors={
            "capillary_density":  0.5,   # good glucose control halves dropout
            "mean_tortuosity":    0.6,
            "av_nicking_index":   0.4,
        },
    )


def build_hypertension_scenario(
    systolic_bp_mmhg: float = 150.0,
    n_steps:          int   = 48,
    dt_days:          float = 30.0,
) -> DiseaseScenario:
    """
    Build a hypertension-driven vascular remodeling scenario.

    Args:
        systolic_bp_mmhg: systolic blood pressure in mmHg (>140 = Stage 2).
        n_steps / dt_days: simulation grid.

    Returns:
        DiseaseScenario.
    """
    excess_bp = max(systolic_bp_mmhg - 120.0, 0.0)
    severity  = float(np.clip(excess_bp / 60.0, 0.05, 1.0))

    return DiseaseScenario(
        name=f"Hypertension (SBP={systolic_bp_mmhg}mmHg)",
        disease_type=DiseaseType.HYPERTENSION,
        progression_rates={
            "crae":                  -2.5e-4,
            "avr":                   -3.0e-5,
            "av_nicking_index":      +6.0e-5,
            "av_crossing_score":     +4.0e-5,
            "mean_tortuosity":       +8.0e-6,
            "vessel_density_global": -4.0e-6,
            "mean_narrowing_focal":  +3.0e-5,
        },
        rate_noise_scale=0.15,
        initial_severity=severity,
        n_steps=n_steps,
        dt_days=dt_days,
        description=(
            f"Hypertensive retinopathy at SBP={systolic_bp_mmhg}mmHg. "
            "Models arterial narrowing, AV crossing changes, and AVR decline."
        ),
        intervention_factors={
            "crae":              0.3,   # antihypertensives reduce rate by 70%
            "avr":               0.3,
            "av_nicking_index":  0.2,
        },
    )


def build_glaucoma_scenario(
    iop_mmhg: float = 22.0,
    n_steps:  int   = 72,
    dt_days:  float = 30.0,
) -> DiseaseScenario:
    """
    Build a glaucoma-associated vascular change scenario.

    Args:
        iop_mmhg: intraocular pressure in mmHg (>21 = elevated).
        n_steps / dt_days: simulation grid.

    Returns:
        DiseaseScenario.
    """
    excess_iop = max(iop_mmhg - 15.0, 0.0)
    severity   = float(np.clip(excess_iop / 20.0, 0.05, 1.0))

    return DiseaseScenario(
        name=f"Glaucoma (IOP={iop_mmhg}mmHg)",
        disease_type=DiseaseType.GLAUCOMA,
        progression_rates={
            "peripapillary_vessel_density": -5.0e-5,
            "capillary_density":            -3.5e-5,
            "vessel_density_global":        -2.0e-6,
            "fractal_dimension":            -5.0e-6,
            "mean_tortuosity":              +3.0e-6,
        },
        rate_noise_scale=0.18,
        initial_severity=severity,
        n_steps=n_steps,
        dt_days=dt_days,
        description=(
            f"Glaucoma-associated vascular change at IOP={iop_mmhg}mmHg. "
            "Models peripapillary vessel density loss and capillary dropout."
        ),
        intervention_factors={
            "peripapillary_vessel_density": 0.35,
            "capillary_density":            0.40,
        },
    )


# ─── Physiological Simulator ──────────────────────────────────────────────────

class PhysiologicalSimulator:
    """
    Predict retinal vascular evolution under disease and intervention.

    Usage:
        sim       = PhysiologicalSimulator(seed=42)
        scenario  = build_dr_scenario(initial_hba1c=9.0, n_steps=60)
        traj      = sim.simulate(initial_biomarkers, scenario)

        # Counterfactual: what if glucose is controlled?
        cf        = sim.counterfactual(initial_biomarkers, scenario)
        print(cf.benefit_score)

        # MC ensemble for uncertainty
        ensemble  = sim.mc_ensemble(initial_biomarkers, scenario, n_samples=50)
        print(ensemble.risk_probability)
    """

    def __init__(self, seed: int = 42, n_ci_samples: int = 20):
        self.seed         = seed
        self.n_ci_samples = n_ci_samples

    def _run_single(
        self,
        initial:            Dict[str, float],
        scenario:           DiseaseScenario,
        apply_intervention: bool = False,
        noise_rng:          Optional[np.random.Generator] = None,
    ) -> Tuple[List[SimulationStep], int]:
        """
        Run one deterministic (or noisy) simulation trajectory.

        Returns:
            (steps, n_clamped) where n_clamped is the count of individual
            biomarker values that were clamped to physical bounds across all steps.
        """
        state          = {k: v for k, v in initial.items() if math.isfinite(v)}
        steps:         List[SimulationStep] = []
        n_clamped      = 0

        for step_idx in range(scenario.n_steps):
            t_days = step_idx * scenario.dt_days
            rates  = scenario.effective_rates(apply_intervention, noise_rng)

            new_state = {}
            for bm, val in state.items():
                rate = rates.get(bm, 0.0)
                new_val, was_clamped = _clamp(val + rate, bm)
                new_state[bm]       = new_val
                if was_clamped:
                    n_clamped += 1

            # Keep biomarkers not in scenario unchanged
            for bm, val in state.items():
                if bm not in new_state:
                    new_state[bm] = val

            flags = _add_clinical_flags(new_state)
            steps.append(SimulationStep(
                t_days=t_days + scenario.dt_days,
                step=step_idx,
                biomarkers=new_state,
                notes=flags,
            ))
            state = new_state

        return steps, n_clamped

    def simulate(
        self,
        initial_biomarkers: Dict[str, float],
        scenario:           DiseaseScenario,
        apply_intervention: bool = False,
    ) -> SimulationTrajectory:
        """
        Run a deterministic simulation with CI derived from MC noise.

        Args:
            initial_biomarkers: flat dict from BiomarkerReport.to_dict().
            scenario:           DiseaseScenario.
            apply_intervention: if True, apply scenario.intervention_factors.

        Returns:
            SimulationTrajectory with CI.
        """
        main_steps, main_n_clamped = self._run_single(
            initial_biomarkers, scenario,
            apply_intervention=apply_intervention,
            noise_rng=None,
        )

        # Build CI from N noisy realisations
        all_runs: List[List[SimulationStep]] = []
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_ci_samples):
            run, _ = self._run_single(
                initial_biomarkers, scenario,
                apply_intervention=apply_intervention,
                noise_rng=rng,
            )
            all_runs.append(run)

        bm_names = list(initial_biomarkers.keys())
        ci_lower: Dict[str, List[float]] = {bm: [] for bm in bm_names}
        ci_upper: Dict[str, List[float]] = {bm: [] for bm in bm_names}

        for step_idx in range(len(main_steps)):
            for bm in bm_names:
                vals = [r[step_idx].biomarkers.get(bm, float("nan"))
                        for r in all_runs
                        if step_idx < len(r)]
                finite = [v for v in vals if math.isfinite(v)]
                if finite:
                    ci_lower[bm].append(float(np.percentile(finite, 5)))
                    ci_upper[bm].append(float(np.percentile(finite, 95)))
                else:
                    ci_lower[bm].append(float("nan"))
                    ci_upper[bm].append(float("nan"))

        # Plausibility: fraction of steps where all values are physically valid
        n_valid = sum(
            1 for s in main_steps
            if all(math.isfinite(v) for v in s.biomarkers.values())
        )
        plausibility = n_valid / max(len(main_steps), 1)

        # n_clamped comes from _run_single() which counts actual clamp events.
        # The previous implementation incorrectly re-counted non-finite values
        # in the final states (which are always finite after clamping → always 0).

        return SimulationTrajectory(
            scenario=scenario,
            initial_state={k: v for k, v in initial_biomarkers.items()
                           if math.isfinite(v)},
            steps=main_steps,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            plausibility_score=plausibility,
            n_bounds_clamped=main_n_clamped,
        )

    def counterfactual(
        self,
        initial_biomarkers: Dict[str, float],
        scenario:           DiseaseScenario,
    ) -> CounterfactualResult:
        """
        Compare disease trajectory vs controlled/treated trajectory.

        Uses the intervention_factors defined in the scenario to model treatment.

        Args:
            initial_biomarkers: flat dict from BiomarkerReport.to_dict().
            scenario:           DiseaseScenario with intervention_factors set.

        Returns:
            CounterfactualResult.
        """
        baseline     = self.simulate(initial_biomarkers, scenario, apply_intervention=False)
        intervention = self.simulate(initial_biomarkers, scenario, apply_intervention=True)

        delta_final: Dict[str, float] = {}
        for bm in baseline.biomarker_names:
            b_final = baseline.final_state.get(bm, float("nan"))
            i_final = intervention.final_state.get(bm, float("nan"))
            delta_final[bm] = float(b_final - i_final) if (math.isfinite(b_final) and math.isfinite(i_final)) else float("nan")

        # Benefit score: fraction of adverse biomarkers where intervention reduces the delta
        # Inline sets to avoid circular import via rapid_twin.twin.__init__
        _ADVERSE_IF_INCREASING = {
            "mean_tortuosity", "tortuosity_std", "mean_narrowing_focal",
            "mean_narrowing_diffuse_z", "murray_violation_rate", "caliber_jump_rate",
            "tapering_violation_rate", "av_nicking_index", "av_crossing_score",
            "microaneurysm_proxy", "capillary_dropout_index", "neovascularization_index",
        }
        _ADVERSE_IF_DECREASING = {
            "crae", "crve", "avr", "fractal_dimension", "vessel_density_global",
            "peripapillary_vessel_density", "capillary_density",
            "vessel_density_zone_a", "vessel_density_zone_b",
        }
        n_adverse_helped = 0
        n_adverse        = 0
        for bm, d in delta_final.items():
            if not math.isfinite(d):
                continue
            if bm in _ADVERSE_IF_INCREASING and d > 0:   # baseline > intervention = good
                n_adverse += 1
                n_adverse_helped += 1
            elif bm in _ADVERSE_IF_DECREASING and d < 0:  # baseline < intervention = baseline lower = bad
                n_adverse += 1
                n_adverse_helped += 1

        benefit_score = float(n_adverse_helped / max(n_adverse, 1))

        return CounterfactualResult(
            baseline=baseline,
            intervention=intervention,
            delta_final=delta_final,
            benefit_score=benefit_score,
            scenario=scenario,
        )

    def mc_ensemble(
        self,
        initial_biomarkers: Dict[str, float],
        scenario:           DiseaseScenario,
        n_samples:          int = 50,
        high_risk_fn:       Optional[Callable[[Dict[str, float]], bool]] = None,
    ) -> EnsembleResult:
        """
        Monte Carlo ensemble to quantify simulation uncertainty.

        Args:
            initial_biomarkers: flat dict from BiomarkerReport.to_dict().
            scenario:           DiseaseScenario.
            n_samples:          number of MC runs.
            high_risk_fn:       optional callable (biomarker_dict) → bool;
                                determines if final state is "high risk".
                                Default: AVR < 0.67 or capillary_density < 0.1.

        Returns:
            EnsembleResult with mean, std, CI, and risk probability.
        """
        if high_risk_fn is None:
            def high_risk_fn(bm: Dict[str, float]) -> bool:
                avr = bm.get("avr", 1.0)
                cap = bm.get("capillary_density", 1.0)
                return (math.isfinite(avr) and avr < 0.67) or \
                       (math.isfinite(cap) and cap < 0.1)

        rng            = np.random.default_rng(self.seed)
        final_states:  List[Dict[str, float]] = []
        high_risk_count = 0

        for _ in range(n_samples):
            run, _ = self._run_single(initial_biomarkers, scenario, noise_rng=rng)
            final  = run[-1].biomarkers if run else initial_biomarkers
            final_states.append(final)
            if high_risk_fn(final):
                high_risk_count += 1

        bm_names = list(initial_biomarkers.keys())
        mean_final: Dict[str, float] = {}
        std_final:  Dict[str, float] = {}
        ci_lo:      Dict[str, float] = {}
        ci_hi:      Dict[str, float] = {}

        for bm in bm_names:
            vals   = [s.get(bm, float("nan")) for s in final_states]
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                mean_final[bm] = float(np.mean(finite))
                std_final[bm]  = float(np.std(finite))
                ci_lo[bm]      = float(np.percentile(finite, 5))
                ci_hi[bm]      = float(np.percentile(finite, 95))
            else:
                for d in (mean_final, std_final, ci_lo, ci_hi):
                    d[bm] = float("nan")

        return EnsembleResult(
            n_samples=n_samples,
            mean_final=mean_final,
            std_final=std_final,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            risk_probability=float(high_risk_count / max(n_samples, 1)),
            scenario=scenario,
        )
