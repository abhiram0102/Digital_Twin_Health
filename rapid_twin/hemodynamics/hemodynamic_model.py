"""
Retinal Vascular Hemodynamic Modeling.

Lightweight, graph-based approximation of retinal blood flow physics.
Does NOT require full computational fluid dynamics (CFD).
Uses Poiseuille flow, Murray's law, and graph-BFS pressure simulation.

Provides:
  - validate_murray_law()      — per-bifurcation Murray's law compliance
  - compute_vessel_resistance() — Poiseuille resistance R = 8ηL / (πr⁴)
  - estimate_flow_directions()  — root-to-leaf flow direction inference
  - HemodynamicModel            — full pressure/flow simulation
  - HemodynamicResult           — structured output
  - MurrayLawReport             — per-bifurcation Murray analysis

Physical units:
  Pressures in mmHg (gauge relative to venous return = 0).
  Radii in pixels (1 px ≈ 2–5 μm depending on camera/resolution).
  Flow in arbitrary units proportional to mmHg·px⁴ (relative, not absolute).
  Viscosity η fixed at 0.003 Pa·s (blood viscosity at physiologic conditions).

Design contracts:
  - No silent NaN propagation: every degenerate path is documented.
  - Validated against Murray's law bounds.
  - Runs without AV labels (but improves with them).
  - Results carry an explicit validity flag and issue list.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False
    warnings.warn("networkx required for hemodynamic modeling.")


# ─── Constants ────────────────────────────────────────────────────────────────

ETA         = 0.003    # blood viscosity, Pa·s
PI          = np.pi
MURRAY_EXP  = 3.0      # Murray's law exponent (= 3 for volume flow conservation)
NORMAL_AVR  = 0.67     # normal arterio-venous ratio (dimensionless)


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class MurrayLawReport:
    """Per-bifurcation Murray's law analysis."""
    n_bifurcations:       int = 0
    n_compliant:          int = 0
    n_violations:         int = 0
    mean_deviation:       float = float("nan")  # relative deviation; 0=perfect
    std_deviation:        float = float("nan")
    compliance_fraction:  float = float("nan")  # compliant / total
    violations:           List[Dict] = field(default_factory=list)


@dataclass
class HemodynamicResult:
    """
    Full output of the hemodynamic simulation.

    All per-node/per-edge dicts are keyed by node_id / (src, dst) pairs.
    """
    success:                 bool
    error:                   Optional[str]

    # Per-node
    node_pressure:           Dict[int, float]   # mmHg
    node_flow:               Dict[int, float]   # relative flow units

    # Per-edge
    edge_resistance:         Dict[Tuple[int,int], float]   # relative resistance
    edge_flow:               Dict[Tuple[int,int], float]   # relative flow
    edge_pressure_drop:      Dict[Tuple[int,int], float]   # mmHg
    edge_direction:          Dict[Tuple[int,int], int]     # +1 or -1

    # Global
    total_resistance:        float = float("nan")
    perfusion_index:         float = float("nan")  # total_flow / max_theoretical_flow
    murray_report:           Optional[MurrayLawReport] = None

    # Validity
    is_valid:                bool = False
    issues:                  List[str] = field(default_factory=list)

    @property
    def mean_pressure(self) -> float:
        if not self.node_pressure:
            return float("nan")
        return float(np.mean(list(self.node_pressure.values())))

    @property
    def max_pressure(self) -> float:
        if not self.node_pressure:
            return float("nan")
        return float(max(self.node_pressure.values()))


# ─── Murray's Law ─────────────────────────────────────────────────────────────

def validate_murray_law(
    graph,
    hierarchy=None,
    deviation_threshold: float = 0.25,
    exponent: float = MURRAY_EXP,
) -> MurrayLawReport:
    """
    Evaluate Murray's law compliance at every bifurcation.

    Murray's law: r_parent^γ = Σ r_child_i^γ   (γ = 3 for flow, 2.7 observed empirically)

    A bifurcation is "compliant" if:
      |r_parent - (Σ r_child^γ)^(1/γ)| / r_parent < deviation_threshold

    Args:
        graph:               VascularGraph.
        hierarchy:           BranchTree (optional; improves parent/child assignment).
        deviation_threshold: relative deviation threshold for compliance.
        exponent:            Murray's law exponent γ (default 3.0).

    Returns:
        MurrayLawReport.
    """
    if not _NX or not hasattr(graph, "nx_graph"):
        return MurrayLawReport()

    G     = graph.nx_graph
    nodes = {n.node_id: n for n in graph.nodes}

    # Build edge caliber lookup
    edge_cal: Dict[Tuple[int,int], float] = {}
    for e in graph.edges:
        r = max(e.mean_diameter / 2.0, 1e-6)  # convert diameter → radius
        edge_cal[(e.src, e.dst)] = r
        edge_cal[(e.dst, e.src)] = r

    # Use hierarchy to identify parent direction where available
    if hierarchy is not None and hierarchy.is_valid:
        depth = {nid: bn.depth for nid, bn in hierarchy.nodes.items()}
    else:
        depth = None

    report = MurrayLawReport()
    violations = []

    for node in graph.nodes:
        if not node.is_bifurcation:
            continue
        nb = list(G.neighbors(node.node_id))
        if len(nb) < 2:
            continue

        report.n_bifurcations += 1

        # Identify parent and children via depth
        if depth is not None:
            d_self = depth.get(node.node_id, 0)
            parent_nb  = [n for n in nb if depth.get(n, d_self) < d_self]
            children_nb = [n for n in nb if depth.get(n, d_self) >= d_self]
        else:
            # Fallback: parent = highest-caliber edge
            calibers   = {n: edge_cal.get((node.node_id, n), 0.0) for n in nb}
            parent_nb  = [max(nb, key=lambda n: calibers[n])]
            children_nb = [n for n in nb if n not in parent_nb]

        if not parent_nb or not children_nb:
            continue

        r_parent = edge_cal.get((node.node_id, parent_nb[0]), 0.0)
        if r_parent < 1e-6:
            continue

        r_children = [edge_cal.get((node.node_id, c), 0.0) for c in children_nb]
        murray_rhs = sum(r ** exponent for r in r_children) ** (1.0 / exponent)

        deviation = abs(r_parent - murray_rhs) / (r_parent + 1e-6)

        if deviation < deviation_threshold:
            report.n_compliant += 1
        else:
            report.n_violations += 1
            violations.append({
                "node_id":       node.node_id,
                "r_parent":      round(r_parent, 4),
                "murray_rhs":    round(murray_rhs, 4),
                "deviation":     round(float(deviation), 4),
            })

    deviations_all = []
    for node in graph.nodes:
        if not node.is_bifurcation:
            continue
        nb = list(G.neighbors(node.node_id))
        if len(nb) < 2:
            continue
        calibers = sorted([edge_cal.get((node.node_id, n), 0.0) for n in nb], reverse=True)
        r_p = calibers[0]
        if r_p < 1e-6:
            continue
        murray_rhs = sum(r ** exponent for r in calibers[1:]) ** (1.0 / exponent)
        deviations_all.append(abs(r_p - murray_rhs) / (r_p + 1e-6))

    if deviations_all:
        report.mean_deviation = float(np.mean(deviations_all))
        report.std_deviation  = float(np.std(deviations_all))

    report.violations = violations
    total = report.n_bifurcations
    report.compliance_fraction = report.n_compliant / total if total > 0 else float("nan")

    return report


# ─── Poiseuille Resistance ────────────────────────────────────────────────────

def compute_vessel_resistance(
    graph,
    viscosity: float = ETA,
) -> Dict[Tuple[int,int], float]:
    """
    Compute Poiseuille resistance for each vessel edge.

    R = 8 × η × L / (π × r⁴)

    where r = mean_diameter / 2 (pixels), L = length (pixels).

    Values are proportional (pixel units) — suitable for relative comparisons
    within the same image.  Cross-image comparison requires physical calibration.

    Args:
        graph:     VascularGraph.
        viscosity: blood viscosity η (Pa·s, default 0.003).

    Returns:
        Dict keyed by (src, dst) edge pair.
    """
    resistance: Dict[Tuple[int,int], float] = {}
    for e in graph.edges:
        r = max(e.mean_diameter / 2.0, 1e-6)
        L = max(e.length, 1e-6)
        R = (8.0 * viscosity * L) / (PI * r ** 4)
        resistance[(e.src, e.dst)] = float(R)
        resistance[(e.dst, e.src)] = float(R)  # undirected
    return resistance


# ─── Flow Direction Inference ─────────────────────────────────────────────────

def estimate_flow_directions(
    graph,
    hierarchy=None,
) -> Dict[Tuple[int,int], int]:
    """
    Infer flow direction on each edge: +1 = src→dst, -1 = dst→src.

    Strategy:
      1. If BranchTree is available: flow goes from lower depth to higher depth.
      2. Fallback: flow goes from higher-caliber endpoint to lower-caliber endpoint.

    Args:
        graph:     VascularGraph.
        hierarchy: BranchTree (optional, strongly recommended).

    Returns:
        Dict[(src, dst) → +1 or -1]
    """
    directions: Dict[Tuple[int,int], int] = {}

    if hierarchy is not None and hierarchy.is_valid:
        depth = {nid: bn.depth for nid, bn in hierarchy.nodes.items()}
        for e in graph.edges:
            d_src = depth.get(e.src, 0)
            d_dst = depth.get(e.dst, 0)
            directions[(e.src, e.dst)] = +1 if d_src <= d_dst else -1
            directions[(e.dst, e.src)] = -1 if d_src <= d_dst else +1
        return directions

    # Fallback: use node degree as proxy for centrality
    degree_map = {n.node_id: n.degree for n in graph.nodes}
    for e in graph.edges:
        d_src = degree_map.get(e.src, 1)
        d_dst = degree_map.get(e.dst, 1)
        directions[(e.src, e.dst)] = +1 if d_src >= d_dst else -1
        directions[(e.dst, e.src)] = -1 if d_src >= d_dst else +1

    return directions


# ─── Hemodynamic Model ────────────────────────────────────────────────────────

class HemodynamicModel:
    """
    Lightweight graph-based hemodynamic simulator for retinal vasculature.

    Simulates:
      - Pressure distribution via BFS from the root (optic disc / artery entry).
      - Poiseuille flow on each edge.
      - Murray's law compliance at bifurcations.
      - Perfusion index: total modeled flow / theoretical maximum.

    Usage:
        model = HemodynamicModel(root_pressure=80.0)
        result = model.simulate(graph, hierarchy=branch_tree)
        print(result.perfusion_index)
        print(result.murray_report.compliance_fraction)
    """

    def __init__(
        self,
        root_pressure:       float = 80.0,  # mmHg, typical central retinal artery systolic
        viscosity:           float = ETA,
        pressure_floor:      float = 0.0,   # venous return pressure (mmHg)
        validate_murray:     bool  = True,
        max_bfs_depth:       int   = 200,
    ):
        self.root_pressure   = root_pressure
        self.viscosity       = viscosity
        self.pressure_floor  = pressure_floor
        self.validate_murray = validate_murray
        self.max_bfs_depth   = max_bfs_depth

    def simulate(
        self,
        graph,
        hierarchy=None,
        root_id: Optional[int] = None,
    ) -> HemodynamicResult:
        """
        Run the full hemodynamic simulation.

        Args:
            graph:     VascularGraph.
            hierarchy: BranchTree (optional; improves flow direction + Murray).
            root_id:   explicit root node id; auto-selected if None.

        Returns:
            HemodynamicResult.
        """
        empty = HemodynamicResult(
            success=False, error=None,
            node_pressure={}, node_flow={},
            edge_resistance={}, edge_flow={},
            edge_pressure_drop={}, edge_direction={},
        )

        if not graph.nodes:
            empty.error = "Empty graph — no nodes to simulate."
            empty.issues.append(empty.error)
            return empty

        if not _NX or not hasattr(graph, "nx_graph"):
            empty.error = "networkx required for hemodynamic simulation."
            empty.issues.append(empty.error)
            return empty

        G      = graph.nx_graph
        issues: List[str] = []

        # ── Step 1: Resistance ────────────────────────────────────────────────
        resistance = compute_vessel_resistance(graph, self.viscosity)

        # ── Step 2: Root selection ────────────────────────────────────────────
        if root_id is None:
            if hierarchy is not None and hierarchy.is_valid:
                root_id = hierarchy.root_id
            else:
                # Fallback: highest-degree node
                root_id = max(graph.nodes, key=lambda n: n.degree).node_id

        if root_id not in {n.node_id for n in graph.nodes}:
            empty.error = f"root_id={root_id} not in graph."
            empty.issues.append(empty.error)
            return empty

        # ── Step 3: Flow directions ───────────────────────────────────────────
        directions = estimate_flow_directions(graph, hierarchy)

        # ── Step 4: BFS pressure simulation ──────────────────────────────────
        node_pressure: Dict[int, float] = {root_id: self.root_pressure}
        node_flow:     Dict[int, float] = {}
        edge_flow:     Dict[Tuple[int,int], float] = {}
        edge_pdrop:    Dict[Tuple[int,int], float] = {}

        # BFS queue: (node_id, depth)
        queue: deque = deque([(root_id, 0)])
        visited: set = {root_id}

        while queue:
            nid, depth = queue.popleft()

            if depth > self.max_bfs_depth:
                issues.append(
                    f"BFS depth cap {self.max_bfs_depth} reached at node {nid}. "
                    "Pressures beyond this depth are not simulated."
                )
                break

            p_here = node_pressure.get(nid, self.pressure_floor)

            for nb in G.neighbors(nid):
                edge_key = (nid, nb)
                R = resistance.get(edge_key, resistance.get((nb, nid), 1.0))

                # Direction-aware pressure drop
                dir_flag = directions.get(edge_key, +1)
                if dir_flag == +1:
                    # Flow from nid → nb
                    r_min = max(e.mean_diameter / 2 for e in graph.edges
                                if (e.src == nid and e.dst == nb)
                                or (e.src == nb and e.dst == nid)) / 2.0 \
                            if graph.edges else 1.0
                    # Q = ΔP / R  (Hagen-Poiseuille)
                    # Pressure at nb = max(pressure_floor, p_here - Q*R)
                    # We compute Q iteratively:
                    # Q ∝ (p_here - p_nb) / R → set p_nb via downstream resistance
                    # Simplified: equal pressure drop per unit resistance
                    downstream_R = self._downstream_resistance(G, nb, resistance, visited)
                    total_R = R + downstream_R
                    if total_R < 1e-30:
                        total_R = 1e-30
                    Q = (p_here - self.pressure_floor) / total_R
                    p_drop = Q * R
                    p_nb   = max(self.pressure_floor, p_here - p_drop)

                    edge_flow[(nid, nb)]  = float(Q)
                    edge_pdrop[(nid, nb)] = float(p_drop)
                else:
                    p_nb   = p_here
                    p_drop = 0.0
                    Q      = 0.0

                if nb not in visited:
                    node_pressure[nb] = p_nb
                    visited.add(nb)
                    queue.append((nb, depth + 1))
                elif nb in node_pressure:
                    # Update pressure if we find a higher-pressure path
                    node_pressure[nb] = max(node_pressure[nb], p_nb)

        # ── Step 5: Node flows (sum of incident edge flows) ───────────────────
        for nid in G.nodes():
            inflow = sum(
                edge_flow.get((nb, nid), 0.0)
                for nb in G.neighbors(nid)
                if directions.get((nb, nid), +1) == +1
            )
            node_flow[nid] = float(inflow)

        # ── Step 6: Total resistance and perfusion index ──────────────────────
        total_Q = sum(
            edge_flow.get((nid, nb), 0.0)
            for nid in G.nodes()
            for nb in G.neighbors(nid)
            if directions.get((nid, nb), +1) == +1
        )
        # Maximum theoretical flow if all resistance = min observed resistance
        min_R = min(resistance.values()) if resistance else 1.0
        max_Q = self.root_pressure / min_R if min_R > 0 else 0.0
        perfusion = float(total_Q / (max_Q + 1e-8))

        # ── Step 7: Murray's law ──────────────────────────────────────────────
        murray_report = None
        if self.validate_murray:
            try:
                murray_report = validate_murray_law(graph, hierarchy)
                if murray_report.n_violations > 0:
                    issues.append(
                        f"Murray's law: {murray_report.n_violations} violations "
                        f"({murray_report.compliance_fraction:.1%} compliant)."
                    )
            except Exception as exc:
                issues.append(f"Murray validation failed: {exc}")

        # ── Sanity checks ─────────────────────────────────────────────────────
        for nid, p in node_pressure.items():
            if not np.isfinite(p):
                node_pressure[nid] = self.pressure_floor
                issues.append(f"Non-finite pressure at node {nid} — clamped to floor.")

        return HemodynamicResult(
            success=True,
            error=None,
            node_pressure=node_pressure,
            node_flow=node_flow,
            edge_resistance={k: float(v) for k, v in resistance.items() if len(k) == 2},
            edge_flow={k: float(v) for k, v in edge_flow.items()},
            edge_pressure_drop={k: float(v) for k, v in edge_pdrop.items()},
            edge_direction=directions,
            total_resistance=float(min_R),
            perfusion_index=float(np.clip(perfusion, 0.0, 1.0)),
            murray_report=murray_report,
            is_valid=True,
            issues=issues,
        )

    def _downstream_resistance(
        self,
        G: "nx.Graph",
        start: int,
        resistance: Dict[Tuple[int,int], float],
        exclude: set,
        max_depth: int = 10,
    ) -> float:
        """
        Estimate total downstream resistance from a node via BFS.

        Parallel flow paths combine as the harmonic sum of resistances:
            1/R_total = Σ (1/R_i)
        so R_total = 1 / Σ(1/R_i).

        Using arithmetic mean (the old implementation) underestimates
        resistance for high-resistance branches and overestimates for
        low-resistance branches, violating Kirchhoff's current law.
        """
        inv_sum = 0.0
        count   = 0
        queue: deque = deque([(start, 0)])
        seen: set = {start}

        while queue:
            nid, d = queue.popleft()
            if d >= max_depth:
                continue
            for nb in G.neighbors(nid):
                if nb in seen or nb in exclude:
                    continue
                seen.add(nb)
                R = resistance.get((nid, nb), resistance.get((nb, nid), 1.0))
                if R > 1e-30:
                    inv_sum += 1.0 / R
                count += 1
                queue.append((nb, d + 1))

        if inv_sum < 1e-30:
            # No downstream paths found — return a large sentinel resistance
            return 1.0 / 1e-30 if count == 0 else 1.0
        return 1.0 / inv_sum

    def print_summary(self, result: HemodynamicResult):
        """Print a formatted summary of the hemodynamic simulation."""
        print(f"\n{'─'*60}")
        print("  Hemodynamic Simulation Summary")
        print(f"{'─'*60}")
        if not result.success:
            print(f"  FAILED: {result.error}")
            return

        print(f"  Nodes simulated    : {len(result.node_pressure)}")
        print(f"  Mean pressure      : {result.mean_pressure:.2f} mmHg")
        print(f"  Max pressure       : {result.max_pressure:.2f} mmHg")
        print(f"  Perfusion index    : {result.perfusion_index:.4f}")
        print(f"  Total edges        : {len(result.edge_flow)}")

        if result.murray_report:
            mr = result.murray_report
            print(f"\n  Murray's Law:")
            print(f"    Bifurcations     : {mr.n_bifurcations}")
            print(f"    Compliant        : {mr.n_compliant}  "
                  f"({mr.compliance_fraction:.1%})")
            print(f"    Mean deviation   : {mr.mean_deviation:.4f}")

        if result.issues:
            print(f"\n  Issues ({len(result.issues)}):")
            for issue in result.issues[:5]:
                print(f"    ⚠ {issue}")

        print(f"{'─'*60}\n")
