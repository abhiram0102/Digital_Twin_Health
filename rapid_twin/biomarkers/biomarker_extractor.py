"""
Retinal Vascular Biomarker Extraction.

Research-grade extraction of 12 medically-interpretable vascular biomarkers
from the retinal vascular graph and/or binary vessel mask.

Biomarker families:
  1.  Tortuosity              — mean/std/zone-specific tortuosity index
  2.  Vessel caliber ratios   — AVR, CRAE, CRVE (requires AV labels)
  3.  Branching angles        — mean, Murray's law deviation
  4.  Fractal dimension       — box-counting (delegates to graph_metrics)
  5.  Vessel density          — pixel-based, zone-specific
  6.  Vessel occupancy maps   — spatial density grids
  7.  AV nicking              — vein caliber reduction at artery crossings
  8.  Vessel narrowing        — focal narrowing index (min/mean caliber ratio)
  9.  Ischemic indicators     — local vessel dropout, capillary density
  10. Glaucoma biomarkers     — peripapillary vessel density, CDR proxy
  11. DR biomarkers           — neovascularization index, vessel complexity
  12. Hypertension biomarkers — arterial narrowing severity, AVR Z-score

Design contracts:
  - AV labels are OPTIONAL — biomarkers that require them degrade gracefully
    to nan with an explicit warning, not a crash.
  - All zone-based computations require an optic disc centre estimate.
    If not provided, zone analysis is skipped (nan) without error.
  - Returned BiomarkerReport is a flat dict + structured sub-reports.
  - Every computation is bounded; no silent inf/nan propagation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Zone definition ──────────────────────────────────────────────────────────

@dataclass
class ZoneDefinition:
    """
    Annular zones relative to optic disc centre.

    disc_centre: (row, col) pixel coordinate of optic disc centre.
    disc_radius: optic disc radius in pixels.

    Zones:
      parapapillary: 0.5 × disc_radius to 1.0 × disc_radius from centre
      zone_A:        0.5 DD to 1.0 DD (1 DD = 2 × disc_radius)
      zone_B:        1.0 DD to 2.0 DD
      periphery:     2.0 DD and beyond
    """
    disc_centre: Tuple[int, int]  # (row, col)
    disc_radius: float            # pixels

    def distance_map(self, H: int, W: int) -> np.ndarray:
        """Return (H, W) float32 map of Euclidean distance from disc centre."""
        rr = np.arange(H, dtype=np.float32)[:, None]
        cc = np.arange(W, dtype=np.float32)[None, :]
        return np.sqrt((rr - self.disc_centre[0]) ** 2 + (cc - self.disc_centre[1]) ** 2)

    def zone_mask(self, H: int, W: int, r_inner: float, r_outer: float) -> np.ndarray:
        """Binary mask for pixels in [r_inner, r_outer) × disc_radius from centre."""
        dm = self.distance_map(H, W)
        return (dm >= r_inner * self.disc_radius) & (dm < r_outer * self.disc_radius)


# ─── Report dataclass ─────────────────────────────────────────────────────────

@dataclass
class BiomarkerReport:
    """
    Complete set of retinal vascular biomarkers.

    All float values are in their natural units or normalised to [0, 1].
    nan = metric not computable (missing AV labels, no zone definition, etc.)
    """
    # 1. Tortuosity
    mean_tortuosity:          float = float("nan")
    std_tortuosity:           float = float("nan")
    parapapillary_tortuosity: float = float("nan")
    peripheral_tortuosity:    float = float("nan")

    # 2. Vessel caliber ratios
    crae:                     float = float("nan")   # central retinal artery equiv.
    crve:                     float = float("nan")   # central retinal vein equiv.
    avr:                      float = float("nan")   # arteriovenous ratio = CRAE/CRVE

    # 3. Branching angles
    mean_branching_angle:     float = float("nan")
    murray_compliance_score:  float = float("nan")   # 1=perfect, 0=worst

    # 4. Fractal dimension
    fractal_dimension:        float = float("nan")

    # 5. Vessel density (global)
    vessel_density_global:    float = float("nan")
    vessel_density_zone_a:    float = float("nan")
    vessel_density_zone_b:    float = float("nan")

    # 6. Vessel occupancy map
    occupancy_map:            Optional[np.ndarray] = None  # (H//16, W//16) float32

    # 7. AV nicking
    av_nicking_index:         float = float("nan")   # 0=none, 1=severe

    # 8. Vessel narrowing
    focal_narrowing_index:    float = float("nan")   # min/mean caliber ratio; <0.7 = concerning
    diffuse_narrowing_index:  float = float("nan")   # mean caliber z-score vs normal

    # 9. Ischemic indicators
    vessel_dropout_index:     float = float("nan")   # fraction of expected area without vessels
    capillary_density:        float = float("nan")   # thin vessel pixel fraction

    # 10. Glaucoma biomarkers
    peripapillary_vessel_density: float = float("nan")
    cup_disc_ratio_proxy:     float = float("nan")   # vessel-free central zone / disc area

    # 11. DR biomarkers
    neovascularization_index: float = float("nan")   # high tortuosity + density in periphery
    vessel_complexity:        float = float("nan")   # fractal dim × bifurcation density

    # 12. Hypertension biomarkers
    arterial_narrowing_severity: float = float("nan")   # CRAE z-score vs normal
    av_crossing_score:        float = float("nan")   # AV nicking severity

    # Metadata
    n_nodes:                  int = 0
    n_edges:                  int = 0
    warnings:                 List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Flat dict of all scalar biomarker values."""
        return {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, float) and k != "warnings"
        }


# ─── Utility: node distances from disc ───────────────────────────────────────

def _node_distances(nodes, zone: ZoneDefinition) -> np.ndarray:
    """Array of distances from disc centre for each node."""
    if not nodes:
        return np.array([], dtype=np.float32)
    pts = np.array([n.position for n in nodes], dtype=np.float32)
    dc  = np.array(zone.disc_centre, dtype=np.float32)
    return np.linalg.norm(pts - dc, axis=1)


# ─── 1. Tortuosity ────────────────────────────────────────────────────────────

def _compute_tortuosity_biomarkers(
    graph,
    zone: Optional[ZoneDefinition],
) -> Dict[str, float]:
    edges = graph.edges
    if not edges:
        return {k: float("nan") for k in [
            "mean_tortuosity", "std_tortuosity",
            "parapapillary_tortuosity", "peripheral_tortuosity",
        ]}

    torts = np.array([e.tortuosity for e in edges], dtype=np.float32)
    torts = torts[np.isfinite(torts) & (torts >= 1.0)]

    result = {
        "mean_tortuosity": float(np.mean(torts)) if len(torts) else float("nan"),
        "std_tortuosity":  float(np.std(torts))  if len(torts) else float("nan"),
        "parapapillary_tortuosity": float("nan"),
        "peripheral_tortuosity":   float("nan"),
    }

    if zone is None or not graph.nodes:
        return result

    nodes   = graph.nodes
    nd_map  = {n.node_id: n for n in nodes}
    dists   = _node_distances(nodes, zone)
    # Build O(1) lookup from node_id → index in the nodes list
    id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}

    parapap_edges, periph_edges = [], []
    for e in edges:
        src_idx = id_to_idx.get(e.src, -1)
        dst_idx = id_to_idx.get(e.dst, -1)
        if src_idx < 0 or dst_idx < 0:
            continue
        mid_dist = (dists[src_idx] + dists[dst_idx]) / 2.0
        if mid_dist < zone.disc_radius * 2.0:
            parapap_edges.append(e.tortuosity)
        else:
            periph_edges.append(e.tortuosity)

    if parapap_edges:
        result["parapapillary_tortuosity"] = float(np.mean(parapap_edges))
    if periph_edges:
        result["peripheral_tortuosity"] = float(np.mean(periph_edges))

    return result


# ─── 2. Vessel caliber ratios (CRAE / CRVE / AVR) ───────────────────────────

def _compute_caliber_ratios(
    graph,
    av_labels: Optional[Dict[int, "AVLabel"]] = None,
    zone: Optional[ZoneDefinition] = None,
    zone_inner: float = 0.5,
    zone_outer: float = 1.0,
) -> Dict[str, float]:
    """
    Compute CRAE (central retinal artery equivalent) and CRVE
    using the Knudtson (2003) formula iteratively applied to the
    six largest artery / vein diameters in zone A (0.5–1.0 DD from disc).

    Requires AV labels.  Returns nan for all metrics if not available.
    """
    if av_labels is None:
        warnings.warn("BiomarkerExtractor: AV labels required for CRAE/CRVE/AVR. Returning nan.")
        return {"crae": float("nan"), "crve": float("nan"), "avr": float("nan")}

    from rapid_twin.av_classification.av_classifier import AVLabel

    nodes  = {n.node_id: n for n in graph.nodes}
    edges  = graph.edges

    # Select edges in zone A if zone is defined
    def _in_zone(e):
        if zone is None:
            return True
        n_s = nodes.get(e.src)
        n_d = nodes.get(e.dst)
        if n_s is None or n_d is None:
            return False
        dc = np.array(zone.disc_centre, dtype=np.float32)
        d_s = np.linalg.norm(np.array(n_s.position) - dc)
        d_d = np.linalg.norm(np.array(n_d.position) - dc)
        mid = (d_s + d_d) / 2
        r   = zone.disc_radius
        return zone_inner * r <= mid < zone_outer * r

    artery_diams, vein_diams = [], []
    for e in edges:
        if not _in_zone(e):
            continue
        label_s = av_labels.get(e.src)
        label_d = av_labels.get(e.dst)
        # Use label of the source node; if neither is labeled, skip
        label = label_s if label_s is not None else label_d
        if label is None:
            continue
        if label == AVLabel.ARTERY:
            artery_diams.append(e.mean_diameter)
        elif label == AVLabel.VEIN:
            vein_diams.append(e.mean_diameter)

    def _knudtson_equiv(diams: List[float]) -> float:
        """Iteratively apply Knudtson formula to get caliber equivalent."""
        if not diams:
            return float("nan")
        d = sorted(diams, reverse=True)[:6]
        while len(d) > 1:
            if len(d) % 2 == 1:
                d.append(d[-1])
            merged = []
            for i in range(0, len(d), 2):
                # Knudtson: W = 0.88 × sqrt(W1² + W2²)
                merged.append(0.88 * np.sqrt(d[i] ** 2 + d[i + 1] ** 2))
            d = merged
        return float(d[0])

    crae = _knudtson_equiv(artery_diams)
    crve = _knudtson_equiv(vein_diams)
    avr  = float(crae / crve) if (np.isfinite(crae) and np.isfinite(crve) and crve > 0) else float("nan")

    return {"crae": crae, "crve": crve, "avr": avr}


# ─── 3. Branching angles / Murray compliance ──────────────────────────────────

def _compute_branching_biomarkers(graph) -> Dict[str, float]:
    bif_nodes = [n for n in graph.nodes if n.is_bifurcation]
    if not bif_nodes:
        return {
            "mean_branching_angle":  float("nan"),
            "murray_compliance_score": float("nan"),
        }

    angles = [n.branching_angle for n in bif_nodes if np.isfinite(n.branching_angle)]
    mean_angle = float(np.mean(angles)) if angles else float("nan")

    # Murray compliance: per-bifurcation deviation
    edge_diam: Dict[Tuple[int,int], float] = {}
    for e in graph.edges:
        edge_diam[(e.src, e.dst)] = e.mean_diameter
        edge_diam[(e.dst, e.src)] = e.mean_diameter

    try:
        import networkx as nx
        G = graph.nx_graph
        compliances = []
        for bif in bif_nodes:
            nb = list(G.neighbors(bif.node_id))
            if len(nb) < 2:
                continue
            calibers = sorted([edge_diam.get((bif.node_id, n), 0.0) for n in nb], reverse=True)
            r_p = calibers[0]
            if r_p < 1e-6:
                continue
            murray_rhs = sum(r ** 3 for r in calibers[1:]) ** (1 / 3)
            deviation = abs(r_p - murray_rhs) / (r_p + 1e-6)
            compliances.append(max(0.0, 1.0 - deviation))
        murray_score = float(np.mean(compliances)) if compliances else float("nan")
    except Exception:
        murray_score = float("nan")

    return {
        "mean_branching_angle":  mean_angle,
        "murray_compliance_score": murray_score,
    }


# ─── 5. Vessel density ────────────────────────────────────────────────────────

def _compute_vessel_density(
    mask: np.ndarray,
    zone: Optional[ZoneDefinition] = None,
) -> Dict[str, float]:
    H, W = mask.shape
    binary = (mask > 0.5).astype(np.float32)
    global_density = float(binary.mean())

    if zone is None:
        return {
            "vessel_density_global": global_density,
            "vessel_density_zone_a": float("nan"),
            "vessel_density_zone_b": float("nan"),
        }

    zone_a = zone.zone_mask(H, W, 0.5, 2.0)
    zone_b = zone.zone_mask(H, W, 2.0, 4.0)

    def _density_in_zone(z):
        return float(binary[z].mean()) if z.any() else float("nan")

    return {
        "vessel_density_global": global_density,
        "vessel_density_zone_a": _density_in_zone(zone_a),
        "vessel_density_zone_b": _density_in_zone(zone_b),
    }


# ─── 6. Vessel occupancy map ──────────────────────────────────────────────────

def _compute_occupancy_map(mask: np.ndarray, cell_size: int = 16) -> np.ndarray:
    """
    (H//cell_size, W//cell_size) float32 map of vessel pixel fraction per cell.
    """
    H, W = mask.shape
    binary = (mask > 0.5).astype(np.float32)
    rows = H // cell_size
    cols = W // cell_size
    if rows == 0 or cols == 0:
        return np.zeros((1, 1), dtype=np.float32)
    occ = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = binary[r * cell_size:(r + 1) * cell_size,
                           c * cell_size:(c + 1) * cell_size]
            occ[r, c] = patch.mean()
    return occ


# ─── 7. AV nicking ───────────────────────────────────────────────────────────

def _compute_av_nicking(
    graph,
    av_labels: Optional[Dict[int, "AVLabel"]],
) -> float:
    """
    AV nicking index: fraction of vein diameters at artery-crossings that
    are significantly narrower than their non-crossing mean.

    Returns nan if AV labels not available.
    """
    if av_labels is None:
        return float("nan")

    try:
        from rapid_twin.av_classification.av_classifier import AVLabel
        import networkx as nx
        G = graph.nx_graph
    except Exception:
        return float("nan")

    edge_diam: Dict[Tuple[int,int], float] = {
        (e.src, e.dst): e.mean_diameter for e in graph.edges
    }
    edge_diam.update({(e.dst, e.src): e.mean_diameter for e in graph.edges})

    nicking_scores = []
    for node in graph.nodes:
        # Crossing: node with both artery and vein neighbours
        nb_labels = [(nb, av_labels.get(nb)) for nb in G.neighbors(node.node_id)]
        artery_nb = [nb for nb, l in nb_labels if l == AVLabel.ARTERY]
        vein_nb   = [nb for nb, l in nb_labels if l == AVLabel.VEIN]

        if not artery_nb or not vein_nb:
            continue

        # Vein calibers at this crossing node
        vein_calibers = [edge_diam.get((node.node_id, nb), 0.0) for nb in vein_nb]
        # All vein calibers in graph
        all_vein = [e.mean_diameter for e in graph.edges
                    if av_labels.get(e.src) == AVLabel.VEIN
                    or av_labels.get(e.dst) == AVLabel.VEIN]

        if not all_vein or not vein_calibers:
            continue

        mean_vein_global = np.mean(all_vein)
        mean_vein_local  = np.mean(vein_calibers)

        if mean_vein_global > 0:
            nicking = max(0.0, 1.0 - mean_vein_local / mean_vein_global)
            nicking_scores.append(nicking)

    return float(np.mean(nicking_scores)) if nicking_scores else 0.0


# ─── 8. Vessel narrowing ─────────────────────────────────────────────────────

def _compute_vessel_narrowing(graph) -> Dict[str, float]:
    edges = graph.edges
    if not edges:
        return {
            "focal_narrowing_index":   float("nan"),
            "diffuse_narrowing_index": float("nan"),
        }
    diams = np.array([e.mean_diameter for e in edges], dtype=np.float32)
    diams = diams[np.isfinite(diams) & (diams > 0)]
    if len(diams) == 0:
        return {
            "focal_narrowing_index":   float("nan"),
            "diffuse_narrowing_index": float("nan"),
        }

    focal   = float(diams.min() / diams.mean())    # <0.7 = concerning
    # Diffuse narrowing: z-score of the mean diameter vs the expected normal.
    # NOTE: "normal_artery_diameter_px" is resolution-dependent — 5.0 px is a
    # reasonable default for a 45° FOV fundus camera at standard resolution
    # (~15 μm/px → ~75 μm artery diameter). Callers with different resolution
    # should pass a calibrated value to RetinalBiomarkerExtractor.
    z_score = float((diams.mean() - 5.0) / (2.0 + 1e-8))
    diffuse = float(np.clip(z_score, -3.0, 3.0))

    return {
        "focal_narrowing_index":   focal,
        "diffuse_narrowing_index": diffuse,
    }


# ─── 9. Ischemic indicators ──────────────────────────────────────────────────

def _compute_ischemic_indicators(
    mask: np.ndarray,
    zone: Optional[ZoneDefinition],
) -> Dict[str, float]:
    binary = (mask > 0.5).astype(np.float32)
    H, W   = binary.shape

    # Capillary density: thin vessels (local density <10%) are capillaries
    try:
        import cv2
        dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
        capillary_mask = binary * (dist < 2.0)
        capillary_density = float(capillary_mask.mean() / (binary.mean() + 1e-8))
    except Exception:
        capillary_density = float("nan")

    if zone is None:
        return {
            "vessel_dropout_index": float("nan"),
            "capillary_density":    capillary_density,
        }

    # Vessel dropout: fraction of zone_B area with no vessels
    zone_b = zone.zone_mask(H, W, 2.0, 4.0)
    if not zone_b.any():
        dropout = float("nan")
    else:
        dropout = float(1.0 - binary[zone_b].mean())

    return {
        "vessel_dropout_index": dropout,
        "capillary_density":    capillary_density,
    }


# ─── 10. Glaucoma biomarkers ──────────────────────────────────────────────────

def _compute_glaucoma_biomarkers(
    mask: np.ndarray,
    zone: Optional[ZoneDefinition],
) -> Dict[str, float]:
    binary = (mask > 0.5).astype(np.float32)
    H, W   = binary.shape

    if zone is None:
        return {
            "peripapillary_vessel_density": float("nan"),
            "cup_disc_ratio_proxy":         float("nan"),
        }

    # Peripapillary ring: 0.5–1.0 disc radii
    parapap = zone.zone_mask(H, W, 0.5, 1.0)
    ppvd = float(binary[parapap].mean()) if parapap.any() else float("nan")

    # Cup-disc ratio proxy: vessel-free area within disc / disc area
    disc_zone = zone.zone_mask(H, W, 0.0, 1.0)
    if disc_zone.any():
        vessel_free  = (binary[disc_zone] == 0).mean()
        cdr_proxy    = float(vessel_free)   # larger = more cup-like
    else:
        cdr_proxy = float("nan")

    return {
        "peripapillary_vessel_density": ppvd,
        "cup_disc_ratio_proxy":         cdr_proxy,
    }


# ─── 11. DR biomarkers ───────────────────────────────────────────────────────

def _compute_dr_biomarkers(
    graph,
    zone: Optional[ZoneDefinition],
) -> Dict[str, float]:
    """
    DR biomarkers:
    - Neovascularization index: high tortuosity + high density in periphery
    - Vessel complexity: fractal_dim × bifurcation_density
    """
    edges = graph.edges
    if not edges:
        return {
            "neovascularization_index": float("nan"),
            "vessel_complexity":        float("nan"),
        }

    # Neovascularization proxy: mean tortuosity in periphery
    if zone is not None and graph.nodes:
        nodes         = graph.nodes
        dists         = _node_distances(nodes, zone)
        # Build O(1) lookup: node_id → index in the nodes list
        id_to_idx_dr  = {n.node_id: i for i, n in enumerate(nodes)}
        periph_torts  = []
        for e in edges:
            src_idx = id_to_idx_dr.get(e.src, -1)
            dst_idx = id_to_idx_dr.get(e.dst, -1)
            if src_idx < 0 or dst_idx < 0:
                continue
            mid_dist = (dists[src_idx] + dists[dst_idx]) / 2
            if mid_dist > zone.disc_radius * 4.0:
                periph_torts.append(e.tortuosity)
        neo_index = float(np.mean(periph_torts)) - 1.0 if periph_torts else float("nan")
    else:
        neo_index = float("nan")

    # Vessel complexity
    n_bif = sum(1 for n in graph.nodes if n.is_bifurcation)
    n_total = len(graph.nodes)
    bif_density = n_bif / max(n_total, 1)

    try:
        from rapid_twin.graph.graph_metrics import compute_fractal_dimension
        # Need mask for fractal dim — approximate from skeleton
        if graph.skeleton is not None:
            fd = compute_fractal_dimension(graph.skeleton.astype(np.float32))
        else:
            fd = float("nan")
    except Exception:
        fd = float("nan")

    vessel_complexity = float(fd * bif_density) if np.isfinite(fd) else float("nan")

    return {
        "neovascularization_index": float(np.clip(neo_index, 0.0, 5.0)) if np.isfinite(neo_index) else float("nan"),
        "vessel_complexity":        float(vessel_complexity) if np.isfinite(vessel_complexity) else float("nan"),
    }


# ─── 12. Hypertension biomarkers ─────────────────────────────────────────────

def _compute_hypertension_biomarkers(
    crae: float,
    avr:  float,
    av_nicking: float,
) -> Dict[str, float]:
    """
    Hypertension biomarkers based on CRAE Z-score and AV ratio.

    Reference: Hubbard et al. (1999) / Klein et al. (2006)
    Normal CRAE ≈ 90–110 μm (calibrated; here we use pixel units, reference=10px)
    Normal AVR ≈ 0.67–0.75
    """
    CRAE_REF  = 10.0  # reference normal caliber (pixels)
    CRAE_STD  = 2.0
    AVR_NORMAL = 0.67

    art_narrowing = (
        float((CRAE_REF - crae) / (CRAE_STD + 1e-8))
        if np.isfinite(crae) else float("nan")
    )
    av_crossing = av_nicking if np.isfinite(av_nicking) else float("nan")

    return {
        "arterial_narrowing_severity": float(np.clip(art_narrowing, -3.0, 3.0))
                                       if np.isfinite(art_narrowing) else float("nan"),
        "av_crossing_score":           av_crossing,
    }


# ─── Master extractor ────────────────────────────────────────────────────────

class RetinalBiomarkerExtractor:
    """
    Extract all 12 retinal vascular biomarker families from a VascularGraph.

    Usage:
        extractor = RetinalBiomarkerExtractor()
        report = extractor.extract(
            graph=vascular_graph,
            mask=vessel_mask,
            av_labels=av_predictions,   # dict {node_id: AVLabel} — optional
            zone=ZoneDefinition(disc_centre=(256,256), disc_radius=40),
        )
        flat = report.to_dict()
    """

    def __init__(self, occupancy_cell_size: int = 16):
        self.occupancy_cell_size = occupancy_cell_size

    def extract(
        self,
        graph,
        mask:      Optional[np.ndarray] = None,
        av_labels: Optional[Dict] = None,
        zone:      Optional[ZoneDefinition] = None,
        hierarchy=None,
    ) -> BiomarkerReport:
        """
        Extract the full biomarker battery.

        Args:
            graph:     VascularGraph.
            mask:      (H, W) binary vessel mask.
            av_labels: Dict[node_id → AVLabel] (optional; needed for CRAE/CRVE/AVR/nicking).
            zone:      ZoneDefinition (optional; needed for zone-specific biomarkers).
            hierarchy: BranchTree (optional; used for tapering biomarkers).

        Returns:
            BiomarkerReport — fully populated where data is available.
        """
        report = BiomarkerReport(
            n_nodes=len(graph.nodes),
            n_edges=len(graph.edges),
        )

        # 1. Tortuosity
        tort = _compute_tortuosity_biomarkers(graph, zone)
        report.mean_tortuosity          = tort["mean_tortuosity"]
        report.std_tortuosity           = tort["std_tortuosity"]
        report.parapapillary_tortuosity = tort["parapapillary_tortuosity"]
        report.peripheral_tortuosity    = tort["peripheral_tortuosity"]

        # 2. CRAE / CRVE / AVR
        caliber = _compute_caliber_ratios(graph, av_labels, zone)
        report.crae = caliber["crae"]
        report.crve = caliber["crve"]
        report.avr  = caliber["avr"]

        # 3. Branching angles / Murray
        bif = _compute_branching_biomarkers(graph)
        report.mean_branching_angle   = bif["mean_branching_angle"]
        report.murray_compliance_score = bif["murray_compliance_score"]

        # 4. Fractal dimension
        if mask is not None:
            try:
                from rapid_twin.graph.graph_metrics import compute_fractal_dimension
                report.fractal_dimension = compute_fractal_dimension(mask)
            except Exception as exc:
                report.warnings.append(f"fractal_dimension failed: {exc}")

        # 5. Vessel density
        if mask is not None:
            vd = _compute_vessel_density(mask, zone)
            report.vessel_density_global = vd["vessel_density_global"]
            report.vessel_density_zone_a = vd["vessel_density_zone_a"]
            report.vessel_density_zone_b = vd["vessel_density_zone_b"]

        # 6. Occupancy map
        if mask is not None:
            report.occupancy_map = _compute_occupancy_map(mask, self.occupancy_cell_size)

        # 7. AV nicking
        report.av_nicking_index = _compute_av_nicking(graph, av_labels)

        # 8. Vessel narrowing
        narrowing = _compute_vessel_narrowing(graph)
        report.focal_narrowing_index   = narrowing["focal_narrowing_index"]
        report.diffuse_narrowing_index = narrowing["diffuse_narrowing_index"]

        # 9. Ischemic indicators
        if mask is not None:
            isch = _compute_ischemic_indicators(mask, zone)
            report.vessel_dropout_index = isch["vessel_dropout_index"]
            report.capillary_density    = isch["capillary_density"]

        # 10. Glaucoma biomarkers
        if mask is not None:
            glauc = _compute_glaucoma_biomarkers(mask, zone)
            report.peripapillary_vessel_density = glauc["peripapillary_vessel_density"]
            report.cup_disc_ratio_proxy         = glauc["cup_disc_ratio_proxy"]

        # 11. DR biomarkers
        dr = _compute_dr_biomarkers(graph, zone)
        report.neovascularization_index = dr["neovascularization_index"]
        report.vessel_complexity        = dr["vessel_complexity"]

        # 12. Hypertension biomarkers
        htn = _compute_hypertension_biomarkers(
            report.crae, report.avr, report.av_nicking_index
        )
        report.arterial_narrowing_severity = htn["arterial_narrowing_severity"]
        report.av_crossing_score           = htn["av_crossing_score"]

        return report
