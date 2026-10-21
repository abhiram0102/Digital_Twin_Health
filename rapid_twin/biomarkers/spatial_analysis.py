"""
Spatial Retinal Vascular Analysis — Stage 1, Functional Vascular Characterization.

Provides optic-disc-centered spatial decomposition for all vascular biomarkers:

  RetinalCoordinateSystem  — disc-centred polar coordinate mapping
  VascularZone             — enum (PERIPAPILLARY, ZONE_A, ZONE_B, PERIPHERAL)
  VascularQuadrant         — enum (SUPERIOR, INFERIOR, NASAL, TEMPORAL)
  QuadrantBiomarkers       — per-quadrant biomarker aggregation
  SpatialBiomarkerMap      — 2-D spatial distribution of any scalar biomarker
  SpatialVascularAnalyzer  — orchestrates all spatial analyses

Clinical rationale:
  - Glaucoma primarily affects peripapillary vessels and superior/inferior arcades.
  - DR non-perfusion preferentially affects the temporal periphery.
  - Hypertension affects AV crossing patterns most strongly in the Zone A ring.
  - Macular ischaemia is best characterised within 1 disc-diameter of the fovea.

Coordinate convention:
  - Origin: optic disc centre (row, col) in image pixels.
  - Distance: Euclidean from disc centre, normalised by disc radius.
  - Angle: degrees from horizontal (0° = nasal/right, 90° = superior/up for OD).
  - Quadrant threshold: 45° per quadrant around each cardinal axis.

Design contracts:
  - All methods accept VascularGraph; never modify inputs.
  - Returns explicit nan for zones with no vessel data.
  - Spatial maps are always float32 with shape (n_rings, n_sectors).
  - Works without AV labels (degrades gracefully).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Spatial classification enums ────────────────────────────────────────────

class VascularZone(Enum):
    """Concentric disc-centred ring zones."""
    PERIPAPILLARY = "peripapillary"   # 0–1.0 disc radii (on the disc margin)
    ZONE_A        = "zone_a"          # 1.0–2.0 disc radii (NVZ / Knudtson zone)
    ZONE_B        = "zone_b"          # 2.0–3.5 disc radii (mid-periphery)
    PERIPHERAL    = "peripheral"      # > 3.5 disc radii (far periphery)

    @classmethod
    def from_distance(cls, normalised_dist: float) -> "VascularZone":
        """Classify a normalised distance (d / disc_radius) into a zone."""
        if normalised_dist <= 1.0:
            return cls.PERIPAPILLARY
        if normalised_dist <= 2.0:
            return cls.ZONE_A
        if normalised_dist <= 3.5:
            return cls.ZONE_B
        return cls.PERIPHERAL


class VascularQuadrant(Enum):
    """Four canonical retinal quadrants."""
    SUPERIOR  = "superior"
    INFERIOR  = "inferior"
    NASAL     = "nasal"
    TEMPORAL  = "temporal"

    @classmethod
    def from_angle(cls, angle_deg: float, is_left_eye: bool = False) -> "VascularQuadrant":
        """
        Map an angle (0° = right / nasal side for right eye) to a quadrant.

        For left eye, nasal and temporal are reversed.
        """
        a = angle_deg % 360
        # canonical quadrant assignment (right eye)
        if 45 <= a < 135:
            q = cls.SUPERIOR
        elif 135 <= a < 225:
            q = cls.NASAL if not is_left_eye else cls.TEMPORAL
        elif 225 <= a < 315:
            q = cls.INFERIOR
        else:
            q = cls.TEMPORAL if not is_left_eye else cls.NASAL
        return q


# ─── Coordinate system ────────────────────────────────────────────────────────

@dataclass
class RetinalCoordinateSystem:
    """
    Disc-centred polar coordinate system.

    Attributes:
        disc_centre:   (row, col) of the optic disc centre in image pixels.
        disc_radius:   optic disc radius in image pixels.
        is_left_eye:   True if this is a left-eye fundus image (mirrors nasal/temporal).
        fovea_offset:  approximate (Δrow, Δcol) from disc to fovea.
                       Default: fovea is ~2.5 disc-diameters temporal to the disc.
    """
    disc_centre:  Tuple[float, float]
    disc_radius:  float
    is_left_eye:  bool = False
    fovea_offset: Tuple[float, float] = None   # (Δrow, Δcol)

    def __post_init__(self):
        if self.disc_radius <= 0:
            raise ValueError(f"disc_radius must be > 0, got {self.disc_radius}")
        if self.fovea_offset is None:
            # Right eye: fovea is ~2.5 DD temporal (left) and slightly inferior
            temporal_dir = -1.0 if not self.is_left_eye else 1.0
            self.fovea_offset = (0.5 * self.disc_radius,
                                 temporal_dir * 2.5 * self.disc_radius)

    def to_polar(
        self, row: float, col: float
    ) -> Tuple[float, float, VascularZone, VascularQuadrant]:
        """
        Convert image (row, col) to polar coordinates + zone + quadrant.

        Returns:
            (distance_px, angle_deg, zone, quadrant)
            distance_px: Euclidean distance from disc centre in pixels.
            angle_deg:   angle in degrees (0° = right/nasal, CCW positive).
            zone:        VascularZone.
            quadrant:    VascularQuadrant.
        """
        dr = row - self.disc_centre[0]
        dc = col - self.disc_centre[1]
        dist = float(np.sqrt(dr ** 2 + dc ** 2))
        angle = float(np.degrees(np.arctan2(-dr, dc))) % 360   # image y axis inverted

        norm_dist = dist / max(self.disc_radius, 1)
        zone      = VascularZone.from_distance(norm_dist)
        quadrant  = VascularQuadrant.from_angle(angle, self.is_left_eye)

        return dist, angle, zone, quadrant

    @property
    def fovea_centre(self) -> Tuple[float, float]:
        """Approximate fovea centre in image pixel coordinates."""
        return (
            self.disc_centre[0] + self.fovea_offset[0],
            self.disc_centre[1] + self.fovea_offset[1],
        )


# ─── Biomarker containers ─────────────────────────────────────────────────────

@dataclass
class QuadrantBiomarkers:
    """Per-quadrant vascular biomarkers."""
    quadrant:         VascularQuadrant
    node_count:       int
    edge_count:       int
    mean_tortuosity:  float
    mean_diameter:    float
    vessel_density:   float     # nodes per unit area proxy
    av_ratio:         float     # mean artery / vein ratio (nan if no AV labels)
    tortuosity_std:   float


@dataclass
class ZoneBiomarkers:
    """Per-zone vascular biomarkers."""
    zone:             VascularZone
    node_count:       int
    edge_count:       int
    mean_tortuosity:  float
    mean_diameter:    float
    mean_length:      float
    vessel_density:   float
    quadrant_data:    Dict[VascularQuadrant, QuadrantBiomarkers] = field(default_factory=dict)


@dataclass
class SpatialBiomarkerMap:
    """
    2-D spatial map of a single scalar biomarker.

    Shape: (n_zones, n_quadrants) — typically (4, 4).
    NaN where insufficient data.
    """
    biomarker_name: str
    values:         np.ndarray    # (n_zones, n_quadrants) float32
    zone_labels:    List[str]
    quadrant_labels: List[str]
    stats:          Dict[str, float] = field(default_factory=dict)  # mean, std, min, max

    def to_dict(self) -> Dict[str, float]:
        out = {}
        for zi, z in enumerate(self.zone_labels):
            for qi, q in enumerate(self.quadrant_labels):
                k = f"{self.biomarker_name}_{z}_{q}"
                v = float(self.values[zi, qi])
                out[k] = v
        out.update({f"{self.biomarker_name}_{k}": v for k, v in self.stats.items()})
        return out


@dataclass
class SpatialAnalysisReport:
    """
    Full spatial analysis output for one image.

    Attributes:
        coord_system:    RetinalCoordinateSystem used.
        zone_biomarkers: per-zone statistics.
        quadrant_maps:   per-biomarker SpatialBiomarkerMap.
        peripapillary_density: vessel density in peripapillary zone.
        macular_density:       vessel density in macular zone proxy.
        superior_inferior_ratio: SIR — asymmetry between S and I quadrants.
        nasal_temporal_ratio:    NTR — asymmetry between N and T quadrants.
        spatial_embedding:       (D,) float32 vector for ML consumption.
    """
    coord_system:           RetinalCoordinateSystem
    zone_biomarkers:        Dict[VascularZone, ZoneBiomarkers]
    quadrant_maps:          Dict[str, SpatialBiomarkerMap]
    peripapillary_density:  float
    macular_density:        float
    superior_inferior_ratio: float
    nasal_temporal_ratio:    float
    spatial_embedding:      np.ndarray

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "peripapillary_density":  self.peripapillary_density,
            "macular_density":        self.macular_density,
            "superior_inferior_ratio": self.superior_inferior_ratio,
            "nasal_temporal_ratio":    self.nasal_temporal_ratio,
        }
        for zone, zb in self.zone_biomarkers.items():
            z = zone.value
            out[f"{z}_node_count"]     = float(zb.node_count)
            out[f"{z}_mean_tortuosity"] = float(zb.mean_tortuosity)
            out[f"{z}_mean_diameter"]   = float(zb.mean_diameter)
            out[f"{z}_vessel_density"]  = float(zb.vessel_density)
        for name, smap in self.quadrant_maps.items():
            out.update(smap.to_dict())
        return out


# ─── Spatial Analyzer ─────────────────────────────────────────────────────────

class SpatialVascularAnalyzer:
    """
    Performs spatial retinal vascular analysis relative to the optic disc.

    Usage:
        coord  = RetinalCoordinateSystem(disc_centre=(256, 256), disc_radius=40)
        sa     = SpatialVascularAnalyzer(coord)
        report = sa.analyze(graph, mask)

        print(report.peripapillary_density)
        print(report.superior_inferior_ratio)
        embedding = report.spatial_embedding   # (30,) feature vector
    """

    def __init__(self, coord_system: RetinalCoordinateSystem):
        self.coord = coord_system

    def _assign_nodes(self, graph) -> Dict[int, Tuple[VascularZone, VascularQuadrant]]:
        """Assign zone + quadrant to every node."""
        assignments: Dict[int, Tuple[VascularZone, VascularQuadrant]] = {}
        for node in graph.nodes:
            r, c    = node.position[0], node.position[1]
            _, _, z, q = self.coord.to_polar(r, c)
            assignments[node.node_id] = (z, q)
        return assignments

    def _zone_stats(
        self,
        graph,
        assignments: Dict[int, Tuple[VascularZone, VascularQuadrant]],
        mask:        Optional[np.ndarray] = None,
    ) -> Dict[VascularZone, ZoneBiomarkers]:
        """Compute per-zone biomarkers."""
        node_map = {n.node_id: n for n in graph.nodes}
        edge_map = {(e.src, e.dst): e for e in graph.edges}
        edge_map.update({(e.dst, e.src): e for e in graph.edges})

        zone_nodes: Dict[VascularZone, List] = {z: [] for z in VascularZone}
        zone_edges: Dict[VascularZone, List] = {z: [] for z in VascularZone}

        for nid, (z, q) in assignments.items():
            zone_nodes[z].append(nid)

        # Assign each edge to the zone of its geometric midpoint.
        # Using only the source node creates systematic bias for edges that
        # cross zone boundaries; the midpoint gives an unbiased attribution.
        node_positions = {n.node_id: n.position for n in graph.nodes}
        for edge in graph.edges:
            src_pos = node_positions.get(edge.src)
            dst_pos = node_positions.get(edge.dst)
            if src_pos is not None and dst_pos is not None:
                mid_r = (src_pos[0] + dst_pos[0]) / 2.0
                mid_c = (src_pos[1] + dst_pos[1]) / 2.0
                _, _, z_mid, _ = self.coord.to_polar(mid_r, mid_c)
                zone_edges[z_mid].append(edge)
            else:
                # Fallback: attribute to source zone if position unavailable
                z_src = assignments.get(edge.src, (VascularZone.PERIPHERAL, None))[0]
                zone_edges[z_src].append(edge)

        # Approximate image area per zone ring
        H = mask.shape[0] if mask is not None else 512
        W = mask.shape[1] if mask is not None else 512
        dr = self.coord.disc_radius

        def zone_area(z: VascularZone) -> float:
            limits = {
                VascularZone.PERIPAPILLARY: (0.0, 1.0 * dr),
                VascularZone.ZONE_A:        (1.0 * dr, 2.0 * dr),
                VascularZone.ZONE_B:        (2.0 * dr, 3.5 * dr),
                VascularZone.PERIPHERAL:    (3.5 * dr, max(H, W) / 2),
            }
            r_lo, r_hi = limits[z]
            return max(np.pi * (r_hi ** 2 - r_lo ** 2), 1.0)

        results: Dict[VascularZone, ZoneBiomarkers] = {}

        for zone in VascularZone:
            nids  = zone_nodes[zone]
            edges = zone_edges[zone]
            n_n   = len(nids)
            n_e   = len(edges)

            tortuosities = [e.tortuosity   for e in edges if np.isfinite(e.tortuosity)]
            diameters    = [e.mean_diameter for e in edges if np.isfinite(e.mean_diameter)]
            lengths      = [e.length        for e in edges if np.isfinite(e.length)]

            results[zone] = ZoneBiomarkers(
                zone=zone,
                node_count=n_n,
                edge_count=n_e,
                mean_tortuosity=float(np.mean(tortuosities)) if tortuosities else float("nan"),
                mean_diameter=float(np.mean(diameters))      if diameters    else float("nan"),
                mean_length=float(np.mean(lengths))          if lengths      else float("nan"),
                vessel_density=float(n_n / zone_area(zone)),
                quadrant_data={},
            )

        return results

    def _quadrant_stats(
        self,
        graph,
        assignments: Dict[int, Tuple[VascularZone, VascularQuadrant]],
    ) -> Dict[VascularQuadrant, QuadrantBiomarkers]:
        """Compute per-quadrant biomarkers (across all zones combined)."""
        edge_map = {(e.src, e.dst): e for e in graph.edges}
        edge_map.update({(e.dst, e.src): e for e in graph.edges})

        q_nodes: Dict[VascularQuadrant, List[int]] = {q: [] for q in VascularQuadrant}
        q_edges: Dict[VascularQuadrant, List] = {q: [] for q in VascularQuadrant}

        for nid, (_, q) in assignments.items():
            q_nodes[q].append(nid)

        # Assign each edge to the quadrant of its geometric midpoint (unbiased).
        node_positions_q = {n.node_id: n.position for n in graph.nodes}
        for edge in graph.edges:
            src_pos = node_positions_q.get(edge.src)
            dst_pos = node_positions_q.get(edge.dst)
            if src_pos is not None and dst_pos is not None:
                mid_r = (src_pos[0] + dst_pos[0]) / 2.0
                mid_c = (src_pos[1] + dst_pos[1]) / 2.0
                _, _, _, q_mid = self.coord.to_polar(mid_r, mid_c)
                q_edges[q_mid].append(edge)
            else:
                q_src = assignments.get(edge.src, (None, VascularQuadrant.TEMPORAL))[1]
                q_edges[q_src].append(edge)

        results: Dict[VascularQuadrant, QuadrantBiomarkers] = {}
        total_nodes = max(len(graph.nodes), 1)

        for quad in VascularQuadrant:
            nids  = q_nodes[quad]
            edges = q_edges[quad]

            tortuosities = [e.tortuosity   for e in edges if np.isfinite(e.tortuosity)]
            diameters    = [e.mean_diameter for e in edges if np.isfinite(e.mean_diameter)]

            results[quad] = QuadrantBiomarkers(
                quadrant=quad,
                node_count=len(nids),
                edge_count=len(edges),
                mean_tortuosity=float(np.mean(tortuosities)) if tortuosities else float("nan"),
                mean_diameter=float(np.mean(diameters))      if diameters    else float("nan"),
                vessel_density=float(len(nids) / total_nodes),
                av_ratio=float("nan"),   # requires AV labels
                tortuosity_std=float(np.std(tortuosities)) if tortuosities else float("nan"),
            )

        return results

    def _spatial_map(
        self,
        biomarker_name: str,
        values_by_zone_quad: Dict[Tuple[VascularZone, VascularQuadrant], float],
    ) -> SpatialBiomarkerMap:
        zones  = list(VascularZone)
        quads  = list(VascularQuadrant)
        grid   = np.full((len(zones), len(quads)), float("nan"), dtype=np.float32)

        for zi, z in enumerate(zones):
            for qi, q in enumerate(quads):
                v = values_by_zone_quad.get((z, q), float("nan"))
                grid[zi, qi] = v

        finite = grid[np.isfinite(grid)]
        stats  = {
            "mean": float(np.mean(finite)) if len(finite) > 0 else float("nan"),
            "std":  float(np.std(finite))  if len(finite) > 0 else float("nan"),
            "min":  float(np.min(finite))  if len(finite) > 0 else float("nan"),
            "max":  float(np.max(finite))  if len(finite) > 0 else float("nan"),
        }

        return SpatialBiomarkerMap(
            biomarker_name=biomarker_name,
            values=grid,
            zone_labels=[z.value for z in zones],
            quadrant_labels=[q.value for q in quads],
            stats=stats,
        )

    def analyze(
        self,
        graph,
        mask: Optional[np.ndarray] = None,
    ) -> SpatialAnalysisReport:
        """
        Full spatial analysis of a VascularGraph.

        Args:
            graph: VascularGraph from graph_builder.
            mask:  optional (H, W) binary float32 mask (improves density estimates).

        Returns:
            SpatialAnalysisReport.
        """
        assignments  = self._assign_nodes(graph)
        zone_data    = self._zone_stats(graph, assignments, mask)
        quad_data    = self._quadrant_stats(graph, assignments)

        # Peripapillary and macular density
        peri_density = zone_data[VascularZone.PERIPAPILLARY].vessel_density
        # Macular proxy: zone_a in temporal quadrant
        temporal_za = self._zone_quad_stats(graph, assignments, VascularZone.ZONE_A,
                                             VascularQuadrant.TEMPORAL)
        macula_density = temporal_za.get("vessel_density", float("nan"))

        # SIR and NTR from quadrant densities
        sup = quad_data[VascularQuadrant.SUPERIOR].vessel_density
        inf = quad_data[VascularQuadrant.INFERIOR].vessel_density
        nas = quad_data[VascularQuadrant.NASAL].vessel_density
        tem = quad_data[VascularQuadrant.TEMPORAL].vessel_density

        sir = float(sup / max(inf, 1e-8))
        ntr = float(nas / max(tem, 1e-8))

        # Build tortuosity spatial map
        tort_map_data: Dict[Tuple[VascularZone, VascularQuadrant], float] = {}
        edge_by_zone_quad: Dict[Tuple[VascularZone, VascularQuadrant], List[float]] = {}
        for edge in graph.edges:
            zq_src = assignments.get(edge.src)
            if zq_src is None:
                continue
            z, q = zq_src
            key  = (z, q)
            edge_by_zone_quad.setdefault(key, [])
            if np.isfinite(edge.tortuosity):
                edge_by_zone_quad[key].append(edge.tortuosity)

        for key, vals in edge_by_zone_quad.items():
            tort_map_data[key] = float(np.mean(vals)) if vals else float("nan")

        tort_map = self._spatial_map("tortuosity", tort_map_data)

        # Spatial embedding (12 zone×quadrant means + 4 zone aggregates + 4 quadrant aggregates)
        embedding_parts = []
        for zone in VascularZone:
            zb = zone_data[zone]
            embedding_parts.extend([
                _safe(zb.vessel_density * 1000),   # scale for embedding
                _safe(zb.mean_tortuosity - 1.0),   # tortuosity excess
                _safe(zb.mean_diameter / 10.0),
                float(zb.node_count) / max(len(graph.nodes), 1),
            ])
        for quad in VascularQuadrant:
            qb = quad_data[quad]
            embedding_parts.extend([
                qb.vessel_density,
                _safe(qb.mean_tortuosity - 1.0),
                _safe(qb.tortuosity_std),
            ])
        # Asymmetry features
        embedding_parts.extend([
            _safe(sir - 1.0),
            _safe(ntr - 1.0),
        ])

        embedding = np.array(embedding_parts, dtype=np.float32)
        embedding = np.where(np.isfinite(embedding), embedding, 0.0).astype(np.float32)

        return SpatialAnalysisReport(
            coord_system=self.coord,
            zone_biomarkers=zone_data,
            quadrant_maps={"tortuosity": tort_map},
            peripapillary_density=float(peri_density),
            macular_density=float(macula_density),
            superior_inferior_ratio=sir,
            nasal_temporal_ratio=ntr,
            spatial_embedding=embedding,
        )

    def _zone_quad_stats(
        self,
        graph,
        assignments,
        zone: VascularZone,
        quad: VascularQuadrant,
    ) -> Dict[str, float]:
        """Compute stats for a single (zone, quadrant) cell."""
        nids = [
            nid for nid, (z, q) in assignments.items()
            if z == zone and q == quad
        ]
        edges = [
            e for e in graph.edges
            if assignments.get(e.src, (None, None))[0] == zone
            and assignments.get(e.src, (None, None))[1] == quad
        ]
        return {
            "node_count":    float(len(nids)),
            "vessel_density": float(len(nids) / max(len(graph.nodes), 1)),
            "mean_tortuosity": float(
                np.mean([e.tortuosity for e in edges if np.isfinite(e.tortuosity)])
            ) if edges else float("nan"),
        }


def _safe(v: float, default: float = 0.0) -> float:
    return float(v) if np.isfinite(v) else default
