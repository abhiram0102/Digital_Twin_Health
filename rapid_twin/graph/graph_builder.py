"""
Retinal Vascular Graph Construction.
Converts binary vessel segmentation mask into a structured graph:
  Nodes = vessel bifurcations / intersections
  Edges = vessel segments between nodes
  Node features: branching angle, vessel density, tortuosity, position
  Edge features: diameter, curvature, length, intensity profile
"""

import cv2
import numpy as np
import networkx as nx
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

try:
    from skimage.morphology import skeletonize, thin
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available; falling back to OpenCV skeletonization")


@dataclass
class VascularNode:
    node_id: int
    position: Tuple[int, int]       # (row, col) pixel coords
    degree: int                      # number of edges
    branching_angle: float           # mean angle between vessel branches (degrees)
    local_vessel_density: float      # fraction of vessel pixels in local neighborhood
    tortuosity: float                # local curvature measure
    is_bifurcation: bool
    is_endpoint: bool


@dataclass
class VascularEdge:
    src: int
    dst: int
    length: float                    # pixels
    mean_diameter: float             # pixels
    curvature: float                 # integral curvature / length
    mean_intensity: float            # mean pixel value along vessel
    tortuosity: float                # actual path length / straight distance


@dataclass
class VascularGraph:
    nodes: List[VascularNode]
    edges: List[VascularEdge]
    nx_graph: nx.Graph
    skeleton: np.ndarray
    node_features: np.ndarray       # (N, node_feature_dim)
    edge_features: np.ndarray       # (E, edge_feature_dim)
    edge_index: np.ndarray          # (2, E) — COO format


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin binary vessel mask to 1-pixel skeleton."""
    binary = (mask > 0.5).astype(np.uint8)
    if SKIMAGE_AVAILABLE:
        skel = skeletonize(binary.astype(bool)).astype(np.uint8)
    else:
        # OpenCV Zhang-Suen thinning approximation
        skel = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opening = cv2.dilate(eroded, element)
            subset = temp - opening
            skel = cv2.bitwise_or(skel, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
    return skel


def _find_junction_points(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find bifurcation points (degree >= 3) and endpoints (degree == 1)
    in skeleton using 3x3 neighborhood connectivity.
    """
    skel = (skeleton > 0).astype(np.uint8)
    # Neighbor count via convolution
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel) * skel

    junctions = np.argwhere(neighbor_count >= 3)   # bifurcations / crossings
    endpoints = np.argwhere(neighbor_count == 1)    # vessel endpoints

    return junctions, endpoints


def _estimate_vessel_diameter(vessel_mask: np.ndarray, point: Tuple[int, int],
                               radius: int = 10) -> float:
    """Local vessel diameter estimate via distance transform."""
    dist = cv2.distanceTransform(vessel_mask.astype(np.uint8), cv2.DIST_L2, 5)
    r, c = point
    patch = dist[max(0, r-radius):r+radius, max(0, c-radius):c+radius]
    if patch.size == 0:
        return 1.0
    return float(patch.max() * 2)  # radius → diameter


def _compute_tortuosity(path_coords: np.ndarray) -> float:
    """
    Vessel tortuosity = actual path length / straight-line distance.
    Values > 1; closer to 1 = straighter vessel.
    """
    if len(path_coords) < 2:
        return 1.0
    path_len = sum(
        np.linalg.norm(path_coords[i+1] - path_coords[i])
        for i in range(len(path_coords) - 1)
    )
    straight = np.linalg.norm(path_coords[-1] - path_coords[0])
    return float(path_len / (straight + 1e-6))


def _compute_branching_angle(node_pos: np.ndarray, neighbor_positions: List[np.ndarray]) -> float:
    """Mean angle between vessel branches at a bifurcation node."""
    if len(neighbor_positions) < 2:
        return 0.0
    vectors = [n - node_pos for n in neighbor_positions]
    angles = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos_a = np.dot(vectors[i], vectors[j]) / (
                np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return float(np.mean(angles)) if angles else 0.0


def _local_vessel_density(vessel_mask: np.ndarray, pos: Tuple[int, int], radius: int = 20) -> float:
    r, c = pos
    patch = vessel_mask[max(0, r-radius):r+radius, max(0, c-radius):c+radius]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def _trace_path_bfs(skeleton: np.ndarray, start: Tuple[int, int], end: Tuple[int, int],
                    node_positions: set, max_steps: int = 1000) -> Optional[np.ndarray]:
    """
    BFS path tracing along skeleton between two junction/endpoint nodes.

    Uses a parent-pointer dict instead of carrying the full path list on each
    queue entry.  The previous implementation stored path + [(nr, nc)] on
    every expansion which is O(L²) in memory and time for path length L.
    Parent-pointer reconstruction is O(L) time and O(V) memory.
    """
    from collections import deque
    h, w = skeleton.shape
    # parent[pos] = previous pos on the BFS tree (None for start)
    parent: dict = {start: None}
    queue   = deque([(start, 0)])   # (pos, steps)
    dirs    = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while queue:
        (r, c), steps = queue.popleft()

        if (r, c) == end:
            # Reconstruct path by backtracking through parent pointers
            path = []
            pos  = end
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            path.reverse()
            return np.array(path)

        if steps >= max_steps:
            return None

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            np_ = (nr, nc)
            if (0 <= nr < h and 0 <= nc < w
                    and skeleton[nr, nc]
                    and np_ not in parent):
                parent[np_] = (r, c)
                if np_ == end or np_ not in node_positions:
                    queue.append((np_, steps + 1))

    return None


class RetinalGraphBuilder:
    """
    Converts segmented vessel mask to a VascularGraph.
    """

    def __init__(self, min_branch_length: int = 10, max_nodes: int = 512):
        self.min_branch_length = min_branch_length
        self.max_nodes = max_nodes

    def build(self, vessel_mask: np.ndarray,
              intensity_image: Optional[np.ndarray] = None) -> VascularGraph:
        """
        Build vascular graph from binary vessel mask.

        Args:
            vessel_mask: (H, W) float in [0,1] or binary
            intensity_image: (H, W) float, original image for intensity features

        Returns:
            VascularGraph
        """
        mask = (vessel_mask > 0.5).astype(np.uint8)
        if intensity_image is None:
            intensity_image = mask.astype(np.float32)

        # 1. Skeletonize
        skeleton = _skeletonize_mask(mask)

        # 2. Find junction and endpoint pixels
        junctions, endpoints = _find_junction_points(skeleton)

        # 3. Merge nearby nodes (NMS-style)
        all_pts = np.vstack([junctions, endpoints]) if (len(junctions) and len(endpoints)) \
                  else (junctions if len(junctions) else endpoints)

        if len(all_pts) == 0:
            return self._empty_graph(skeleton)

        all_pts = self._nms_nodes(all_pts, radius=5)
        if len(all_pts) > self.max_nodes:
            all_pts = all_pts[:self.max_nodes]

        # 4. Build NetworkX graph
        G = nx.Graph()
        node_positions = {tuple(p): i for i, p in enumerate(all_pts)}
        node_set = set(node_positions.keys())

        # 4a. Add nodes
        vascular_nodes = []
        for i, pos in enumerate(all_pts):
            pos_t = tuple(pos)
            # Degree = neighbor count in skeleton
            r, c = pos
            neighbors_in_skel = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r+dr, c+dc
                    if (dr, dc) != (0, 0) and 0 <= nr < skeleton.shape[0] and \
                       0 <= nc < skeleton.shape[1] and skeleton[nr, nc]:
                        neighbors_in_skel.append(np.array([nr, nc]))

            degree = len(neighbors_in_skel)
            b_angle = _compute_branching_angle(np.array(pos), neighbors_in_skel)
            density = _local_vessel_density(mask, pos_t)
            is_bif = degree >= 3
            is_end = degree == 1

            vn = VascularNode(
                node_id=i, position=pos_t, degree=degree,
                branching_angle=b_angle, local_vessel_density=density,
                tortuosity=0.0,  # filled after edge tracing
                is_bifurcation=is_bif, is_endpoint=is_end
            )
            vascular_nodes.append(vn)
            G.add_node(i, pos=pos_t, degree=degree, angle=b_angle, density=density)

        # 4b. Add edges: trace skeleton paths between adjacent nodes
        vascular_edges = []
        visited_pairs = set()

        for i, pos in enumerate(all_pts):
            r, c = tuple(pos)
            # Search outward for connected nodes
            neighbors = self._find_connected_nodes(skeleton, (r, c), node_set, node_positions,
                                                   max_dist=200)
            for j_pos in neighbors:
                j = node_positions[j_pos]
                pair = tuple(sorted([i, j]))
                if pair in visited_pairs or i == j:
                    continue
                visited_pairs.add(pair)

                path = _trace_path_bfs(skeleton, (r, c), j_pos, node_set)
                if path is None or len(path) < self.min_branch_length:
                    continue

                path_arr = path.astype(np.float32)
                length = sum(np.linalg.norm(path_arr[k+1] - path_arr[k])
                             for k in range(len(path_arr) - 1))
                tort = _compute_tortuosity(path_arr)
                diam = np.mean([_estimate_vessel_diameter(mask, tuple(pt.astype(int)), radius=8)
                                for pt in path_arr[::max(1, len(path_arr)//5)]])

                # Intensity along path
                int_vals = [intensity_image[int(pt[0]), int(pt[1])]
                            for pt in path_arr if 0 <= int(pt[0]) < intensity_image.shape[0]
                            and 0 <= int(pt[1]) < intensity_image.shape[1]]
                mean_int = float(np.mean(int_vals)) if int_vals else 0.0

                # Curvature: mean abs angle change between consecutive segments
                curvature = self._path_curvature(path_arr)

                ve = VascularEdge(src=i, dst=j, length=float(length),
                                  mean_diameter=float(diam), curvature=float(curvature),
                                  mean_intensity=mean_int, tortuosity=float(tort))
                vascular_edges.append(ve)
                G.add_edge(i, j, length=length, diameter=diam, curvature=curvature,
                           intensity=mean_int, tortuosity=tort)

        # 5. Update node tortuosity as mean of incident edge tortuosities
        for vn in vascular_nodes:
            inc_torts = [e.tortuosity for e in vascular_edges if e.src == vn.node_id or e.dst == vn.node_id]
            vn.tortuosity = float(np.mean(inc_torts)) if inc_torts else 1.0

        # 6. Build feature matrices
        node_feats = self._build_node_features(vascular_nodes)
        edge_feats, edge_index = self._build_edge_features(vascular_edges)

        return VascularGraph(
            nodes=vascular_nodes, edges=vascular_edges, nx_graph=G,
            skeleton=skeleton, node_features=node_feats,
            edge_features=edge_feats, edge_index=edge_index
        )

    def _nms_nodes(self, pts: np.ndarray, radius: int = 5) -> np.ndarray:
        """Non-max suppression: keep one point per radius neighborhood."""
        if len(pts) == 0:
            return pts
        kept = []
        pts_list = pts.tolist()
        used = [False] * len(pts_list)
        for i, p in enumerate(pts_list):
            if used[i]:
                continue
            kept.append(p)
            for j, q in enumerate(pts_list):
                if not used[j] and np.linalg.norm(np.array(p) - np.array(q)) < radius:
                    used[j] = True
        return np.array(kept)

    def _find_connected_nodes(self, skeleton, start, node_set, node_positions, max_dist=200):
        """BFS to find nearby node positions connected via skeleton."""
        from collections import deque
        h, w = skeleton.shape
        queue = deque([start])
        visited = {start}
        found = []
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        dist = {start: 0}

        while queue:
            pos = queue.popleft()
            d = dist[pos]
            if d > max_dist:
                continue
            r, c = pos
            for dr, dc in dirs:
                np_ = (r+dr, c+dc)
                if 0 <= np_[0] < h and 0 <= np_[1] < w and skeleton[np_] and np_ not in visited:
                    visited.add(np_)
                    dist[np_] = d + 1
                    if np_ in node_set and np_ != start:
                        found.append(np_)
                    elif np_ not in node_set:
                        queue.append(np_)
        return found

    def _path_curvature(self, path: np.ndarray) -> float:
        """Mean absolute angle change along path (radians)."""
        if len(path) < 3:
            return 0.0
        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angles.append(abs(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return float(np.mean(angles)) if angles else 0.0

    def _build_node_features(self, nodes: List[VascularNode]) -> np.ndarray:
        """
        Node feature vector (dim=16):
          0  row_norm              row / max_pos  (spatial)
          1  col_norm              col / max_pos  (spatial)
          2  degree_norm           degree / 6     (topology)
          3  is_bifurcation        bool           (topology)
          4  is_endpoint           bool           (topology)
          5  branching_angle_norm  angle / 180    (geometry)
          6  local_vessel_density  [0,1]          (texture)
          7  tortuosity_excess     tortuosity-1   (shape)
          8  deg_sin               sin(angle)     (cyclic angle encoding)
          9  deg_cos               cos(angle)     (cyclic angle encoding)
         10  is_crossing           degree >= 4    (AV crossing proxy)
         11  degree_sq_norm        (degree/6)²    (non-linear degree)
         12  log1p_density         log1p(density) (log-scale density)
         13  tortuosity_norm       clip(t,1,3)/3  (bounded tortuosity)
         14  row_centered          (row-0.5)/0.5  (signed spatial)
         15  col_centered          (col-0.5)/0.5  (signed spatial)

        Features 10–15 replace the 6 zero-padding slots that were present in
        the original implementation and conveyed no information to downstream
        GNN models.
        """
        if not nodes:
            return np.zeros((0, 16), dtype=np.float32)
        feats = []
        max_pos = max((max(n.position) for n in nodes), default=1.0)
        for n in nodes:
            r_n = n.position[0] / (max_pos + 1e-6)
            c_n = n.position[1] / (max_pos + 1e-6)
            tort_norm = float(np.clip(n.tortuosity, 1.0, 3.0) / 3.0)
            row = [
                r_n,
                c_n,
                n.degree / 6.0,
                float(n.is_bifurcation),
                float(n.is_endpoint),
                n.branching_angle / 180.0,
                n.local_vessel_density,
                max(0.0, n.tortuosity - 1.0),
                float(np.sin(np.radians(n.branching_angle))),
                float(np.cos(np.radians(n.branching_angle))),
                # Previously zero-padded — now meaningful features:
                float(n.degree >= 4),                        # is_crossing
                float((n.degree / 6.0) ** 2),               # degree_sq_norm
                float(np.log1p(n.local_vessel_density)),     # log1p_density
                tort_norm,                                   # tortuosity_norm
                float((r_n - 0.5) / 0.5),                   # row_centered
                float((c_n - 0.5) / 0.5),                   # col_centered
            ]
            feats.append(row)
        return np.array(feats, dtype=np.float32)

    def _build_edge_features(self, edges: List[VascularEdge]):
        """
        Edge feature vector (dim=8):
        [length_norm, diameter_norm, curvature, mean_intensity, tortuosity-1,
         diam/length, 0, 0]
        """
        if not edges:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((2, 0), dtype=np.int64)
        max_len = max(e.length for e in edges) + 1e-6
        max_diam = max(e.mean_diameter for e in edges) + 1e-6
        feats = []
        src_idx, dst_idx = [], []
        for e in edges:
            row = [e.length / max_len, e.mean_diameter / max_diam,
                   np.clip(e.curvature / (np.pi + 1e-6), 0, 1),
                   np.clip(e.mean_intensity, 0, 1),
                   np.clip(e.tortuosity - 1.0, 0, 5) / 5.0,
                   np.clip(e.mean_diameter / (e.length + 1e-6), 0, 1),
                   0.0, 0.0]
            feats.append(row)
            src_idx.append(e.src); dst_idx.append(e.dst)
            # Undirected: add reverse
            feats.append(row)
            src_idx.append(e.dst); dst_idx.append(e.src)

        edge_index = np.array([src_idx, dst_idx], dtype=np.int64)
        return np.array(feats, dtype=np.float32), edge_index

    def _empty_graph(self, skeleton: np.ndarray) -> VascularGraph:
        G = nx.Graph()
        return VascularGraph([], [], G, skeleton,
                             np.zeros((0, 16), np.float32),
                             np.zeros((0, 8), np.float32),
                             np.zeros((2, 0), np.int64))
