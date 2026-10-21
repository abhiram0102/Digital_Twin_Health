"""
Retinal Vascular Branch Hierarchy Modeling.

Builds a tree representation of the retinal vascular network, rooted at the
most central / highest-degree node (a proxy for the optic disc region when
the disc is not separately segmented).

Provides:
  - BranchNode / BranchTree dataclasses
  - build_branch_hierarchy()  — BFS tree from chosen root
  - annotate_branch_levels()  — depth/level labels on all nodes
  - compute_hierarchy_stats() — branching ratio, depth distribution, asymmetry

IMPORTANT:
  - Input validation with explicit AssertionError / ValueError messages.
  - No silent NaN returns — every failure path is documented.
  - All loops are bounded to avoid infinite recursion.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    warnings.warn("networkx not available — branch hierarchy will be limited.")


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BranchNode:
    """A single node in the branch hierarchy tree."""
    node_id:    int
    parent_id:  Optional[int]       # None for root
    depth:      int                  # 0 = root, 1 = first generation, …
    children:   List[int] = field(default_factory=list)
    subtree_size: int = 1            # filled after tree construction
    is_leaf:    bool = False
    position:   Tuple[int, int] = (0, 0)
    degree:     int = 0


@dataclass
class BranchTree:
    """
    Hierarchical tree representation of the retinal vascular network.

    Attributes:
        root_id:       node_id of the tree root (proxy for optic disc).
        nodes:         dict mapping node_id → BranchNode.
        max_depth:     deepest level in the tree.
        n_leaves:      number of leaf nodes (endpoints / dead-ends).
        branching_ratio: mean number of children per non-leaf node.
        depth_counts:  list of node counts per depth level.
        asymmetry_index: tree imbalance in [0, 1].  0 = perfectly symmetric.
        is_valid:      False if the input graph was empty or degenerate.
    """
    root_id:        int
    nodes:          Dict[int, BranchNode]
    max_depth:      int
    n_leaves:       int
    branching_ratio: float
    depth_counts:   List[int]
    asymmetry_index: float
    is_valid:       bool = True


# ─── Root selection ───────────────────────────────────────────────────────────

def _select_root(vascular_graph) -> int:
    """
    Heuristic root selection for the vessel hierarchy tree.

    Priority (descending):
      1. Node with highest degree (most branching → likely near optic disc).
      2. Node closest to image centre (retinal geometry prior).
      3. First node (deterministic fallback).

    Returns node_id (index into graph.nodes list).
    """
    nodes = vascular_graph.nodes
    if not nodes:
        raise ValueError("Cannot select root from empty graph.")

    # Heuristic 1: highest degree
    max_deg  = max(n.degree for n in nodes)
    high_deg = [n for n in nodes if n.degree == max_deg]

    if len(high_deg) == 1:
        return high_deg[0].node_id

    # Heuristic 2: among highest-degree nodes, closest to image centre
    # Estimate centre from bounding box of all node positions
    all_pos = np.array([n.position for n in nodes], dtype=np.float32)
    centre  = all_pos.mean(axis=0)

    best_node = min(
        high_deg,
        key=lambda n: np.linalg.norm(np.array(n.position) - centre)
    )
    return best_node.node_id


# ─── Tree construction ────────────────────────────────────────────────────────

def build_branch_hierarchy(
    vascular_graph,
    root_id: Optional[int] = None,
    max_depth: int = 50,
) -> BranchTree:
    """
    Build a branch hierarchy tree from the vascular graph via BFS.

    The tree is rooted at root_id (or auto-selected) and spans the entire
    connected component containing the root.  Nodes in separate components
    are excluded from the primary tree (handled as disconnected subtrees).

    Args:
        vascular_graph: VascularGraph from RetinalGraphBuilder.build().
        root_id:        Explicit root node_id. Auto-selected if None.
        max_depth:      Hard cap on BFS depth to prevent runaway on cycles.

    Returns:
        BranchTree.  is_valid=False if graph is empty.
    """
    nodes_list = vascular_graph.nodes

    if not nodes_list:
        return BranchTree(
            root_id=-1, nodes={}, max_depth=0, n_leaves=0,
            branching_ratio=0.0, depth_counts=[], asymmetry_index=0.0,
            is_valid=False,
        )

    # Build id → VascularNode lookup
    id_to_vnode = {n.node_id: n for n in nodes_list}

    # Resolve root
    if root_id is None:
        root_id = _select_root(vascular_graph)

    if root_id not in id_to_vnode:
        raise ValueError(
            f"root_id={root_id} not found in graph (valid ids: "
            f"{list(id_to_vnode.keys())[:10]}…)"
        )

    # Build adjacency from edges
    adjacency: Dict[int, List[int]] = {n.node_id: [] for n in nodes_list}
    for edge in vascular_graph.edges:
        adjacency[edge.src].append(edge.dst)
        adjacency[edge.dst].append(edge.src)

    # BFS tree construction
    branch_nodes: Dict[int, BranchNode] = {}
    visited = {root_id}
    queue: deque[Tuple[int, Optional[int], int]] = deque([(root_id, None, 0)])

    while queue:
        node_id, parent_id, depth = queue.popleft()

        if depth > max_depth:
            warnings.warn(
                f"build_branch_hierarchy: max_depth={max_depth} reached at node {node_id}. "
                "Truncating hierarchy. Consider increasing max_depth or checking for cycles."
            )
            continue

        vnode  = id_to_vnode[node_id]
        bnode  = BranchNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            position=vnode.position,
            degree=vnode.degree,
        )
        branch_nodes[node_id] = bnode

        # Register as child of parent
        if parent_id is not None and parent_id in branch_nodes:
            branch_nodes[parent_id].children.append(node_id)

        # Enqueue unvisited neighbours
        for nb_id in adjacency.get(node_id, []):
            if nb_id not in visited:
                visited.add(nb_id)
                queue.append((nb_id, node_id, depth + 1))

    # ── Compute subtree sizes (post-order) ────────────────────────────────────
    # Process in reverse BFS order (deepest nodes first)
    bfs_order = list(branch_nodes.keys())  # insertion order = BFS order
    for nid in reversed(bfs_order):
        bn = branch_nodes[nid]
        bn.subtree_size = 1 + sum(
            branch_nodes[c].subtree_size
            for c in bn.children
            if c in branch_nodes
        )

    # ── Leaf annotation ───────────────────────────────────────────────────────
    n_leaves = 0
    for bn in branch_nodes.values():
        bn.is_leaf = len(bn.children) == 0
        if bn.is_leaf:
            n_leaves += 1

    # ── Tree statistics ───────────────────────────────────────────────────────
    depths = [bn.depth for bn in branch_nodes.values()]
    actual_max_depth = max(depths) if depths else 0

    # Depth histogram
    depth_counts: List[int] = [0] * (actual_max_depth + 1)
    for d in depths:
        depth_counts[d] += 1

    # Branching ratio = mean number of children among non-leaf nodes
    non_leaves = [bn for bn in branch_nodes.values() if not bn.is_leaf]
    branching_ratio = (
        float(np.mean([len(bn.children) for bn in non_leaves]))
        if non_leaves else 0.0
    )

    # Asymmetry index: based on subtree size imbalance at each bifurcation
    # For each internal node with >= 2 children:
    # asym_i = |left_size - right_size| / (left_size + right_size)
    # Aggregate as mean over all bifurcation nodes.
    asym_scores = []
    for bn in branch_nodes.values():
        child_sizes = sorted(
            [branch_nodes[c].subtree_size for c in bn.children if c in branch_nodes],
            reverse=True,
        )
        if len(child_sizes) >= 2:
            s1, s2 = child_sizes[0], child_sizes[1]
            asym_scores.append(abs(s1 - s2) / (s1 + s2 + 1e-8))

    asymmetry_index = float(np.mean(asym_scores)) if asym_scores else 0.0

    return BranchTree(
        root_id=root_id,
        nodes=branch_nodes,
        max_depth=actual_max_depth,
        n_leaves=n_leaves,
        branching_ratio=branching_ratio,
        depth_counts=depth_counts,
        asymmetry_index=asymmetry_index,
        is_valid=True,
    )


# ─── Level annotation ─────────────────────────────────────────────────────────

def annotate_branch_levels(
    vascular_graph,
    tree: BranchTree,
) -> Dict[int, int]:
    """
    Return a mapping {node_id: depth_level} for all nodes in the hierarchy tree.

    Nodes not reachable from the root (disconnected components) receive depth=-1.

    Args:
        vascular_graph: VascularGraph (used only to enumerate all node ids).
        tree:           BranchTree produced by build_branch_hierarchy().

    Returns:
        Dict mapping node_id → depth level (0 = root, -1 = disconnected).
    """
    levels: Dict[int, int] = {}

    # Nodes in tree
    for node_id, bnode in tree.nodes.items():
        levels[node_id] = bnode.depth

    # Nodes NOT in tree (disconnected components)
    for vnode in vascular_graph.nodes:
        if vnode.node_id not in levels:
            levels[vnode.node_id] = -1

    return levels


# ─── Hierarchy statistics ─────────────────────────────────────────────────────

def compute_hierarchy_stats(tree: BranchTree) -> Dict[str, float]:
    """
    Derive quantitative statistics from a BranchTree.

    Args:
        tree: BranchTree from build_branch_hierarchy().

    Returns:
        dict with:
          - max_depth:          maximum tree depth.
          - mean_depth:         mean depth across all tree nodes.
          - std_depth:          standard deviation of node depth.
          - n_leaves:           number of leaf nodes.
          - n_internal:         number of internal (non-leaf) nodes.
          - branching_ratio:    mean children per non-leaf node.
          - asymmetry_index:    tree imbalance in [0, 1].
          - root_subtree_size:  total nodes reachable from root.
          - depth_entropy:      Shannon entropy of the depth distribution.
                                High entropy = evenly spread tree.
          - is_valid:           1.0 if tree is valid, 0.0 otherwise.
    """
    if not tree.is_valid or not tree.nodes:
        return {
            "max_depth": 0.0, "mean_depth": 0.0, "std_depth": 0.0,
            "n_leaves": 0.0, "n_internal": 0.0, "branching_ratio": 0.0,
            "asymmetry_index": 0.0, "root_subtree_size": 0.0,
            "depth_entropy": 0.0, "is_valid": 0.0,
        }

    depths = [bn.depth for bn in tree.nodes.values()]
    n_internal = len([bn for bn in tree.nodes.values() if not bn.is_leaf])

    # Depth distribution entropy
    counts = np.array(tree.depth_counts, dtype=np.float32)
    total  = counts.sum()
    if total > 0 and len(counts) > 1:
        probs   = counts / total
        probs   = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    root_subtree = tree.nodes[tree.root_id].subtree_size if tree.root_id in tree.nodes else 0

    return {
        "max_depth":         float(tree.max_depth),
        "mean_depth":        float(np.mean(depths)),
        "std_depth":         float(np.std(depths)),
        "n_leaves":          float(tree.n_leaves),
        "n_internal":        float(n_internal),
        "branching_ratio":   float(tree.branching_ratio),
        "asymmetry_index":   float(tree.asymmetry_index),
        "root_subtree_size": float(root_subtree),
        "depth_entropy":     float(entropy),
        "is_valid":          1.0,
    }


# ─── Convenience: full hierarchy analysis ────────────────────────────────────

def analyse_branch_hierarchy(
    vascular_graph,
    root_id: Optional[int] = None,
    max_depth: int = 50,
) -> Tuple[BranchTree, Dict[int, int], Dict[str, float]]:
    """
    One-call wrapper: build tree, annotate levels, compute stats.

    Args:
        vascular_graph: VascularGraph.
        root_id:        Optional explicit root.
        max_depth:      BFS depth cap.

    Returns:
        (BranchTree, level_map, stats_dict)
    """
    tree      = build_branch_hierarchy(vascular_graph, root_id=root_id, max_depth=max_depth)
    level_map = annotate_branch_levels(vascular_graph, tree)
    stats     = compute_hierarchy_stats(tree)
    return tree, level_map, stats
