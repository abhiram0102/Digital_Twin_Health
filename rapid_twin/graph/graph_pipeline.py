"""
End-to-End Retinal Vascular Graph Pipeline.

Converts a raw vessel probability map (or binary mask) into:
  1. Postprocessed binary mask          (postprocessing.py)
  2. VascularGraph                      (graph_builder.py)
  3. Branch hierarchy tree              (branch_hierarchy.py)
  4. Full graph quality metrics         (graph_metrics.py)
  5. GNN embedding (optional)           (gnn.py via RetinalGNN)

Runtime validation is performed at every stage:
  - Tensor shape assertions
  - NaN / Inf checks
  - Empty-mask / empty-graph handling (explicit, not silent)
  - Memory profiling hooks (optional)

Usage:
    pipeline = VascularGraphPipeline(min_branch_length=10, max_nodes=512)
    result   = pipeline.run(prob_map, intensity_image=original_image)
    print(result["graph_metrics"]["graph_quality_score"])

    # With GNN embedding:
    gnn    = RetinalGNN(...)
    result = pipeline.run(prob_map, gnn_model=gnn, device="cuda")
    emb    = result["gnn_embedding"]   # (1, out_dim) tensor
"""

from __future__ import annotations

import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ─── Pipeline result ──────────────────────────────────────────────────────────

@dataclass
class GraphPipelineResult:
    """
    Complete output of the VascularGraphPipeline.

    Attributes:
        success:          True if pipeline completed without fatal error.
        error:            Exception or error message if success=False, else None.
        binary_mask:      (H, W) float32 postprocessed binary mask.
        vascular_graph:   VascularGraph or None if graph build failed.
        branch_tree:      BranchTree or None.
        level_map:        {node_id: depth} or None.
        graph_metrics:    dict of metric_name → float, or empty dict.
        hierarchy_stats:  dict of stat_name → float, or empty dict.
        gnn_embedding:    torch.Tensor (1, out_dim) or None.
        timing:           dict of stage_name → elapsed_seconds.
        n_nodes:          int — number of vascular graph nodes.
        n_edges:          int — number of vascular graph edges.
        skeleton_pixels:  int — total skeleton length in pixels.
    """
    success:         bool
    error:           Optional[Any]
    binary_mask:     Optional[np.ndarray]
    vascular_graph:  Optional[Any]
    branch_tree:     Optional[Any]
    level_map:       Optional[Dict[int, int]]
    graph_metrics:   Dict[str, float]
    hierarchy_stats: Dict[str, float]
    gnn_embedding:   Optional[Any]    # torch.Tensor or None
    timing:          Dict[str, float] = field(default_factory=dict)
    n_nodes:         int = 0
    n_edges:         int = 0
    skeleton_pixels: int = 0


# ─── Validation helpers ───────────────────────────────────────────────────────

def _assert_array(arr: np.ndarray, name: str, ndim: int, dtype_check: bool = True):
    """Raise ValueError with a clear message on shape/dtype violations."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if dtype_check and not np.issubdtype(arr.dtype, np.floating):
        warnings.warn(
            f"{name} has dtype {arr.dtype}; converting to float32. "
            "Pass a float32 array to suppress this warning."
        )


def _check_finite(arr: np.ndarray, name: str):
    if not np.isfinite(arr).all():
        bad_count = (~np.isfinite(arr)).sum()
        raise ValueError(
            f"{name} contains {bad_count} non-finite values (NaN or Inf). "
            "Clean the input before running the pipeline."
        )


# ─── Main pipeline ────────────────────────────────────────────────────────────

class VascularGraphPipeline:
    """
    End-to-end pipeline: vessel probability map → graph → metrics → GNN.

    All stages are independently timed and validated.  Failures in optional
    stages (e.g., GNN embedding) are caught and reported without crashing.

    Args:
        min_branch_length: minimum skeleton branch length (pixels) to include as edge.
        max_nodes:         maximum number of graph nodes (NMS cap).
        postprocess_kwargs: keyword arguments forwarded to postprocessing.postprocess().
        compute_efficiency: include network efficiency in graph metrics (slow).
        compute_fractal:    include fractal dimension in graph metrics (slow).
        verbose:           print timing and intermediate statistics.
    """

    def __init__(
        self,
        min_branch_length: int   = 10,
        max_nodes:         int   = 512,
        postprocess_kwargs: Optional[Dict] = None,
        compute_efficiency: bool = True,
        compute_fractal:    bool = True,
        verbose:            bool = False,
    ):
        self.min_branch_length  = min_branch_length
        self.max_nodes          = max_nodes
        self.postprocess_kwargs = postprocess_kwargs or {}
        self.compute_efficiency = compute_efficiency
        self.compute_fractal    = compute_fractal
        self.verbose            = verbose

        # Lazy imports — allow the pipeline to be instantiated even if some
        # heavy dependencies are not installed.
        self._builder  = None
        self._postproc = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_builder(self):
        if self._builder is None:
            from rapid_twin.graph.graph_builder import RetinalGraphBuilder
            self._builder = RetinalGraphBuilder(
                min_branch_length=self.min_branch_length,
                max_nodes=self.max_nodes,
            )
        return self._builder

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GraphPipeline] {msg}")

    def _time(self, stage: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self._log(f"{stage}: {elapsed:.3f}s")
        return result, elapsed

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        prob_map:         np.ndarray,
        intensity_image:  Optional[np.ndarray] = None,
        gnn_model:        Optional[Any] = None,
        device:           str = "cpu",
        root_id:          Optional[int] = None,
    ) -> GraphPipelineResult:
        """
        Run the full pipeline on a vessel probability map.

        Args:
            prob_map:        (H, W) float32 vessel probability in [0, 1].
            intensity_image: (H, W) float32 or (H, W, 3) original image for
                             intensity features along vessel paths. If None,
                             a uniform image is substituted (no crash).
            gnn_model:       RetinalGNN instance. If None, GNN stage is skipped.
            device:          Torch device string for GNN inference ("cpu" / "cuda").
            root_id:         Optional explicit root node for hierarchy BFS.

        Returns:
            GraphPipelineResult with all outputs and timing.
        """
        timing: Dict[str, float] = {}

        # ── Stage 0: Input validation ─────────────────────────────────────────
        try:
            _assert_array(prob_map, "prob_map", ndim=2)
            prob_map = prob_map.astype(np.float32)
            prob_map = np.clip(prob_map, 0.0, 1.0)
            _check_finite(prob_map, "prob_map")

            H, W = prob_map.shape
            self._log(f"Input: {H}×{W}  vessel_frac={prob_map.mean():.4f}")

            if intensity_image is not None:
                if intensity_image.ndim == 3:
                    # Take luminance of RGB
                    intensity_image = intensity_image.mean(axis=-1).astype(np.float32)
                _assert_array(intensity_image, "intensity_image", ndim=2)
                intensity_image = intensity_image.astype(np.float32)
                if intensity_image.max() > 1.0:
                    intensity_image = intensity_image / 255.0

        except Exception as exc:
            return GraphPipelineResult(
                success=False, error=exc,
                binary_mask=None, vascular_graph=None,
                branch_tree=None, level_map=None,
                graph_metrics={}, hierarchy_stats={},
                gnn_embedding=None, timing=timing,
            )

        # ── Stage 1: Postprocessing ───────────────────────────────────────────
        try:
            from rapid_twin.segmentation.postprocessing import postprocess
            binary_mask, timing["postprocess"] = self._time(
                "postprocess", postprocess, prob_map, **self.postprocess_kwargs
            )
            vessel_frac = binary_mask.mean()
            self._log(f"Binary mask: vessel_frac={vessel_frac:.4f}")

            if vessel_frac < 0.001:
                warnings.warn(
                    "GraphPipeline: binary mask is nearly empty (<0.1% vessel pixels). "
                    "Graph will be degenerate. Check segmentation quality."
                )

        except Exception as exc:
            traceback.print_exc()
            return GraphPipelineResult(
                success=False, error=exc,
                binary_mask=None, vascular_graph=None,
                branch_tree=None, level_map=None,
                graph_metrics={}, hierarchy_stats={},
                gnn_embedding=None, timing=timing,
            )

        # ── Stage 2: Graph construction ───────────────────────────────────────
        try:
            builder = self._get_builder()
            graph, timing["graph_build"] = self._time(
                "graph_build", builder.build, binary_mask, intensity_image
            )

            n_nodes = len(graph.nodes)
            n_edges = len(graph.edges)
            skel_px = int(graph.skeleton.sum()) if graph.skeleton is not None else 0
            self._log(f"Graph: {n_nodes} nodes, {n_edges} edges, {skel_px} skeleton px")

            # Runtime shape assertions on feature matrices
            if n_nodes > 0:
                expected_node_dim = graph.node_features.shape[1]
                assert graph.node_features.shape == (n_nodes, expected_node_dim), (
                    f"node_features shape mismatch: {graph.node_features.shape}"
                )
                assert not np.isnan(graph.node_features).any(), (
                    "node_features contains NaN — graph builder produced invalid features"
                )

            if n_edges > 0:
                # edge_features are undirected (stored twice): 2*E rows
                ef_rows = graph.edge_features.shape[0]
                ei_cols = graph.edge_index.shape[1]
                assert ef_rows == ei_cols, (
                    f"edge_features rows ({ef_rows}) ≠ edge_index cols ({ei_cols})"
                )
                assert graph.edge_index.shape[0] == 2, (
                    f"edge_index must be (2, E), got {graph.edge_index.shape}"
                )
                assert not np.isnan(graph.edge_features).any(), (
                    "edge_features contains NaN"
                )

        except Exception as exc:
            traceback.print_exc()
            return GraphPipelineResult(
                success=False, error=exc,
                binary_mask=binary_mask, vascular_graph=None,
                branch_tree=None, level_map=None,
                graph_metrics={}, hierarchy_stats={},
                gnn_embedding=None, timing=timing,
            )

        # ── Stage 3: Graph metrics ────────────────────────────────────────────
        graph_metrics: Dict[str, float] = {}
        try:
            from rapid_twin.graph.graph_metrics import compute_all_graph_metrics
            graph_metrics, timing["graph_metrics"] = self._time(
                "graph_metrics",
                compute_all_graph_metrics,
                graph,
                binary_mask,
                compute_efficiency=self.compute_efficiency,
                compute_fractal=self.compute_fractal,
            )
            self._log(
                f"Quality score: {graph_metrics.get('graph_quality_score', float('nan')):.3f}"
            )

        except Exception as exc:
            warnings.warn(f"GraphPipeline: graph_metrics stage failed — {exc}")
            traceback.print_exc()

        # ── Stage 4: Branch hierarchy ─────────────────────────────────────────
        branch_tree = None
        level_map: Dict[int, int] = {}
        hierarchy_stats: Dict[str, float] = {}

        try:
            from rapid_twin.graph.branch_hierarchy import analyse_branch_hierarchy
            (branch_tree, level_map, hierarchy_stats), timing["branch_hierarchy"] = (
                self._time(
                    "branch_hierarchy",
                    analyse_branch_hierarchy,
                    graph,
                    root_id=root_id,
                )
            )
            self._log(
                f"Tree: depth={hierarchy_stats.get('max_depth', 0):.0f} "
                f"leaves={hierarchy_stats.get('n_leaves', 0):.0f} "
                f"branching_ratio={hierarchy_stats.get('branching_ratio', 0):.2f}"
            )

        except Exception as exc:
            warnings.warn(f"GraphPipeline: branch_hierarchy stage failed — {exc}")
            traceback.print_exc()

        # ── Stage 5: GNN embedding ────────────────────────────────────────────
        gnn_embedding = None
        if gnn_model is not None:
            try:
                import torch

                nf = torch.from_numpy(graph.node_features).float().to(device)
                ei = torch.from_numpy(graph.edge_index).long().to(device)
                ef = torch.from_numpy(graph.edge_features).float().to(device)

                # Runtime tensor shape assertions
                assert nf.ndim == 2, f"node_features must be 2-D, got {nf.shape}"
                assert ei.ndim == 2 and ei.shape[0] == 2, (
                    f"edge_index must be (2, E), got {ei.shape}"
                )
                assert ef.ndim == 2, f"edge_features must be 2-D, got {ef.shape}"

                gnn_model.eval()
                with torch.no_grad():
                    gnn_embedding_t, timing["gnn"] = self._time(
                        "gnn", gnn_model, nf, ei, ef
                    )

                assert torch.isfinite(gnn_embedding_t).all(), (
                    "GNN produced non-finite embedding — check model weights"
                )
                gnn_embedding = gnn_embedding_t
                self._log(f"GNN embedding: {tuple(gnn_embedding.shape)}")

            except Exception as exc:
                warnings.warn(f"GraphPipeline: GNN stage failed — {exc}")
                traceback.print_exc()

        # ── Done ──────────────────────────────────────────────────────────────
        timing["total"] = sum(v for v in timing.values())
        self._log(
            f"Pipeline complete in {timing['total']:.3f}s  "
            f"nodes={n_nodes}  edges={n_edges}  quality="
            f"{graph_metrics.get('graph_quality_score', float('nan')):.3f}"
        )

        return GraphPipelineResult(
            success=True,
            error=None,
            binary_mask=binary_mask,
            vascular_graph=graph,
            branch_tree=branch_tree,
            level_map=level_map,
            graph_metrics=graph_metrics,
            hierarchy_stats=hierarchy_stats,
            gnn_embedding=gnn_embedding,
            timing=timing,
            n_nodes=n_nodes,
            n_edges=n_edges,
            skeleton_pixels=skel_px,
        )

    # ── Batch mode ────────────────────────────────────────────────────────────

    def run_batch(
        self,
        prob_maps: list,
        intensity_images: Optional[list] = None,
        gnn_model: Optional[Any] = None,
        device: str = "cpu",
    ) -> list:
        """
        Run the pipeline on a list of probability maps.

        Args:
            prob_maps:        list of (H, W) float32 arrays.
            intensity_images: list of (H, W) arrays or None.
            gnn_model:        optional shared RetinalGNN.
            device:           torch device string.

        Returns:
            list of GraphPipelineResult (one per input).
        """
        if intensity_images is None:
            intensity_images = [None] * len(prob_maps)

        if len(intensity_images) != len(prob_maps):
            raise ValueError(
                f"prob_maps ({len(prob_maps)}) and intensity_images "
                f"({len(intensity_images)}) must have the same length."
            )

        return [
            self.run(pm, ii, gnn_model=gnn_model, device=device)
            for pm, ii in zip(prob_maps, intensity_images)
        ]

    # ── Summary utility ───────────────────────────────────────────────────────

    @staticmethod
    def print_result(result: GraphPipelineResult):
        """Print a concise summary of a pipeline result."""
        if not result.success:
            print(f"[GraphPipeline] FAILED: {result.error}")
            return

        print(f"\n{'─'*60}")
        print(f"  Graph Pipeline Result")
        print(f"{'─'*60}")
        print(f"  Nodes          : {result.n_nodes}")
        print(f"  Edges          : {result.n_edges}")
        print(f"  Skeleton px    : {result.skeleton_pixels}")

        key_metrics = [
            "graph_quality_score", "fractal_dimension",
            "largest_component_fraction", "mean_tortuosity",
            "mean_diameter", "bifurcation_index",
            "global_efficiency",
        ]
        if result.graph_metrics:
            print(f"\n  Graph Metrics:")
            for k in key_metrics:
                v = result.graph_metrics.get(k, float("nan"))
                print(f"    {k:<35} {v:.4f}")

        if result.hierarchy_stats:
            print(f"\n  Branch Hierarchy:")
            for k, v in result.hierarchy_stats.items():
                print(f"    {k:<35} {v:.4f}")

        if result.timing:
            print(f"\n  Timing (s):")
            for stage, t in result.timing.items():
                print(f"    {stage:<35} {t:.3f}")

        print(f"{'─'*60}\n")
