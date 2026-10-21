"""
Config-Driven Pipeline Configuration — Stage 4, Research-Grade Platform.

Enables declarative, reproducible pipeline definitions via Python dicts or
YAML-compatible structures.  No YAML parsing dependency — configs are passed
as plain Python dicts and can be serialized/deserialized by the caller.

Provides:
  StageConfig      — one pipeline stage (module name + kwargs)
  PipelineConfig   — ordered list of stages + global settings
  load_config_from_dict — validate and build PipelineConfig from a dict

Usage:
    config = load_config_from_dict({
        "name": "standard_retinal_pipeline",
        "dataset": "DRIVE",
        "stages": [
            {"name": "graph_build",    "module": "retinal_graph_builder",
             "kwargs": {"min_branch_length": 5, "max_nodes": 512}},
            {"name": "refinement",     "module": "vascular_graph_refiner", "kwargs": {}},
            {"name": "av_classify",    "module": "av_classifier",          "kwargs": {}},
            {"name": "biomarkers",     "module": "biomarker_extractor",    "kwargs": {}},
            {"name": "hemodynamics",   "module": "hemodynamic_model",
             "kwargs": {"root_pressure": 80.0}},
        ],
        "reliability": {"mc_dropout_samples": 20},
        "benchmark":   {"dataset": "DRIVE"},
    })

    print(config.name, config.stage_names)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageConfig:
    """
    Configuration for one pipeline stage.

    Attributes:
        name:    human-readable stage label (used in logs and profiling).
        module:  name of a registered module in the ModuleRegistry.
        kwargs:  constructor/call arguments for this module.
        enabled: if False, this stage is skipped.
        timeout_s: optional per-stage timeout in seconds (advisory).
    """
    name:      str
    module:    str
    kwargs:    Dict[str, Any] = field(default_factory=dict)
    enabled:   bool           = True
    timeout_s: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("StageConfig.name must be a non-empty string.")
        if not self.module:
            raise ValueError("StageConfig.module must be a non-empty string.")


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration.

    Attributes:
        name:        human-readable pipeline name.
        dataset:     target dataset identifier string.
        stages:      ordered list of StageConfig.
        reliability: reliability layer settings.
        benchmark:   benchmark suite settings.
        experiment:  experiment tracking settings.
        metadata:    arbitrary key-value metadata.
    """
    name:        str
    dataset:     str                  = "unknown"
    stages:      List[StageConfig]    = field(default_factory=list)
    reliability: Dict[str, Any]       = field(default_factory=dict)
    benchmark:   Dict[str, Any]       = field(default_factory=dict)
    experiment:  Dict[str, Any]       = field(default_factory=dict)
    metadata:    Dict[str, Any]       = field(default_factory=dict)

    @property
    def stage_names(self) -> List[str]:
        """Names of all enabled stages."""
        return [s.name for s in self.stages if s.enabled]

    @property
    def active_stages(self) -> List[StageConfig]:
        """All enabled stages."""
        return [s for s in self.stages if s.enabled]

    def get_stage(self, name: str) -> Optional[StageConfig]:
        """Return the StageConfig for a given stage name, or None."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict (suitable for JSON / YAML export)."""
        return {
            "name":        self.name,
            "dataset":     self.dataset,
            "stages":      [
                {"name": s.name, "module": s.module,
                 "kwargs": s.kwargs, "enabled": s.enabled}
                for s in self.stages
            ],
            "reliability": self.reliability,
            "benchmark":   self.benchmark,
            "experiment":  self.experiment,
            "metadata":    self.metadata,
        }

    def print_summary(self):
        print(f"\n{'─'*55}")
        print(f"  Pipeline: {self.name}   Dataset: {self.dataset}")
        print(f"  Stages ({len(self.active_stages)} active):")
        for s in self.stages:
            status = "ON " if s.enabled else "OFF"
            print(f"    [{status}]  {s.name:<25}  → {s.module}")
        if self.reliability:
            print(f"  Reliability: {self.reliability}")
        if self.benchmark:
            print(f"  Benchmark:   {self.benchmark}")
        print()


def load_config_from_dict(raw: Dict[str, Any]) -> PipelineConfig:
    """
    Build and validate a PipelineConfig from a plain Python dict.

    Args:
        raw: dict with keys matching PipelineConfig fields.
             "stages" must be a list of dicts with "name" and "module" keys.

    Returns:
        PipelineConfig.

    Raises:
        ValueError: on missing required keys or invalid types.
        TypeError:  if raw is not a dict.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Config must be a dict, got {type(raw).__name__}.")
    if "name" not in raw:
        raise ValueError("Config dict must have a 'name' key.")

    stages: List[StageConfig] = []
    for i, sd in enumerate(raw.get("stages", [])):
        if "name" not in sd:
            raise ValueError(f"Stage {i} missing 'name' key.")
        if "module" not in sd:
            raise ValueError(f"Stage {i} ('{sd['name']}') missing 'module' key.")
        stages.append(StageConfig(
            name=sd["name"],
            module=sd["module"],
            kwargs=sd.get("kwargs", {}),
            enabled=sd.get("enabled", True),
            timeout_s=sd.get("timeout_s"),
        ))

    return PipelineConfig(
        name=raw["name"],
        dataset=raw.get("dataset", "unknown"),
        stages=stages,
        reliability=raw.get("reliability", {}),
        benchmark=raw.get("benchmark", {}),
        experiment=raw.get("experiment", {}),
        metadata=raw.get("metadata", {}),
    )


# ─── Standard pipeline presets ────────────────────────────────────────────────

def standard_pipeline_config(dataset: str = "DRIVE") -> PipelineConfig:
    """
    Return the standard RAPiD-Twin inference pipeline configuration.

    Args:
        dataset: dataset name for benchmark labeling.

    Returns:
        PipelineConfig with all core stages enabled.
    """
    return load_config_from_dict({
        "name":    "rapid_twin_standard",
        "dataset": dataset,
        "stages": [
            {"name": "graph_build",       "module": "retinal_graph_builder",
             "kwargs": {"min_branch_length": 5, "max_nodes": 512}},
            {"name": "graph_refinement",  "module": "vascular_graph_refiner",  "kwargs": {}},
            {"name": "multiscale",        "module": "multiscale_analyzer",
             "kwargs": {"large_thresh": 6.0, "capillary_thresh": 3.0}},
            {"name": "av_classify",       "module": "av_classifier",           "kwargs": {}},
            {"name": "biomarkers",        "module": "biomarker_extractor",     "kwargs": {}},
            {"name": "hemodynamics",      "module": "hemodynamic_model",
             "kwargs": {"root_pressure": 80.0}},
        ],
        "reliability": {"mc_dropout_samples": 20, "ece_threshold": 0.05},
        "benchmark":   {"dataset": dataset},
        "experiment":  {"track": True, "log_interval_steps": 10},
    })
