"""
Module Registry — Stage 4, Research-Grade Platform.

Provides a plugin-style registration system for all RAPiD-Twin components.
Modules are registered by name and retrieved by name, enabling config-driven
pipeline composition without hard imports.

Features:
  - @register() decorator for declarative registration
  - GLOBAL_REGISTRY singleton shared across the platform
  - get_module() / list_modules() for runtime discovery
  - Schema validation hooks (optional)
  - Duplicate registration raises explicit ValueError

Design:
  - Thread-safe for read-only use (write at import time only)
  - No external dependencies
  - Works with any callable (class, function, factory)

Usage:
    from rapid_twin.platform.registry import register, get_module

    @register("my_segmentor", tags=["segmentation"])
    class MySegmentor:
        ...

    cls = get_module("my_segmentor")
    instance = cls(...)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Set, Type


# ─── Registry ─────────────────────────────────────────────────────────────────

class ModuleRegistry:
    """
    Central registry mapping string names to Python callables (classes or functions).

    Attributes:
        _registry:  dict of name → (callable, metadata dict).
        _tags:      dict of tag → set of names registered with that tag.
    """

    def __init__(self, name: str = "default"):
        self.name      = name
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._tags:     Dict[str, Set[str]]        = {}

    def register(
        self,
        name:        str,
        callable_:   Any,
        description: str        = "",
        tags:        List[str]  = None,
        config_schema: Optional[Dict[str, Any]] = None,
        allow_override: bool    = False,
    ) -> None:
        """
        Register a callable under a name.

        Args:
            name:            unique registry key.
            callable_:       class, function, or factory to register.
            description:     human-readable description.
            tags:            optional list of category tags.
            config_schema:   optional dict describing expected config keys.
            allow_override:  if False, raises ValueError on duplicate names.
        """
        if name in self._registry and not allow_override:
            raise ValueError(
                f"Module '{name}' already registered in registry '{self.name}'. "
                f"Set allow_override=True to replace it."
            )

        self._registry[name] = {
            "callable":     callable_,
            "description":  description,
            "tags":         list(tags or []),
            "config_schema": config_schema or {},
        }

        for tag in (tags or []):
            self._tags.setdefault(tag, set()).add(name)

    def get(self, name: str) -> Any:
        """
        Retrieve a registered callable by name.

        Args:
            name: registry key.

        Returns:
            The registered callable.

        Raises:
            KeyError: if name is not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Module '{name}' not found in registry '{self.name}'. "
                f"Available: {available}"
            )
        return self._registry[name]["callable"]

    def list_names(self, tag: Optional[str] = None) -> List[str]:
        """
        List all registered names, optionally filtered by tag.

        Args:
            tag: if given, only return names with this tag.

        Returns:
            Sorted list of names.
        """
        if tag is not None:
            return sorted(self._tags.get(tag, set()))
        return sorted(self._registry.keys())

    def list_tags(self) -> List[str]:
        """Return all registered tags."""
        return sorted(self._tags.keys())

    def describe(self, name: str) -> Dict[str, Any]:
        """Return metadata for a registered module."""
        if name not in self._registry:
            raise KeyError(f"Module '{name}' not in registry.")
        entry = dict(self._registry[name])
        entry.pop("callable")
        entry["name"] = name
        return entry

    def instantiate(self, name: str, **kwargs) -> Any:
        """
        Retrieve and instantiate a registered class with keyword arguments.

        Args:
            name:    registry key.
            **kwargs: constructor arguments.

        Returns:
            Instantiated object.
        """
        cls = self.get(name)
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate '{name}' with {list(kwargs.keys())}: {e}"
            ) from e

    def print_summary(self):
        """Print a formatted registry summary."""
        print(f"\n{'─'*60}")
        print(f"  ModuleRegistry '{self.name}'  ({len(self._registry)} modules)")
        print(f"{'─'*60}")
        for name in sorted(self._registry):
            entry = self._registry[name]
            tags  = ", ".join(entry["tags"]) if entry["tags"] else "—"
            desc  = entry["description"][:45] + "…" if len(entry["description"]) > 45 else entry["description"]
            print(f"  {name:<30}  [{tags:<20}]  {desc}")
        print()

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# ─── Global registry singleton ────────────────────────────────────────────────

GLOBAL_REGISTRY = ModuleRegistry(name="rapid_twin")


# ─── Convenience functions ────────────────────────────────────────────────────

def register(
    name:           str,
    tags:           Optional[List[str]] = None,
    description:    str = "",
    config_schema:  Optional[Dict[str, Any]] = None,
    allow_override: bool = False,
    registry:       ModuleRegistry = None,
) -> Callable:
    """
    Decorator to register a class or function in the global registry.

    Usage:
        @register("retinal_graph_builder", tags=["graph", "extraction"])
        class RetinalGraphBuilder:
            ...

        @register("biomarker_extractor", tags=["biomarkers"])
        class RetinalBiomarkerExtractor:
            ...
    """
    _reg = registry if registry is not None else GLOBAL_REGISTRY

    def decorator(callable_: Any) -> Any:
        _reg.register(
            name=name,
            callable_=callable_,
            description=description or (callable_.__doc__ or "")[:80],
            tags=tags or [],
            config_schema=config_schema,
            allow_override=allow_override,
        )
        return callable_

    return decorator


def get_module(name: str, registry: ModuleRegistry = None) -> Any:
    """Retrieve a module from the global registry by name."""
    _reg = registry if registry is not None else GLOBAL_REGISTRY
    return _reg.get(name)


def list_modules(tag: Optional[str] = None, registry: ModuleRegistry = None) -> List[str]:
    """List all registered module names (optionally filtered by tag)."""
    _reg = registry if registry is not None else GLOBAL_REGISTRY
    return _reg.list_names(tag)


# ─── Pre-register core platform modules ──────────────────────────────────────

def _register_core_modules():
    """
    Register the core RAPiD-Twin modules with the global registry.
    Uses lazy imports to avoid circular dependencies.
    """
    registrations = [
        ("retinal_graph_builder",    "rapid_twin.graph.graph_builder",   "RetinalGraphBuilder",    ["graph", "extraction"]),
        ("vascular_graph_refiner",   "rapid_twin.graph.graph_refinement","VascularGraphRefiner",   ["graph", "refinement"]),
        ("vascular_graph_pipeline",  "rapid_twin.graph.graph_pipeline",  "VascularGraphPipeline",  ["graph", "pipeline"]),
        ("av_classifier",            "rapid_twin.av_classification",     "AVClassifier",           ["av", "classification"]),
        ("biomarker_extractor",      "rapid_twin.biomarkers",            "RetinalBiomarkerExtractor", ["biomarkers"]),
        ("hemodynamic_model",        "rapid_twin.hemodynamics",          "HemodynamicModel",       ["hemodynamics"]),
        ("multiscale_analyzer",      "rapid_twin.graph.multiscale",      "MultiScaleVesselAnalyzer", ["graph", "multiscale"]),
        ("spatial_analyzer",         "rapid_twin.biomarkers.spatial_analysis", "SpatialVascularAnalyzer", ["biomarkers", "spatial"]),
        ("longitudinal_twin",        "rapid_twin.twin.longitudinal",     "LongitudinalVascularTwin", ["twin", "longitudinal"]),
        ("physiological_simulator",  "rapid_twin.simulation",            "PhysiologicalSimulator", ["simulation"]),
        ("graph_stability_analyzer", "rapid_twin.graph.graph_robustness","GraphStabilityAnalyzer", ["graph", "robustness"]),
        ("calibration_analyzer",     "rapid_twin.reliability",           "CalibrationAnalyzer",    ["reliability"]),
        ("ood_detector",             "rapid_twin.reliability",           "OODDetector",            ["reliability"]),
        ("benchmark_runner",         "rapid_twin.benchmarks",            "UnifiedBenchmarkRunner", ["benchmarks"]),
    ]

    for reg_name, module_path, class_name, tags in registrations:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            GLOBAL_REGISTRY.register(
                name=reg_name, callable_=cls,
                description=class_name, tags=tags,
                allow_override=True,
            )
        except (ImportError, AttributeError):
            pass   # graceful — optional components may not be available


# Register on import
_register_core_modules()
