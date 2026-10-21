try:
    from .temporal_twin import TemporalFusionTransformer, DigitalTwinStateTracker
except ImportError:
    pass  # torch not available — PyTorch-based twin models disabled

from .longitudinal import (
    RetinalVisit,
    GraphAlignment,
    BiomarkerTrend,
    ProgressionReport,
    PatientTwinState,
    LongitudinalVascularTwin,
    align_graphs,
    estimate_biomarker_trends,
)
