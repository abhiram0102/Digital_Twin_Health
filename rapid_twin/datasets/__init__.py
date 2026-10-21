# Dataset imports require torch
try:
    from .retinal_datasets import (
        SyntheticRAPiDDataset, EyePACSDataset, DRIVEDataset,
        CombinedSegDataset, RetinalAugmentation, collate_graph_batch,
        get_synthetic_loaders, get_segmentation_loaders
    )
except ImportError:
    pass
