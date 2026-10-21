"""RAPiD-Twin Global Configuration"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PreprocessingConfig:
    image_size: Tuple[int, int] = (512, 512)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    gaussian_kernel: int = 5
    median_kernel: int = 3
    frangi_sigmas: List[float] = field(default_factory=lambda: [1, 2, 4, 8])
    quality_blur_threshold: float = 100.0
    quality_illumination_threshold: float = 0.3


@dataclass
class SegmentationConfig:
    in_channels: int = 3
    out_channels: int = 1
    features: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    dropout: float = 0.1
    attention: bool = True
    deep_supervision: bool = True


@dataclass
class GraphConfig:
    min_branch_length: int = 10
    max_nodes: int = 512
    node_feature_dim: int = 16
    edge_feature_dim: int = 8


@dataclass
class GNNConfig:
    node_in_dim: int = 16
    edge_in_dim: int = 8
    hidden_dim: int = 128
    out_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1


@dataclass
class RetinalEncoderConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    embed_dim: int = 256
    fpn_dim: int = 256
    window_size: int = 7
    dropout: float = 0.1


@dataclass
class PhysioEncoderConfig:
    input_dim: int = 8
    hidden_dim: int = 128
    embed_dim: int = 256
    seq_len: int = 168
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    use_tcn: bool = False


@dataclass
class FusionConfig:
    modality_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    fusion_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    output_dim: int = 256
    use_qformer: bool = False
    num_queries: int = 8


@dataclass
class TwinConfig:
    state_dim: int = 256
    hidden_dim: int = 512
    num_quantiles: int = 7
    num_heads: int = 8
    num_encoder_steps: int = 24
    dropout: float = 0.1
    seq_len: int = 52
    memory_size: int = 52


@dataclass
class DiseaseHeadConfig:
    eye_tasks: List[str] = field(default_factory=lambda: [
        "diabetic_retinopathy", "glaucoma", "macular_degeneration", "retinal_vein_occlusion"])
    eye_classes: List[int] = field(default_factory=lambda: [5, 2, 4, 2])
    systemic_tasks: List[str] = field(default_factory=lambda: [
        "diabetes", "hypertension", "cardiovascular"])
    systemic_classes: List[int] = field(default_factory=lambda: [3, 3, 4])
    input_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    seg_lr: float = 1e-4
    seg_epochs: int = 100
    seg_batch_size: int = 8
    seg_weight_decay: float = 1e-5
    main_lr: float = 3e-4
    main_epochs: int = 150
    main_batch_size: int = 16
    main_weight_decay: float = 1e-4
    warmup_epochs: int = 10
    seg_loss_weight: float = 1.0
    eye_disease_weight: float = 1.0
    systemic_risk_weight: float = 0.8
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    aug_rotation: float = 30.0
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_flip: bool = True
    aug_elastic: bool = True
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    patience: int = 15
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_root: str = "data"
    gradient_checkpointing: bool = False
    amp: bool = True
    experiment_tracker: str = "none"
    label_smoothing: float = 0.1


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    model_checkpoint: str = ""
    seg_checkpoint: str = ""
    device: str = "auto"
    image_size: int = 512
    max_upload_mb: int = 20


@dataclass
class RAPiDTwinConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    retinal_encoder: RetinalEncoderConfig = field(default_factory=RetinalEncoderConfig)
    physio_encoder: PhysioEncoderConfig = field(default_factory=PhysioEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    disease_head: DiseaseHeadConfig = field(default_factory=DiseaseHeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)


DEFAULT_CONFIG = RAPiDTwinConfig()
