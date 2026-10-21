# RAPiD-Twin: Retina-Anchored Physiological Digital Twin

A complete PyTorch ML system that builds a dynamic computational representation
of a patient's health state using retinal microvasculature as the primary anchor signal.

---

## Architecture Overview

```
Retina Camera
      ↓
Adaptive Retinal Preprocessing       [preprocessing/pipeline.py]
  ├─ Image Quality Assessment         (Laplacian variance, illumination, CNN)
  ├─ Green Channel Extraction
  ├─ Illumination Normalization       (morphological background subtraction)
  ├─ CLAHE Contrast Enhancement
  ├─ Noise Removal                    (Gaussian + median filter)
  └─ Frangi Vessel Enhancement        (Hessian eigenvalue analysis)
      ↓
Vessel Segmentation                  [segmentation/attention_unet.py]
  └─ Attention U-Net                  (deep supervision, attention gates)
      ↓
Retinal Graph Construction           [graph/graph_builder.py]
  ├─ Skeletonization
  ├─ Junction / Endpoint Detection
  ├─ Node features: branching angle, tortuosity, density
  └─ Edge features: diameter, curvature, length, intensity
      ↓
┌─────────────────────────────────┐
│   Parallel Encoding Streams     │
│                                 │
│  Vascular Graph                 │  [models/gnn.py]
│  → Graph Attention Network      │   GAT layers + edge-conditioned attention
│  → vascular_embedding (256d)    │   Global pooling (attn + mean + max)
│                                 │
│  Retinal Image                  │  [models/retinal_encoder.py]
│  → ConvNeXt Backbone            │   Local texture, lesions, hemorrhages
│  → Swin Transformer Head        │   Global macular structure
│  → retinal_embedding (256d)     │
│                                 │
│  Physiological Signals          │  [models/physio_encoder.py]
│  → Transformer Time-Series      │   HR, BP, SpO2, HRV, RR, Temp, steps
│  → physio_embedding (256d)      │   CLS token + temporal attention
└─────────────────────────────────┘
      ↓
Multimodal Fusion Transformer        [fusion/multimodal_fusion.py]
  ├─ Tokens: [graph | retina | physio]
  ├─ Cross-modal self-attention
  ├─ Learned modality importance weights
  └─ fused_embedding (256d)
      ↓
Temporal Fusion Transformer          [twin/temporal_twin.py]
  ├─ Learns patient-specific baseline
  ├─ Long-term trend modeling (LSTM + temporal attention)
  ├─ Anomaly deviation detection
  └─ health_state (256d)
      ↓
Multi-Task Disease Prediction        [models/disease_heads.py]
  ├─ Eye: Diabetic Retinopathy (5-class)
  ├─ Eye: Glaucoma (2-class)
  ├─ Eye: Macular Degeneration (4-class)
  ├─ Eye: Retinal Vein Occlusion (2-class)
  ├─ Systemic: Diabetes Risk (3-class)
  ├─ Systemic: Hypertension Risk (3-class)
  └─ Systemic: Cardiovascular Risk (4-class)
      ↓
Health Insight Engine
  ├─ vascular_health_score
  ├─ retinal_integrity_score
  ├─ metabolic_risk_score
  ├─ cardiovascular_risk_score
  └─ lifestyle_health_score
```

---

## Repository Structure

```
rapid_twin/
├── configs/
│   └── config.py                    # All hyperparameters (dataclasses)
│
├── preprocessing/
│   ├── quality_assessment.py        # Adaptive QA: blur, illumination, occlusion
│   └── pipeline.py                  # Full preprocessing: CLAHE, Frangi, normalization
│
├── segmentation/
│   └── attention_unet.py            # Attention U-Net + deep supervision + Dice+BCE loss
│
├── graph/
│   └── graph_builder.py             # Skeleton → vascular graph (nodes, edges, features)
│
├── models/
│   ├── gnn.py                       # GAT + Graph Transformer + global pooling
│   ├── retinal_encoder.py           # ConvNeXt + Swin Transformer hybrid
│   ├── physio_encoder.py            # TCN and Transformer time-series encoder
│   ├── disease_heads.py             # Multi-task heads + focal loss + uncertainty weighting
│   └── rapid_twin_model.py          # Integrated RAPiD-Twin orchestrator
│
├── fusion/
│   └── multimodal_fusion.py         # Cross-modal transformer + health state projector
│
├── twin/
│   └── temporal_twin.py             # TFT: GRN, variable selection, LSTM, temporal attn
│
├── training/
│   └── trainer.py                   # Seg trainer + main trainer + cosine LR + AMP
│
├── evaluation/
│   └── evaluator.py                 # AUC-ROC, Dice, IoU, sensitivity, specificity, F1
│
├── explainability/
│   └── gradcam.py                   # Grad-CAM + GNNExplainer + Integrated Gradients
│
├── datasets/
│   └── retinal_datasets.py          # DRIVE, EyePACS, APTOS + synthetic data + collate
│
├── utils/
│   ├── metrics.py                   # Dice, IoU, sensitivity, set_seed
│   └── visualization.py            # Overlay, graph drawing, health radar, report
│
├── train_segmentation.py            # Entry: train Attention U-Net
├── train_rapid_twin.py              # Entry: train full model
├── evaluate.py                      # Entry: evaluate + export JSON report
├── inference.py                     # Entry: single-patient inference + demo
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-org/rapid-twin.git
cd rapid-twin
pip install -r requirements.txt
```

**Core dependencies:**
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `opencv-python >= 4.7.0`
- `scikit-image >= 0.21.0`
- `scikit-learn >= 1.3.0`
- `networkx >= 3.0`
- `scipy >= 1.10.0`

---

## Quick Start

### 1. Run the inference demo (synthetic patient)

```bash
python -m rapid_twin.inference
```

### 2. Train vessel segmentation

```bash
# With real DRIVE/STARE data:
python -m rapid_twin.train_segmentation \
    --data_root /path/to/datasets \
    --epochs 100 \
    --batch_size 8 \
    --image_size 512

# With synthetic data (no dataset needed):
python -m rapid_twin.train_segmentation --synthetic --epochs 5 --image_size 128
```

### 3. Train full RAPiD-Twin

```bash
python -m rapid_twin.train_rapid_twin \
    --synthetic \
    --epochs 10 \
    --batch_size 4 \
    --image_size 128
```

### 4. Evaluate

```bash
python -m rapid_twin.evaluate \
    --checkpoint checkpoints/rapid_twin/best_rapid_twin.pt \
    --output evaluation_report.json
```

### 5. Inference on a patient image

```python
from rapid_twin.inference import RAPiDTwinInference
import numpy as np

engine = RAPiDTwinInference(
    model_checkpoint="checkpoints/rapid_twin/best_rapid_twin.pt",
    seg_checkpoint="checkpoints/segmentation/best_seg_model.pt",
    device="cuda",
    image_size=512
)

# Process single retinal image
result = engine.process_image("patient_fundus.jpg")

# With physiological signals: shape (T, 8) — HR, BP_sys, BP_dia, SpO2, RR, Temp, HRV, steps
physio = np.load("patient_vitals_24h.npy")   # (24, 8)
result = engine.process_with_physio("patient_fundus.jpg", physio)

# Get explainability report
explanation = engine.explain_prediction(image_rgb, target_task="diabetic_retinopathy")

# Digital twin updates automatically with each visit
result2 = engine.process_image("patient_fundus_6months_later.jpg")
```

---

## Supported Datasets

| Dataset     | Type           | Task                        | Images  |
|-------------|----------------|-----------------------------|---------|
| DRIVE       | Segmentation   | Vessel segmentation         | 40      |
| STARE       | Segmentation   | Vessel segmentation         | 20      |
| CHASE_DB1   | Segmentation   | Vessel segmentation         | 28      |
| EyePACS     | Classification | Diabetic retinopathy (0-4)  | 88,702  |
| APTOS 2019  | Classification | Diabetic retinopathy (0-4)  | 3,662   |
| Messidor-2  | Classification | Diabetic retinopathy (0-3)  | 1,748   |
| UK Biobank  | Multimodal     | Systemic disease signals    | 67,000+ |
| MIMIC-IV    | Physiological  | Vital sign time series      | 40,000+ |

---

## Expected Performance

| Task                          | Metric       | Expected Range |
|-------------------------------|--------------|----------------|
| Vessel Segmentation           | Dice         | 95–97%         |
| Vessel Segmentation           | IoU          | 90–94%         |
| Diabetic Retinopathy          | AUC-ROC      | 92–97%         |
| Glaucoma Detection            | AUC-ROC      | 88–92%         |
| Macular Degeneration          | AUC-ROC      | 85–91%         |
| Multimodal Systemic Risk      | AUC-ROC      | 82–90%         |

---

## Model Components

### Attention U-Net (Vessel Segmentation)
- Encoder: 4 ConvBlock stages with MaxPool
- Decoder: TransposedConv + Attention Gates on skip connections
- Deep supervision: auxiliary segmentation heads at 3 decoder levels
- Loss: weighted BCE + Dice (configurable) with positive class weighting

### Graph Neural Network (Vascular Topology)
- GATLayer: multi-head edge-conditioned attention, sparse message passing
- GraphTransformerLayer: full attention for long-range vascular interactions
- GlobalPooling: attention-weighted + mean + max pooling concatenated
- Output: 256-d vascular embedding capturing abnormal branching, narrowing, tortuosity

### Retinal Structure Encoder (Image Features)
- ConvNeXt-Tiny backbone: depthwise 7×7 conv, LayerNorm, inverted bottleneck
- Swin Transformer head: window-based attention with relative positional bias
- Multi-scale FPN: combines 3 intermediate feature maps via lateral connections
- Output: 256-d embedding capturing lesions, exudates, macular structure

### Physiological Signal Encoder
- Transformer: CLS token + sinusoidal positional encoding + temporal attention weights
- Alternative: TCN with dilated causal convolutions (exponential dilation: 1,2,4,8,16,32)
- Handles missing signals via learned per-channel imputation
- Output: 256-d embedding from 8 physiological channels × up to 168 timesteps

### Multimodal Fusion Transformer
- Tokenizes each modality (graph, image, physio) → projected to 512-d
- Learnable modality type embeddings distinguish signal sources
- 4 cross-modal self-attention layers
- Handles missing modalities via learned missing-value embeddings

### Temporal Fusion Transformer (Digital Twin)
- Gated Residual Networks (GRN) for stable gradient flow
- LSTM encoder captures short-term dynamics
- Temporal self-attention with causal masking
- Learns patient-specific baseline from early history
- Outputs: quantile predictions, anomaly score, updated health state

### Multi-Task Disease Heads
- Shared representation layer (2×Linear + LayerNorm + GELU)
- 7 independent task-specific heads (eye + systemic)
- Uncertainty-weighted loss balancing (Kendall et al. 2018)
- Focal loss for class imbalance (γ=2.0)
- Learnable temperature scaling for calibration

---

## Explainability

| Method               | Target              | Output                              |
|----------------------|---------------------|-------------------------------------|
| Grad-CAM             | Retinal image       | Spatial attention heatmap (H×W)     |
| GNNExplainer         | Vascular graph      | Edge/node importance masks          |
| Integrated Gradients | Image + physio      | Per-pixel + per-timestep attribution|
| Modality Ablation    | All modalities      | Per-modality importance score       |

---

## Digital Twin Health State

The twin maintains a 256-d health state vector updated at each visit.
This is projected to 5 named dimensions:

- **vascular_health** — retinal vessel integrity (tortuosity, caliber, density)
- **retinal_integrity** — structural retinal health (lesions, exudates, disc)
- **metabolic_risk** — diabetes and metabolic dysfunction indicators
- **cardiovascular_risk** — BP-related vascular changes + heart rate dynamics
- **lifestyle_health** — activity, sleep, behavioral signal patterns

The twin computes an **anomaly score** [0,1] measuring deviation from the
patient's personal baseline — enabling early warning before clinical symptoms.

---

## Configuration

All hyperparameters are in `configs/config.py` as Python dataclasses.
No YAML/JSON files needed — edit directly or override programmatically:

```python
from rapid_twin.configs.config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG
cfg.gnn.hidden_dim = 256
cfg.gnn.num_layers = 6
cfg.training.main_lr = 1e-4
cfg.physio_encoder.seq_len = 336   # 2 weeks of hourly readings
```

---

## Citation

If you use RAPiD-Twin in research, please cite:

```
RAPiD-Twin: Retina-Anchored Physiological Digital Twin
A multimodal graph-neural-network system for longitudinal health modeling
using retinal microvascular topology as the primary anchor signal.
```

---

## License

MIT License. See LICENSE file for details.
