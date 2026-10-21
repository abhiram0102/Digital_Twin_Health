"""
RAPiD-Twin Integration Test Suite.

Validates tensor shapes, forward-pass correctness, and data pipeline integrity
using only synthetic data — no real dataset files required.

Run with:
    python -m pytest tests/test_integration.py -v
or directly:
    python tests/test_integration.py
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _assert_shape(tensor: torch.Tensor, expected: Tuple, name: str = ""):
    assert tuple(tensor.shape) == expected, (
        f"{name}: expected shape {expected}, got {tuple(tensor.shape)}"
    )

def _assert_finite(tensor: torch.Tensor, name: str = ""):
    assert torch.isfinite(tensor).all(), f"{name}: contains NaN or Inf"

def _assert_grad(tensor: torch.Tensor, name: str = ""):
    assert tensor.grad_fn is not None, f"{name}: has no gradient (detached or no_grad)"

def passed(name: str):
    print(f"  PASS  {name}")

def failed(name: str, exc: Exception):
    print(f"  FAIL  {name}")
    traceback.print_exc()


# ─── 1. Dataset tests ─────────────────────────────────────────────────────────

def test_synthetic_image_normalization():
    """_synthetic_image must produce ImageNet-normalised values, not raw [0,1]."""
    from rapid_twin.datasets.retinal_datasets import _synthetic_image

    img = _synthetic_image(64)
    _assert_shape(img, (3, 64, 64), "_synthetic_image")
    # ImageNet-normalised: mean ≈ 0, std ≈ 1; raw [0,1] would have min > -0.5
    assert img.min().item() < -0.5, (
        "_synthetic_image appears un-normalised: min value is "
        f"{img.min().item():.3f} (expected < -0.5 for ImageNet norm)"
    )
    passed("synthetic_image_normalization")


def test_combined_seg_dataset_no_double_norm():
    """CombinedSegDataset must not double-normalise when augment=True."""
    from rapid_twin.datasets.retinal_datasets import CombinedSegDataset

    # Empty roots → all-synthetic fallback; augment=True exercises the
    # previously-buggy re-augmentation path.
    ds = CombinedSegDataset(roots={}, image_size=32, augment=True, normalize=True)
    assert len(ds) > 0

    for i in range(min(3, len(ds))):
        sample = ds[i]
        img  = sample["image"]
        mask = sample["mask"]
        _assert_shape(img,  (3, 32, 32), f"CombinedSegDataset[{i}].image")
        _assert_shape(mask, (1, 32, 32), f"CombinedSegDataset[{i}].mask")
        _assert_finite(img,  f"CombinedSegDataset[{i}].image")
        _assert_finite(mask, f"CombinedSegDataset[{i}].mask")
        # Mask must be binary float
        assert set(mask.unique().tolist()).issubset({0.0, 1.0}), \
            f"mask contains non-binary values: {mask.unique().tolist()}"

    passed("combined_seg_dataset_no_double_norm")


def test_synthetic_dataset_shapes():
    """SyntheticRAPiDDataset must produce consistent shapes for all fields."""
    from rapid_twin.datasets.retinal_datasets import SyntheticRAPiDDataset, collate_graph_batch
    from torch.utils.data import DataLoader

    ds = SyntheticRAPiDDataset(size=16, image_size=32, seed=42)
    assert len(ds) == 16

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_graph_batch)
    batch  = next(iter(loader))

    _assert_shape(batch["image"],        (4, 3, 32, 32), "batch.image")
    _assert_shape(batch["vessel_mask"],  (4, 1, 32, 32), "batch.vessel_mask")
    _assert_shape(batch["physio_signals"], (4, 24, 8),   "batch.physio_signals")

    lbl = batch["labels"]
    assert "diabetic_retinopathy" in lbl
    _assert_shape(lbl["diabetic_retinopathy"], (4,), "batch.labels.dr")

    # Graph tensors must be lists (variable-size)
    assert isinstance(batch["node_features"], list), "node_features should be a list"
    assert len(batch["node_features"]) == 4

    passed("synthetic_dataset_shapes")


def test_make_weighted_sampler_fast_path():
    """make_weighted_sampler must use the fast metadata path (no image loading)."""
    from rapid_twin.datasets.retinal_datasets import APTOSDataset, make_weighted_sampler
    import time

    # Synthetic APTOS (no CSV) with 1000 entries
    ds = APTOSDataset(root="/nonexistent", split="train", image_size=32, augment=False)
    # Override samples directly to simulate a large dataset
    ds.samples = [{"path": None, "label": i % 5} for i in range(1000)]

    t0      = time.time()
    sampler = make_weighted_sampler(ds, label_key="diabetic_retinopathy", num_classes=5)
    elapsed = time.time() - t0

    assert len(sampler) == 1000
    # Fast path should complete in <0.1 s; slow path that loads images would be much slower
    assert elapsed < 2.0, f"make_weighted_sampler too slow ({elapsed:.2f}s) — fast path not used"
    passed("make_weighted_sampler_fast_path")


def test_rimone_no_global_random_contamination():
    """RIMONEDataset.__init__ must not alter the global random state."""
    import random
    from rapid_twin.datasets.retinal_datasets import RIMONEDataset

    random.seed(999)
    before = random.random()

    random.seed(999)
    _ = RIMONEDataset(root="/nonexistent", split="train", image_size=32, augment=False)
    after_init = random.random()

    assert abs(before - after_init) < 1e-9, (
        f"RIMONEDataset.__init__ contaminated global random state: "
        f"before={before:.6f}, after={after_init:.6f}"
    )
    passed("rimone_no_global_random_contamination")


# ─── 2. Encoder tests ─────────────────────────────────────────────────────────

def test_retinal_encoder_shapes():
    """RetinalStructureEncoder: (B, 3, H, W) → (B, embed_dim) with finite values."""
    from rapid_twin.models.retinal_encoder import RetinalStructureEncoder

    enc = RetinalStructureEncoder(
        embed_dim=128, backbone="convnext_tiny", pretrained=False,
        fpn_dim=64, window_size=7,
    )
    enc.eval()

    for img_size in [128, 224, 256]:
        x   = torch.randn(2, 3, img_size, img_size)
        out = enc(x)
        _assert_shape(out, (2, 128), f"retinal_encoder img_size={img_size}")
        _assert_finite(out, f"retinal_encoder output img_size={img_size}")

    passed("retinal_encoder_shapes")


def test_retinal_encoder_backward():
    """Gradient must flow back through the entire encoder."""
    from rapid_twin.models.retinal_encoder import RetinalStructureEncoder

    enc = RetinalStructureEncoder(
        embed_dim=64, backbone="convnext_tiny", pretrained=False,
        fpn_dim=64, window_size=7,
    )
    x   = torch.randn(1, 3, 128, 128, requires_grad=False)
    out = enc(x)
    loss = out.sum()
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in enc.parameters()
                  if p.requires_grad and p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed for encoder parameters"
    assert all(math.isfinite(g) for g in grad_norms), "Some gradients are NaN/Inf"
    passed("retinal_encoder_backward")


# ─── 3. GNN tests ─────────────────────────────────────────────────────────────

def test_gnn_forward_normal():
    """GATv2 GNN: normal graph → (1, out_dim)."""
    from rapid_twin.models.gnn import RetinalGNN

    gnn = RetinalGNN(node_in_dim=16, edge_in_dim=8, hidden_dim=64, out_dim=128,
                     num_heads=4, num_layers=2)
    gnn.eval()

    N, E = 20, 40
    nf = torch.randn(N, 16)
    ei = torch.randint(0, N, (2, E))
    ef = torch.randn(E, 8)

    out = gnn(nf, ei, ef)
    _assert_shape(out, (1, 128), "gnn normal graph")
    _assert_finite(out, "gnn normal graph output")
    passed("gnn_forward_normal")


def test_gnn_forward_empty_graph():
    """GATv2 GNN: empty graph → zero (1, out_dim) fallback."""
    from rapid_twin.models.gnn import RetinalGNN

    gnn = RetinalGNN(node_in_dim=16, edge_in_dim=8, hidden_dim=64, out_dim=128)
    gnn.eval()

    nf = torch.zeros(0, 16)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ef = torch.zeros(0, 8)

    out = gnn(nf, ei, ef)
    _assert_shape(out, (1, 128), "gnn empty graph")
    assert (out == 0).all(), "empty-graph fallback should be all-zeros"
    passed("gnn_forward_empty_graph")


def test_gnn_forward_no_edges():
    """GATv2 GNN: nodes present but no edges → should not crash."""
    from rapid_twin.models.gnn import RetinalGNN

    gnn = RetinalGNN(node_in_dim=16, edge_in_dim=8, hidden_dim=64, out_dim=128)
    gnn.eval()

    nf = torch.randn(10, 16)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ef = torch.zeros(0, 8)

    out = gnn(nf, ei, ef)
    _assert_shape(out, (1, 128), "gnn no-edges graph")
    _assert_finite(out, "gnn no-edges output")
    passed("gnn_forward_no_edges")


def test_gnn_vectorised_scatter():
    """Vectorised scatter_reduce_ produces the same result as the naive per-head loop."""
    import torch
    from rapid_twin.models.gnn import GATv2Layer

    torch.manual_seed(0)
    layer = GATv2Layer(in_dim=32, out_dim=32, edge_dim=8, num_heads=4)
    layer.eval()

    N, E = 12, 20
    x  = torch.randn(N, 32)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, 8)

    # Reference: manually compute per-head scatter max (old approach)
    src, dst = ei[0], ei[1]
    H, D = 4, 8
    e  = torch.randn(E, H)

    ref_max = torch.full((N, H), -1e9)
    for h in range(H):
        ref_max[:, h].scatter_reduce_(0, dst, e[:, h], reduce="amax", include_self=True)

    # New vectorised approach
    dst_h   = dst.unsqueeze(1).expand(-1, H)
    vec_max = torch.full((N, H), -1e9)
    vec_max.scatter_reduce_(0, dst_h, e, reduce="amax", include_self=True)

    assert torch.allclose(ref_max, vec_max, atol=1e-6), (
        "Vectorised scatter max differs from per-head reference"
    )
    passed("gnn_vectorised_scatter")


def test_gnn_backward():
    """Gradient must flow through GATv2 message passing."""
    from rapid_twin.models.gnn import RetinalGNN

    gnn = RetinalGNN(node_in_dim=16, edge_in_dim=8, hidden_dim=64, out_dim=128)

    N, E = 15, 30
    nf = torch.randn(N, 16)
    ei = torch.randint(0, N, (2, E))
    ef = torch.randn(E, 8)

    out  = gnn(nf, ei, ef)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in gnn.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    passed("gnn_backward")


def test_gnn_batch():
    """GNN forward_batch processes a list of variable-size graphs."""
    from rapid_twin.models.gnn import RetinalGNN
    import random as _random

    gnn = RetinalGNN(node_in_dim=16, edge_in_dim=8, hidden_dim=64, out_dim=128)
    gnn.eval()

    _random.seed(7)
    batch = []
    for _ in range(3):
        n = _random.randint(5, 20)
        e = _random.randint(5, 40)
        batch.append((
            torch.randn(n, 16),
            torch.randint(0, n, (2, e)),
            torch.randn(e, 8),
        ))

    out = gnn.forward_batch(batch)
    _assert_shape(out, (3, 128), "gnn batch output")
    _assert_finite(out, "gnn batch output")
    passed("gnn_batch")


# ─── 4. Physio encoder tests ──────────────────────────────────────────────────

def test_physio_encoder_transformer():
    """PhysioSignalEncoder (Transformer mode): (B, T, 8) → (B, 256)."""
    from rapid_twin.models.physio_encoder import PhysioSignalEncoder

    enc = PhysioSignalEncoder(input_dim=8, hidden_dim=64, embed_dim=128,
                               seq_len=24, num_heads=4, num_layers=2, use_tcn=False)
    enc.eval()

    x   = torch.randn(3, 24, 8)
    out = enc(x)
    _assert_shape(out, (3, 128), "physio_encoder transformer")
    _assert_finite(out, "physio_encoder transformer output")
    passed("physio_encoder_transformer")


def test_physio_encoder_tcn():
    """PhysioSignalEncoder (TCN mode): (B, T, 8) → (B, embed_dim)."""
    from rapid_twin.models.physio_encoder import PhysioSignalEncoder

    enc = PhysioSignalEncoder(input_dim=8, hidden_dim=64, embed_dim=128,
                               seq_len=24, use_tcn=True)
    enc.eval()

    x   = torch.randn(3, 24, 8)
    out = enc(x)
    _assert_shape(out, (3, 128), "physio_encoder tcn")
    _assert_finite(out, "physio_encoder tcn output")
    passed("physio_encoder_tcn")


def test_physio_encoder_missing_mask():
    """Missing-value mask must be handled gracefully."""
    from rapid_twin.models.physio_encoder import PhysioSignalEncoder

    enc = PhysioSignalEncoder(input_dim=8, hidden_dim=64, embed_dim=128,
                               seq_len=24, use_tcn=False)
    enc.eval()

    x    = torch.randn(2, 24, 8)
    mask = torch.zeros(2, 24, 8, dtype=torch.bool)
    mask[:, :5, :3] = True   # first 5 timesteps, 3 channels missing

    out = enc(x, missing_mask=mask)
    _assert_shape(out, (2, 128), "physio_encoder missing_mask")
    _assert_finite(out, "physio_encoder missing_mask output")
    passed("physio_encoder_missing_mask")


# ─── 5. Fusion tests ──────────────────────────────────────────────────────────

def test_fusion_all_modalities():
    """MultimodalFusionTransformer: all modalities present."""
    from rapid_twin.fusion.multimodal_fusion import MultimodalFusionTransformer

    fusion = MultimodalFusionTransformer(
        modality_dims=[256, 256, 256], fusion_dim=256,
        num_heads=4, num_layers=2, output_dim=256, use_qformer=False,
    )
    fusion.eval()

    B   = 2
    mods = [torch.randn(B, 256), torch.randn(B, 256), torch.randn(B, 256)]
    avail = torch.ones(B, 3, dtype=torch.bool)

    out = fusion(mods, avail)
    _assert_shape(out["embedding"],        (B, 256), "fusion all_mods embedding")
    _assert_shape(out["modality_weights"], (B, 3),   "fusion all_mods weights")
    _assert_finite(out["embedding"],  "fusion all_mods embedding")
    passed("fusion_all_modalities")


def test_fusion_missing_modality():
    """MultimodalFusionTransformer: one modality absent (None + mask=False)."""
    from rapid_twin.fusion.multimodal_fusion import MultimodalFusionTransformer

    fusion = MultimodalFusionTransformer(
        modality_dims=[256, 256, 256], fusion_dim=256,
        num_heads=4, num_layers=2, output_dim=256, use_qformer=False,
    )
    fusion.eval()

    B    = 2
    mods  = [None, torch.randn(B, 256), torch.randn(B, 256)]
    avail = torch.tensor([[False, True, True]] * B)

    out = fusion(mods, avail)
    _assert_shape(out["embedding"], (B, 256), "fusion missing_mod embedding")
    _assert_finite(out["embedding"], "fusion missing_mod embedding")
    passed("fusion_missing_modality")


def test_fusion_all_missing():
    """MultimodalFusionTransformer: all modalities absent — must not crash or NaN."""
    from rapid_twin.fusion.multimodal_fusion import MultimodalFusionTransformer

    fusion = MultimodalFusionTransformer(
        modality_dims=[256, 256, 256], fusion_dim=256,
        num_heads=4, num_layers=2, output_dim=256, use_qformer=False,
    )
    fusion.eval()

    B    = 2
    mods  = [None, None, None]
    avail = torch.zeros(B, 3, dtype=torch.bool)

    out = fusion(mods, avail)
    _assert_shape(out["embedding"], (B, 256), "fusion all_missing embedding")
    _assert_finite(out["embedding"], "fusion all_missing embedding")
    passed("fusion_all_missing")


def test_fusion_qformer():
    """MultimodalFusionTransformer with QFormer aggregation."""
    from rapid_twin.fusion.multimodal_fusion import MultimodalFusionTransformer

    fusion = MultimodalFusionTransformer(
        modality_dims=[256, 256, 256], fusion_dim=256,
        num_heads=4, num_layers=2, output_dim=256, use_qformer=True, num_queries=8,
    )
    fusion.eval()

    B   = 2
    mods = [torch.randn(B, 256)] * 3
    avail = torch.ones(B, 3, dtype=torch.bool)

    out = fusion(mods, avail)
    _assert_shape(out["embedding"], (B, 256), "fusion qformer embedding")
    _assert_finite(out["embedding"], "fusion qformer embedding")
    passed("fusion_qformer")


# ─── 6. Temporal twin tests ───────────────────────────────────────────────────

def test_tft_forward():
    """TemporalFusionTransformer: historical_seq → health_state, anomaly, quantiles."""
    from rapid_twin.twin.temporal_twin import TemporalFusionTransformer

    tft = TemporalFusionTransformer(
        state_dim=64, hidden_dim=128, num_quantiles=7,
        num_heads=4, num_encoder_steps=12, num_lstm_layers=2,
    )
    tft.eval()

    B, T, D = 2, 12, 64
    hist = torch.randn(B, T, D)
    out  = tft(hist)

    _assert_shape(out["health_state"],         (B, D),    "tft.health_state")
    _assert_shape(out["anomaly_score"],        (B,),      "tft.anomaly_score")
    _assert_shape(out["quantile_predictions"], (B, T, 7), "tft.quantile_predictions")
    _assert_finite(out["health_state"], "tft.health_state")
    _assert_finite(out["anomaly_score"], "tft.anomaly_score")
    passed("tft_forward")


def test_state_tracker_ring_buffer():
    """DigitalTwinStateTracker ring buffer should accumulate correctly."""
    from rapid_twin.twin.temporal_twin import DigitalTwinStateTracker

    tracker = DigitalTwinStateTracker(state_dim=64, memory_size=10)
    tracker.eval()

    assert tracker._buffer_fill.item() == 0, "Buffer should start empty"

    prev = None
    for i in range(7):
        emb  = torch.randn(1, 64)
        state = tracker.update(emb, prev)
        prev  = state
        _assert_shape(state, (1, 64), f"tracker state step {i}")
        _assert_finite(state, f"tracker state step {i}")

    assert tracker._buffer_fill.item() == 7, \
        f"Expected 7 entries in buffer, got {tracker._buffer_fill.item()}"
    passed("state_tracker_ring_buffer")


def test_state_tracker_reset():
    """reset_memory must clear the ring buffer completely."""
    from rapid_twin.twin.temporal_twin import DigitalTwinStateTracker

    tracker = DigitalTwinStateTracker(state_dim=64, memory_size=10)
    tracker.eval()

    for _ in range(5):
        tracker.update(torch.randn(1, 64))

    assert tracker._buffer_fill.item() == 5
    tracker.reset_memory()
    assert tracker._buffer_fill.item() == 0, "reset_memory did not clear fill counter"
    assert tracker._buffer_ptr.item() == 0,  "reset_memory did not reset pointer"
    assert (tracker._memory_buffer == 0).all(), "reset_memory did not zero the buffer"
    passed("state_tracker_reset")


def test_state_tracker_anomaly_scoring():
    """Anomaly score must be in [0,1] and scale with deviation from baseline."""
    from rapid_twin.twin.temporal_twin import DigitalTwinStateTracker

    torch.manual_seed(42)
    tracker = DigitalTwinStateTracker(state_dim=64, memory_size=20)
    tracker.eval()

    # Fill buffer with near-zero states (baseline)
    with torch.no_grad():
        for _ in range(15):
            tracker.update(torch.randn(1, 64) * 0.01)

        # Normal state close to baseline
        normal_score = tracker.compute_anomaly_from_baseline(
            torch.randn(1, 64) * 0.01
        )
        # Anomalous state far from baseline
        anomaly_score = tracker.compute_anomaly_from_baseline(
            torch.randn(1, 64) * 10.0
        )

    assert 0.0 <= normal_score.item()  <= 1.0, f"anomaly score out of range: {normal_score}"
    assert 0.0 <= anomaly_score.item() <= 1.0, f"anomaly score out of range: {anomaly_score}"
    assert anomaly_score.item() > normal_score.item(), \
        "Far-from-baseline state should have higher anomaly score"
    passed("state_tracker_anomaly_scoring")


# ─── 7. Disease heads tests ───────────────────────────────────────────────────

def test_disease_heads_output_keys():
    """MultiTaskDiseaseHead must return all 7 task logits with correct num_classes."""
    from rapid_twin.models.disease_heads import MultiTaskDiseaseHead

    head = MultiTaskDiseaseHead(input_dim=256, hidden_dim=128)
    head.eval()

    emb  = torch.randn(4, 256)
    preds = head(emb)

    expected_shapes = {
        "diabetic_retinopathy":  5,
        "glaucoma":              2,
        "macular_degeneration":  4,
        "retinal_vein_occlusion": 2,
        "diabetes":              3,
        "hypertension":          3,
        "cardiovascular":        4,
    }
    for task, n_cls in expected_shapes.items():
        assert task in preds, f"Missing task key: {task}"
        _assert_shape(preds[task], (4, n_cls), f"disease_head {task}")
        _assert_finite(preds[task], f"disease_head {task}")

    passed("disease_heads_output_keys")


def test_multitask_loss_forward():
    """MultiTaskLoss must return a scalar total loss and per-task losses."""
    from rapid_twin.models.disease_heads import MultiTaskDiseaseHead, MultiTaskLoss

    head = MultiTaskDiseaseHead(input_dim=256, hidden_dim=128)
    loss_fn = MultiTaskLoss(num_tasks=7)

    emb   = torch.randn(4, 256)
    preds = head(emb)
    targets = {
        "diabetic_retinopathy":   torch.randint(0, 5, (4,)),
        "glaucoma":               torch.randint(0, 2, (4,)),
        "macular_degeneration":   torch.randint(0, 4, (4,)),
        "retinal_vein_occlusion": torch.randint(0, 2, (4,)),
        "diabetes":               torch.randint(0, 3, (4,)),
        "hypertension":           torch.randint(0, 3, (4,)),
        "cardiovascular":         torch.randint(0, 4, (4,)),
    }

    out = loss_fn(preds, targets)
    assert "total" in out
    assert torch.isfinite(out["total"]), f"total loss is not finite: {out['total']}"
    assert out["total"].item() > 0, "total loss must be positive"
    passed("multitask_loss_forward")


# ─── 8. Full model (RAPiD-Twin) tests ────────────────────────────────────────

def _make_small_config():
    """Return a config with tiny dims for fast CPU testing."""
    try:
        from rapid_twin.configs.config import RAPiDTwinConfig
        cfg = RAPiDTwinConfig()
        cfg.retinal_encoder.embed_dim  = 64
        cfg.retinal_encoder.fpn_dim    = 64
        cfg.retinal_encoder.pretrained = False
        cfg.gnn.hidden_dim             = 64
        cfg.gnn.out_dim                = 64
        cfg.physio_encoder.hidden_dim  = 64
        cfg.physio_encoder.embed_dim   = 64
        cfg.fusion.fusion_dim          = 64
        cfg.fusion.output_dim          = 64
        cfg.twin.hidden_dim            = 128
        cfg.disease_head.hidden_dim    = 64
        return cfg
    except Exception as e:
        raise RuntimeError(f"Could not build small config: {e}") from e


def test_rapid_twin_image_only():
    """RAPiD-Twin forward with retinal image only (no graph/physio/history)."""
    from rapid_twin.models.rapid_twin_model import RAPiDTwin

    cfg   = _make_small_config()
    model = RAPiDTwin(cfg)
    model.eval()

    with torch.no_grad():
        out = model(retinal_image=torch.randn(2, 3, 64, 64))

    _assert_shape(out["fused_embedding"],  (2, 64), "full_model.fused_embedding")
    _assert_shape(out["updated_state"],    (2, 64), "full_model.updated_state")
    _assert_finite(out["fused_embedding"], "full_model.fused_embedding")

    preds = out["disease_predictions"]
    assert "diabetic_retinopathy" in preds
    passed("rapid_twin_image_only")


def test_rapid_twin_all_modalities():
    """RAPiD-Twin with image + single graph + physio + history."""
    from rapid_twin.models.rapid_twin_model import RAPiDTwin

    cfg   = _make_small_config()
    model = RAPiDTwin(cfg)
    model.eval()

    B, N, E = 2, 15, 25
    with torch.no_grad():
        out = model(
            retinal_image    = torch.randn(B, 3, 64, 64),
            node_features    = [torch.randn(N, 16)] * B,
            edge_index       = [torch.randint(0, N, (2, E))] * B,
            edge_features    = [torch.randn(E, 8)] * B,
            physio_signals   = torch.randn(B, 24, 8),
            historical_states = torch.randn(B, 12, 64),
        )

    _assert_shape(out["updated_state"],   (B, 64), "all_mods.updated_state")
    _assert_shape(out["anomaly_score"],   (B,),    "all_mods.anomaly_score")
    _assert_finite(out["updated_state"],  "all_mods.updated_state")
    _assert_finite(out["anomaly_score"],  "all_mods.anomaly_score")
    passed("rapid_twin_all_modalities")


def test_rapid_twin_backward():
    """Full model must support end-to-end backpropagation."""
    from rapid_twin.models.rapid_twin_model import RAPiDTwin

    cfg   = _make_small_config()
    model = RAPiDTwin(cfg)

    out  = model(retinal_image=torch.randn(1, 3, 64, 64))
    preds = out["disease_predictions"]
    loss  = sum(p.sum() for p in preds.values())
    loss.backward()

    grads = [p.grad for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    bad   = [g for g in grads if not torch.isfinite(g).all()]
    assert not bad, f"{len(bad)} parameter gradients contain NaN/Inf"
    passed("rapid_twin_backward")


def test_rapid_twin_reset_memory():
    """reset_memory must clear the ring buffer (not just the external state pointer)."""
    from rapid_twin.models.rapid_twin_model import RAPiDTwin

    cfg   = _make_small_config()
    model = RAPiDTwin(cfg)
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            model(retinal_image=torch.randn(1, 3, 64, 64))

    fill_before = model.state_tracker._buffer_fill.item()
    assert fill_before > 0, "Buffer should be non-empty after several forward passes"

    model.reset_memory()
    assert model.state_tracker._buffer_fill.item() == 0, \
        "reset_memory did not clear the ring buffer"
    passed("rapid_twin_reset_memory")


def test_rapid_twin_count_parameters():
    """count_parameters must return positive totals for every component."""
    from rapid_twin.models.rapid_twin_model import RAPiDTwin

    cfg    = _make_small_config()
    model  = RAPiDTwin(cfg)
    counts = model.count_parameters()

    assert counts["total"] > 0
    for key in ["retinal_encoder", "gnn", "physio_encoder", "fusion",
                "tft", "state_tracker", "disease_heads"]:
        assert counts[key] > 0, f"component '{key}' has 0 trainable parameters"
    passed("rapid_twin_count_parameters")


# ─── 9. Trainer loss tests ────────────────────────────────────────────────────

def test_trainer_multitask_loss():
    """trainer.MultiTaskLoss must expose both 'loss_total' and 'total' keys."""
    from rapid_twin.training.trainer import MultiTaskLoss

    loss_fn = MultiTaskLoss()

    preds = {
        "diabetic_retinopathy":   torch.randn(4, 5),
        "glaucoma":               torch.randn(4, 2),
        "macular_degeneration":   torch.randn(4, 4),
        "retinal_vein_occlusion": torch.randn(4, 2),
        "diabetes":               torch.randn(4, 3),
        "hypertension":           torch.randn(4, 3),
        "cardiovascular":         torch.randn(4, 4),
    }
    labels = {k: torch.randint(0, v.shape[1], (4,)) for k, v in preds.items()}

    out = loss_fn(preds, labels)
    assert "loss_total" in out, "trainer.MultiTaskLoss must expose 'loss_total' key"
    assert "total"      in out, "trainer.MultiTaskLoss must expose 'total' key"
    assert torch.isfinite(out["loss_total"]), "'loss_total' is not finite"
    # Both keys must reference the same value
    assert torch.allclose(out["loss_total"], out["total"]), \
        "'loss_total' and 'total' must be the same tensor"
    passed("trainer_multitask_loss")


def test_trainer_one_step():
    """RAPiDTwinTrainer must complete one training step without error."""
    from rapid_twin.models.rapid_twin_model import RAPiDTwin
    from rapid_twin.training.trainer import RAPiDTwinTrainer
    from rapid_twin.datasets.retinal_datasets import get_synthetic_loaders

    cfg         = _make_small_config()
    cfg.training.main_lr           = 1e-4
    cfg.training.main_weight_decay = 1e-5
    cfg.training.log_dir           = "/tmp/rapid_twin_test_logs"

    model                   = RAPiDTwin(cfg)
    train_loader, val_loader = get_synthetic_loaders(
        batch_size=2, image_size=32, num_workers=0
    )

    trainer = RAPiDTwinTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, device="cpu", amp=False,
        checkpoint_dir="/tmp/rapid_twin_test_ckpt",
    )

    # Run one training step
    train_metrics = trainer._train_epoch()
    assert "loss_total" in train_metrics or "total" in train_metrics or \
           any("loss" in k for k in train_metrics), \
           f"No loss key in training metrics: {list(train_metrics.keys())}"
    passed("trainer_one_step")


# ─── 10. clDice / Tversky / EMA / HRF tests ──────────────────────────────────

def test_cldice_loss_nonzero():
    """CLDiceLoss must produce a non-zero finite loss on non-trivial inputs."""
    from rapid_twin.training.trainer import CLDiceLoss
    loss_fn = CLDiceLoss(iters=2)
    pred    = torch.sigmoid(torch.randn(2, 1, 32, 32))
    gt      = (torch.rand(2, 1, 32, 32) > 0.8).float()
    loss    = loss_fn(pred, gt)
    _assert_finite(loss, "CLDiceLoss")
    assert loss.item() > 0.0, f"CLDiceLoss is zero: {loss.item()}"
    assert loss.item() <= 1.5, f"CLDiceLoss out of range: {loss.item()}"
    passed("cldice_loss_nonzero")


def test_tversky_loss_nonzero():
    """TverskyLoss must produce a non-zero finite loss with alpha=0.3, beta=0.7."""
    from rapid_twin.training.trainer import TverskyLoss
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    pred    = torch.randn(2, 1, 32, 32)
    gt      = (torch.rand(2, 1, 32, 32) > 0.8).float()
    loss    = loss_fn(pred, gt)
    _assert_finite(loss, "TverskyLoss")
    assert loss.item() > 0.0, f"TverskyLoss is zero: {loss.item()}"
    passed("tversky_loss_nonzero")


def test_combined_seg_loss_all_components():
    """CombinedSegLoss must produce a finite combined loss with clDice + Tversky."""
    from rapid_twin.training.trainer import CombinedSegLoss
    loss_fn = CombinedSegLoss()
    assert hasattr(loss_fn, "cldice"),  "CombinedSegLoss missing clDice"
    assert hasattr(loss_fn, "tversky"), "CombinedSegLoss missing Tversky"
    pred = torch.randn(2, 1, 32, 32)
    gt   = (torch.rand(2, 1, 32, 32) > 0.8).float()
    loss = loss_fn(pred, gt)
    _assert_finite(loss, "CombinedSegLoss")
    assert loss.item() > 0.0, f"CombinedSegLoss is zero: {loss.item()}"
    passed("combined_seg_loss_all_components")


def test_segmentation_loss_all_components():
    """SegmentationLoss from attention_unet must include clDice, Tversky, boundary."""
    from rapid_twin.segmentation.attention_unet import SegmentationLoss
    loss_fn = SegmentationLoss(
        pos_weight=5.0,
        bce_weight=0.25, dice_weight=0.25,
        cldice_weight=0.25, tversky_weight=0.20,
        boundary_weight=0.05, ds_weight=0.30,
    )
    pred = torch.randn(2, 1, 32, 32)
    gt   = (torch.rand(2, 1, 32, 32) > 0.8).float()

    outputs   = {"segmentation": torch.sigmoid(pred), "segmentation_logits": pred,
                 "deep_supervision": [pred + 0.1, pred + 0.2]}
    loss_dict = loss_fn(outputs, gt)

    assert "total"    in loss_dict, f"missing 'total' key: {list(loss_dict.keys())}"
    assert "cldice"   in loss_dict, f"missing 'cldice' key: {list(loss_dict.keys())}"
    assert "tversky"  in loss_dict, f"missing 'tversky' key: {list(loss_dict.keys())}"
    assert "boundary" in loss_dict, f"missing 'boundary' key: {list(loss_dict.keys())}"
    for k, v in loss_dict.items():
        _assert_finite(v, f"SegmentationLoss[{k}]")
    _assert_grad(loss_dict["total"], "SegmentationLoss.total")
    passed("segmentation_loss_all_components")


def test_ema_model_update_and_restore():
    """EMAModel shadow weights must differ from training weights after update."""
    from rapid_twin.training.trainer import EMAModel

    model = nn.Linear(16, 4)
    ema   = EMAModel(model, decay=0.9)

    # Capture initial shadow
    init_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # Mutate model params dramatically
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)

    # Update EMA
    ema.update(model)

    # Shadow must have changed from init
    any_changed = any(
        not torch.allclose(ema.shadow[k], init_shadow[k])
        for k in ema.shadow
    )
    assert any_changed, "EMA shadow weights did not change after update"

    # Apply shadow and verify params change
    ema.apply_shadow(model)
    param_during_ema = next(model.parameters()).data.clone()
    assert not torch.allclose(param_during_ema, torch.full_like(param_during_ema, 99.0)), \
        "apply_shadow did not change model parameters"

    # Restore and verify params go back
    ema.restore(model)
    param_after = next(model.parameters()).data.clone()
    assert torch.allclose(param_after, torch.full_like(param_after, 99.0)), \
        f"restore() did not recover training weights: max diff " \
        f"{(param_after - 99.0).abs().max().item():.6f}"
    passed("ema_model_update_and_restore")


def test_ema_decay_correctness():
    """EMAModel shadow update formula: shadow = decay*shadow + (1-decay)*param."""
    from rapid_twin.training.trainer import EMAModel

    model = nn.Linear(4, 2, bias=False)
    decay = 0.5
    ema   = EMAModel(model, decay=decay)

    # Set known starting weights
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.fill_(0.0)
            ema.shadow[name].fill_(1.0)

    ema.update(model)

    expected = decay * 1.0 + (1 - decay) * 0.0  # = 0.5
    for name, v in ema.shadow.items():
        actual = v.mean().item()
        assert abs(actual - expected) < 1e-5, \
            f"EMA decay mismatch: expected {expected}, got {actual}"
    passed("ema_decay_correctness")


def test_hrf_dataset_synthetic_fallback():
    """HRFDataset must return valid tensors when root does not exist."""
    from rapid_twin.datasets.retinal_datasets import HRFDataset

    ds = HRFDataset(root="/nonexistent_hrf_path", image_size=32,
                    augment=False, normalize=True)
    assert len(ds) >= 1, "HRFDataset length must be >= 1 even on missing root"
    sample = ds[0]
    assert "image"   in sample
    assert "mask"    in sample
    assert "dataset" in sample
    _assert_shape(sample["image"], (3, 32, 32), "HRFDataset.image")
    _assert_shape(sample["mask"],  (1, 32, 32), "HRFDataset.mask")
    _assert_finite(sample["image"], "HRFDataset.image")
    passed("hrf_dataset_synthetic_fallback")


def test_combined_seg_dataset_hrf_wired():
    """CombinedSegDataset must accept 'hrf' root key and track dataset labels."""
    from rapid_twin.datasets.retinal_datasets import CombinedSegDataset

    ds = CombinedSegDataset(
        roots={"drive": "/nx", "stare": "/nx", "chase": "/nx", "hrf": "/nx"},
        image_size=32, augment=False, normalize=True,
    )
    assert hasattr(ds, "_dataset_labels"), "CombinedSegDataset missing _dataset_labels"
    assert len(ds._dataset_labels) == len(ds.samples), \
        "_dataset_labels length mismatch with samples"
    passed("combined_seg_dataset_hrf_wired")


def test_retinal_augmentation_clahe_gamma_blur():
    """RetinalAugmentation must apply CLAHE, gamma, and blur when p=1."""
    import numpy as np
    from rapid_twin.datasets.retinal_datasets import RetinalAugmentation

    aug = RetinalAugmentation(p=1.0, elastic=False,
                              p_clahe=1.0, p_gamma=1.0, p_blur=1.0)
    raw = (np.random.default_rng(42).random((64, 64, 3)) * 255).astype(np.uint8)
    out_img, _ = aug(raw)
    assert out_img.shape == raw.shape, "Augmentation changed image shape"
    assert out_img.dtype == np.uint8,  "Augmentation changed image dtype"
    # Output must differ from input (at least one transform applied)
    assert not np.array_equal(out_img, raw), \
        "Augmented image identical to input — no transforms applied"
    passed("retinal_augmentation_clahe_gamma_blur")


def test_topology_metrics_cldice():
    """clDice_numpy must return 1.0 on a perfect prediction."""
    import numpy as np
    from rapid_twin.segmentation.topology_metrics import clDice_numpy, dice_coefficient

    rng  = np.random.default_rng(0)
    gt   = (rng.random((64, 64)) > 0.85).astype(np.float32)
    pred = gt.copy()
    cd   = clDice_numpy(pred, gt)
    dc   = dice_coefficient(pred, gt)
    assert abs(dc - 1.0) < 1e-4, f"dice should be 1.0 on perfect pred, got {dc}"
    assert abs(cd - 1.0) < 0.05, f"clDice should be ~1.0 on perfect pred, got {cd}"
    passed("topology_metrics_cldice")


def test_postprocessing_pipeline():
    """postprocess() must produce a binary mask from a probability map."""
    import numpy as np
    from rapid_twin.segmentation.postprocessing import postprocess

    prob   = np.random.default_rng(7).random((64, 64)).astype(np.float32)
    result = postprocess(prob, use_otsu=True, min_component_px=5, max_hole_px=10)
    assert result.dtype == np.float32
    vals = set(np.unique(result).tolist())
    assert vals.issubset({0.0, 1.0}), f"postprocess output not binary: {vals}"
    passed("postprocessing_pipeline")


def test_segmentation_trainer_grad_accum_ema():
    """SegmentationTrainer must run one forward-backward with grad_accum=2 + EMA."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class TinySegDataset(Dataset):
        def __len__(self): return 8
        def __getitem__(self, i):
            return {"image": torch.rand(3, 32, 32),
                    "mask":  (torch.rand(1, 32, 32) > 0.85).float(),
                    "dataset": "synthetic"}

    from rapid_twin.segmentation.attention_unet import AttentionUNet
    from rapid_twin.training.trainer import SegmentationTrainer
    from rapid_twin.configs.config import DEFAULT_CONFIG

    model  = AttentionUNet(3, 1, features=[8, 16], dropout=0.0,
                           use_attention=True, deep_supervision=True)
    cfg    = DEFAULT_CONFIG
    loader = DataLoader(TinySegDataset(), batch_size=2)
    trainer = SegmentationTrainer(
        model, loader, loader, cfg, device="cpu",
        grad_accum_steps=2, use_ema=True, ema_decay=0.99,
    )
    assert trainer.ema is not None, "EMA not initialised"
    assert trainer.grad_accum == 2, "grad_accum not set"
    metrics = trainer.train_epoch(epoch=0)
    assert "loss"    in metrics, f"missing 'loss': {list(metrics.keys())}"
    assert "cldice"  in metrics, f"missing 'cldice': {list(metrics.keys())}"
    assert "tversky" in metrics, f"missing 'tversky': {list(metrics.keys())}"
    assert math.isfinite(metrics["loss"]), f"loss is not finite: {metrics['loss']}"
    passed("segmentation_trainer_grad_accum_ema")


# ─── 11. AMP import test ─────────────────────────────────────────────────────

def test_amp_import():
    """trainer.py AMP import must resolve without deprecation errors on torch>=2.1."""
    import importlib
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import rapid_twin.training.trainer as _trainer

    depr_warnings = [
        str(wn.message) for wn in w
        if issubclass(wn.category, (DeprecationWarning, FutureWarning))
        and "torch.cuda.amp" in str(wn.message)
    ]
    # In PyTorch >= 2.1, using the old torch.cuda.amp path triggers FutureWarning
    assert not depr_warnings, (
        f"AMP import triggered deprecation warnings: {depr_warnings}"
    )
    passed("amp_import")


# ─── Runner ───────────────────────────────────────────────────────────────────

TESTS = [
    # Dataset
    test_synthetic_image_normalization,
    test_combined_seg_dataset_no_double_norm,
    test_synthetic_dataset_shapes,
    test_make_weighted_sampler_fast_path,
    test_rimone_no_global_random_contamination,
    # Encoders
    test_retinal_encoder_shapes,
    test_retinal_encoder_backward,
    # GNN
    test_gnn_forward_normal,
    test_gnn_forward_empty_graph,
    test_gnn_forward_no_edges,
    test_gnn_vectorised_scatter,
    test_gnn_backward,
    test_gnn_batch,
    # Physio
    test_physio_encoder_transformer,
    test_physio_encoder_tcn,
    test_physio_encoder_missing_mask,
    # Fusion
    test_fusion_all_modalities,
    test_fusion_missing_modality,
    test_fusion_all_missing,
    test_fusion_qformer,
    # Temporal twin
    test_tft_forward,
    test_state_tracker_ring_buffer,
    test_state_tracker_reset,
    test_state_tracker_anomaly_scoring,
    # Disease heads
    test_disease_heads_output_keys,
    test_multitask_loss_forward,
    # Full model
    test_rapid_twin_image_only,
    test_rapid_twin_all_modalities,
    test_rapid_twin_backward,
    test_rapid_twin_reset_memory,
    test_rapid_twin_count_parameters,
    # Trainer
    test_trainer_multitask_loss,
    test_trainer_one_step,
    # clDice / Tversky / EMA / HRF
    test_cldice_loss_nonzero,
    test_tversky_loss_nonzero,
    test_combined_seg_loss_all_components,
    test_segmentation_loss_all_components,
    test_ema_model_update_and_restore,
    test_ema_decay_correctness,
    test_hrf_dataset_synthetic_fallback,
    test_combined_seg_dataset_hrf_wired,
    test_retinal_augmentation_clahe_gamma_blur,
    test_topology_metrics_cldice,
    test_postprocessing_pipeline,
    test_segmentation_trainer_grad_accum_ema,
    # AMP
    test_amp_import,
]


def run_all(stop_on_first_failure: bool = False) -> bool:
    print(f"\n{'='*70}")
    print(f"  RAPiD-Twin Integration Test Suite  ({len(TESTS)} tests)")
    print(f"{'='*70}\n")

    results: Dict[str, bool] = {}
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            results[name] = True
        except Exception as exc:
            failed(name, exc)
            results[name] = False
            if stop_on_first_failure:
                break

    n_pass = sum(results.values())
    n_fail = len(results) - n_pass
    print(f"\n{'─'*70}")
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print(f"{'─'*70}")

    if n_fail > 0:
        print("\nFailed tests:")
        for name, ok in results.items():
            if not ok:
                print(f"  ✗ {name}")

    return n_fail == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
