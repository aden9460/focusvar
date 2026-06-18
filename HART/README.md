# FocusVAR for HART

This directory contains the HART backend used by **FocusVAR: Focused Classifier-Free Guidance for Efficient Visual Autoregressive Generation**.

## Features

- FastVAR token pruning for HART scales
- SpaceVAR CFG-difference token pruning
- Layerwise cond-only CFG collapse
- KV-cache-aware cond-only continuation
- GenEval / HPSv2 / ImageReward evaluation entry points

## Inference

```bash
python inference.py --model_path /path/to/model \
  --text_model_path /path/to/Qwen2 \
  --prompt "a photo of a corgi wearing sunglasses" \
  --sample_folder_dir ./outputs/hart
```

## Acceleration controls

The HART model exposes `configure_inference_acceleration(...)` for enabling FocusVAR components:

```python
model.configure_inference_acceleration(
    enable_fastvar_compute_merge=True,
    enable_spacevar_compute_merge=False,
    fastvar_ratio_by_scale={48: 0.4, 64: 0.5},
    fastvar_prune_scales={48, 64},
    fastvar_start_layer=1,
    enable_layerwise_cond_only_collapse=True,
    cond_only_start_scale=48,
    cond_only_start_layer=5,
)
```

Recommended starting points:

- conservative: `cond_only_start_scale=48`, `cond_only_start_layer=15`
- faster: `cond_only_start_scale=36`, `cond_only_start_layer=5`
- pruning scales: `{48, 64}` for the default HART scale schedule

## Notes

Model checkpoints, benchmark datasets, and generated images are intentionally not included. Download the required pretrained HART and text encoder weights from their official sources.
