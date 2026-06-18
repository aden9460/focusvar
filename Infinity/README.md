# FocusVAR for Infinity

This directory keeps a minimal Infinity backend for **FocusVAR: Focused Classifier-Free Guidance for Efficient Visual Autoregressive Generation**.

Only the files required for basic text-to-image inference are included. Evaluation scripts, benchmark datasets, notebooks, generated outputs, and checkpoints are intentionally excluded from the open-source package.

## Contents

- `inference.py` - single-prompt inference entry point
- `tools/run_infinity.py` - model loading and generation helpers
- `infinity/` - model and utility modules required by inference
- `requirements.txt` - upstream dependency reference

## Basic inference

Update checkpoint paths in `inference.py`, then run:

```bash
python inference.py
```

Required external assets are not included:

- Infinity transformer checkpoint
- Infinity VAE checkpoint
- FLAN-T5 text encoder checkpoint

Please download them from the official Infinity release or your own trained weights.

## FocusVAR controls

The Infinity model keeps the FocusVAR/FastVAR controls in `infinity/models/infinity.py` and `infinity/models/fastvar_utils.py`, including:

- cached token pruning
- SpaceVAR CFG-difference pruning
- layerwise cond-only collapse

Use the arguments exposed by `gen_one_img(...)` / `autoregressive_infer_cfg(...)` to enable the desired acceleration strategy.
